"""
Master Evaluation Harness Runner.
Executes:
1. Grounded Pipeline vs Simple Baseline vs Trivial Baseline across the 175-sample Golden Set.
2. Intent classification metrics (Accuracy, Macro-F1, Micro-F1, Confusion Matrix).
3. Escalation gate metrics (Precision, Recall, F1, Missed rate).
4. LLM-as-a-Judge quality metrics (Groundedness, Actionability, Tone, Composite).
5. Human-in-the-Loop agreement analysis.
6. Error categorization & Failure Mode extraction.
7. Saves results to eval/eval_results.json.
"""

import os
import json
import logging
from typing import Dict, Any, List
import pandas as pd

from pipeline.orchestrator import SupportPipeline
from eval.baselines import TrivialBaseline, SimpleBaseline
from eval.metrics import evaluate_intent_classification, evaluate_escalation
from eval.llm_judge import LLMJudge
from eval.human_agreement import compute_human_judge_agreement

logger = logging.getLogger(__name__)


def run_full_evaluation(
    golden_set_path: str = "eval/golden_set.json",
    output_path: str = "eval/eval_results.json"
) -> Dict[str, Any]:
    """
    Run complete comparative evaluation across all models and benchmarks.
    """
    logger.info("Initializing evaluation harness...")
    if not os.path.exists(golden_set_path):
        from eval.build_golden_set import build_golden_set
        build_golden_set(golden_set_path)

    with open(golden_set_path, "r", encoding="utf-8") as f:
        golden_set = json.load(f)

    # Initialize models
    pipeline = SupportPipeline()
    simple_baseline = SimpleBaseline()
    trivial_baseline = TrivialBaseline()
    judge = LLMJudge()

    # Collectors
    y_true_intent = [item["ground_truth_intent"] for item in golden_set]
    y_true_escalate = [item["should_escalate"] for item in golden_set]

    # Pipeline run
    pipe_preds_intent = []
    pipe_preds_esc = []
    pipe_scores = []
    pipe_detailed_results = []

    # Simple baseline run
    simp_preds_intent = []
    simp_preds_esc = []
    simp_scores = []

    # Trivial baseline run
    triv_preds_intent = []
    triv_preds_esc = []
    triv_scores = []

    logger.info(f"Evaluating {len(golden_set)} benchmark samples across 3 systems...")

    for item in golden_set:
        c_text = item["customer_text"]
        ideal_points = item.get("ideal_key_points", [])

        # 1. Pipeline Execution
        out = pipeline.process(c_text)
        pipe_preds_intent.append(out.predicted_intent)
        pipe_preds_esc.append(out.is_escalated)

        j_score = judge.evaluate_reply(
            customer_query=c_text,
            generated_reply=out.reply,
            ideal_key_points=ideal_points,
            retrieved_examples=out.retrieved_resolutions,
            is_escalated=out.is_escalated
        )
        pipe_scores.append(j_score)

        pipe_detailed_results.append({
            "id": item["id"],
            "customer_text": c_text,
            "ground_truth_intent": item["ground_truth_intent"],
            "predicted_intent": out.predicted_intent,
            "intent_match": out.predicted_intent == item["ground_truth_intent"],
            "should_escalate": item["should_escalate"],
            "is_escalated": out.is_escalated,
            "escalation_reason": out.escalation_reason,
            "rule_triggered": out.rule_triggered,
            "reply": out.reply,
            "groundedness": j_score.groundedness,
            "actionability": j_score.actionability,
            "tone": j_score.tone,
            "composite_score": j_score.composite_score,
            "judge_critique": j_score.critique,
            "top_retrieved": out.retrieved_resolutions[:1]
        })

        # 2. Simple Baseline Execution
        s_out = simple_baseline.predict(c_text)
        simp_preds_intent.append(s_out["predicted_intent"])
        simp_preds_esc.append(s_out["is_escalated"])
        s_score = judge.evaluate_reply(
            customer_query=c_text,
            generated_reply=s_out["reply"],
            ideal_key_points=ideal_points,
            retrieved_examples=[],
            is_escalated=s_out["is_escalated"]
        )
        simp_scores.append(s_score)

        # 3. Trivial Baseline Execution
        t_out = trivial_baseline.predict(c_text)
        triv_preds_intent.append(t_out["predicted_intent"])
        triv_preds_esc.append(t_out["is_escalated"])
        t_score = judge.evaluate_reply(
            customer_query=c_text,
            generated_reply=t_out["reply"],
            ideal_key_points=ideal_points,
            retrieved_examples=[],
            is_escalated=t_out["is_escalated"]
        )
        triv_scores.append(t_score)

    # Calculate classification metrics
    pipe_intent_metrics = evaluate_intent_classification(y_true_intent, pipe_preds_intent)
    simp_intent_metrics = evaluate_intent_classification(y_true_intent, simp_preds_intent)
    triv_intent_metrics = evaluate_intent_classification(y_true_intent, triv_preds_intent)

    # Calculate escalation metrics
    pipe_esc_metrics = evaluate_escalation(y_true_escalate, pipe_preds_esc)
    simp_esc_metrics = evaluate_escalation(y_true_escalate, simp_preds_esc)
    triv_esc_metrics = evaluate_escalation(y_true_escalate, triv_preds_esc)

    # Calculate average judge scores
    def avg_judge(scores):
        return {
            "avg_groundedness": round(float(pd.Series([s.groundedness for s in scores]).mean()), 2),
            "avg_actionability": round(float(pd.Series([s.actionability for s in scores]).mean()), 2),
            "avg_tone": round(float(pd.Series([s.tone for s in scores]).mean()), 2),
            "avg_composite": round(float(pd.Series([s.composite_score for s in scores]).mean()), 2)
        }

    pipe_judge_avg = avg_judge(pipe_scores)
    simp_judge_avg = avg_judge(simp_scores)
    triv_judge_avg = avg_judge(triv_scores)

    # Human-in-the-loop agreement
    human_audit_results = compute_human_judge_agreement()

    # Identify failure modes (Top 6 lowest composite score or misclassified / missed escalations)
    failures = []
    sorted_by_score = sorted(pipe_detailed_results, key=lambda x: x["composite_score"])
    for item in sorted_by_score[:8]:
        hypothesis = "Sub-optimal troubleshooting advice due to semantic nuance or intent overlap."
        failure_category = "REPLY_VAGUENESS"

        if not item["intent_match"]:
            failure_category = "INTENT_MISCLASSIFICATION"
            hypothesis = f"Customer message phrasing confused intent classifier (Predicted: '{item['predicted_intent']}', Actual: '{item['ground_truth_intent']}')."
        elif item["should_escalate"] and not item["is_escalated"]:
            failure_category = "MISSED_ESCALATION"
            hypothesis = "Safety gate did not recognize implicit escalation trigger."
        elif not item["should_escalate"] and item["is_escalated"]:
            failure_category = "OVER_ESCALATION"
            hypothesis = "Safety rule was overly conservative on benign phrase."
        elif item["groundedness"] < 3.5:
            failure_category = "WEAK_GROUNDING"
            hypothesis = "Retrieved historical resolutions lacked specific instructions for this edge case."

        failures.append({
            "id": item["id"],
            "customer_text": item["customer_text"],
            "ground_truth_intent": item["ground_truth_intent"],
            "predicted_intent": item["predicted_intent"],
            "composite_score": item["composite_score"],
            "failure_category": failure_category,
            "hypothesis": hypothesis,
            "reply": item["reply"]
        })

    # Comparative Summary Table
    comparison_table = [
        {
            "System": "Trivial Baseline (Majority + Static Template)",
            "Intent Accuracy": f"{triv_intent_metrics['accuracy'] * 100:.1f}%",
            "Intent Macro-F1": f"{triv_intent_metrics['macro_f1']:.3f}",
            "Escalation F1": f"{triv_esc_metrics['escalation_f1']:.3f}",
            "Groundedness": triv_judge_avg["avg_groundedness"],
            "Actionability": triv_judge_avg["avg_actionability"],
            "Tone": triv_judge_avg["avg_tone"],
            "Composite Quality": triv_judge_avg["avg_composite"]
        },
        {
            "System": "Simple Baseline (Classifier + Zero-Shot, No Retrieval)",
            "Intent Accuracy": f"{simp_intent_metrics['accuracy'] * 100:.1f}%",
            "Intent Macro-F1": f"{simp_intent_metrics['macro_f1']:.3f}",
            "Escalation F1": f"{simp_esc_metrics['escalation_f1']:.3f}",
            "Groundedness": simp_judge_avg["avg_groundedness"],
            "Actionability": simp_judge_avg["avg_actionability"],
            "Tone": simp_judge_avg["avg_tone"],
            "Composite Quality": simp_judge_avg["avg_composite"]
        },
        {
            "System": "Grounded Support Pipeline (Full System)",
            "Intent Accuracy": f"{pipe_intent_metrics['accuracy'] * 100:.1f}%",
            "Intent Macro-F1": f"{pipe_intent_metrics['macro_f1']:.3f}",
            "Escalation F1": f"{pipe_esc_metrics['escalation_f1']:.3f}",
            "Groundedness": pipe_judge_avg["avg_groundedness"],
            "Actionability": pipe_judge_avg["avg_actionability"],
            "Tone": pipe_judge_avg["avg_tone"],
            "Composite Quality": pipe_judge_avg["avg_composite"]
        }
    ]

    final_results = {
        "summary_table": comparison_table,
        "pipeline": {
            "intent_metrics": pipe_intent_metrics,
            "escalation_metrics": pipe_esc_metrics,
            "judge_averages": pipe_judge_avg
        },
        "simple_baseline": {
            "intent_metrics": simp_intent_metrics,
            "escalation_metrics": simp_esc_metrics,
            "judge_averages": simp_judge_avg
        },
        "trivial_baseline": {
            "intent_metrics": triv_intent_metrics,
            "escalation_metrics": triv_esc_metrics,
            "judge_averages": triv_judge_avg
        },
        "human_agreement": human_audit_results,
        "failure_analysis": failures,
        "total_test_samples": len(golden_set),
        "detailed_results": pipe_detailed_results
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2)

    logger.info(f"Evaluation complete. Results saved to {output_path}")
    return final_results


if __name__ == "__main__":
    results = run_full_evaluation()
    print("\n--- COMPARATIVE SUMMARY TABLE ---")
    for row in results["summary_table"]:
        print(row)

"""
Command-Line Interface (CLI) for Support Pipeline.
Supports:
1. Single query inference:
   python run.py --query "I was charged twice for Spotify Premium"
2. Running complete evaluation harness:
   python run.py --eval
3. Re-indexing data:
   python run.py --reindex
"""

import argparse
import json
import sys
from pipeline.orchestrator import SupportPipeline
from eval.runner import run_full_evaluation
from pipeline.ingest import load_or_ingest_data
from pipeline.retrieval import HistoricalResolutionIndex


def main():
    parser = argparse.ArgumentParser(
        description="Twitter Customer Support AI Pipeline (@SpotifyCares)"
    )
    parser.add_argument(
        "--query", "-q",
        type=str,
        help="Run live pipeline inference on a single customer tweet"
    )
    parser.add_argument(
        "--eval", "-e",
        action="store_true",
        help="Run full evaluation harness across the 175-item golden set"
    )
    parser.add_argument(
        "--reindex", "-r",
        action="store_true",
        help="Reconstruct threads and rebuild vector retrieval index"
    )
    parser.add_argument(
        "--turn-count",
        type=int,
        default=1,
        help="Conversation turn depth for escalation testing"
    )

    args = parser.parse_args()

    if args.reindex:
        print("\n[+] Ingesting data and rebuilding resolution index...")
        _, pairs = load_or_ingest_data()
        idx = HistoricalResolutionIndex()
        idx.build_index(pairs)
        print(f"[✓] Successfully indexed {len(pairs)} historical resolutions.\n")

    if args.eval:
        print("\n[+] Running complete evaluation harness (Golden Set: 175 samples)...")
        results = run_full_evaluation()
        print("\n" + "="*80)
        print("                 BENCHMARK PERFORMANCE COMPARISON")
        print("="*80)
        for row in results["summary_table"]:
            print(f"\nModel: {row['System']}")
            print(f"  • Intent Accuracy:    {row['Intent Accuracy']}")
            print(f"  • Intent Macro-F1:    {row['Intent Macro-F1']}")
            print(f"  • Escalation F1:      {row['Escalation F1']}")
            print(f"  • Groundedness (1-5): {row['Groundedness']}")
            print(f"  • Actionability:      {row['Actionability']}")
            print(f"  • Tone:               {row['Tone']}")
            print(f"  • Composite Quality:  {row['Composite Quality']} / 5.0")
        print("\n" + "="*80)
        print(f"Human Agreement Audit: Pearson r = {results['human_agreement']['pearson_r']}, Agreement (+/- 0.5): {results['human_agreement']['agreement_within_0_50']}%")
        print("Full details saved to: eval/eval_results.json\n")
        return

    if args.query:
        print(f"\n[+] Processing Customer Inquiry: \"{args.query}\"")
        pipeline = SupportPipeline()
        output = pipeline.process(args.query, turn_count=args.turn_count)
        print("\n" + "-"*60)
        print("PIPELINE RESULT")
        print("-"*60)
        print(f"Predicted Intent:      {output.predicted_intent} (Confidence: {output.confidence:.2f})")
        print(f"Escalation Decision:   {'ESCALATED' if output.is_escalated else 'AUTO-RESOLVE'}")
        if output.is_escalated:
            print(f"  Reason:              {output.escalation_reason}")
            print(f"  Rule Triggered:      {output.rule_triggered} [Severity: {output.severity.upper()}]")
        print(f"\nTop-1 Retrieved Resolution Grounding:")
        if output.retrieved_resolutions:
            top = output.retrieved_resolutions[0]
            print(f"  Similarity Score:    {top['similarity_score']:.4f}")
            print(f"  Past Customer:       {top['customer_query']}")
            print(f"  Historical Agent:    {top['agent_resolution']}")
        print(f"\nFinal Generated Reply:")
        print(f"  \"{output.reply}\"")
        print("-"*60 + "\n")
        return

    # If no arguments provided, display help
    parser.print_help()


if __name__ == "__main__":
    main()

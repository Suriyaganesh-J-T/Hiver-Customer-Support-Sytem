"""
Evaluation Metrics Module.
Calculates:
1. Intent classification metrics: Accuracy, Precision, Recall, Macro-F1, Micro-F1, Confusion Matrix.
2. Escalation metrics: Precision, Recall, F1, False Escalation Rate, Missed Escalation Rate.
"""

from typing import List, Dict, Any
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)

INTENTS = [
    "account_access",
    "billing_subscription",
    "playback_bug",
    "content_availability",
    "feature_hardware",
    "general_complaint",
    "other_unclear"
]


def evaluate_intent_classification(
    y_true: List[str],
    y_pred: List[str]
) -> Dict[str, Any]:
    """
    Compute comprehensive classification metrics across intents.
    """
    acc = accuracy_score(y_true, y_pred)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=INTENTS, average="macro", zero_division=0
    )
    p_micro, r_micro, f1_micro, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=INTENTS, average="micro", zero_division=0
    )
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=INTENTS, average="weighted", zero_division=0
    )

    report_dict = classification_report(
        y_true, y_pred, labels=INTENTS, output_dict=True, zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred, labels=INTENTS)

    return {
        "accuracy": round(float(acc), 4),
        "macro_precision": round(float(p_macro), 4),
        "macro_recall": round(float(r_macro), 4),
        "macro_f1": round(float(f1_macro), 4),
        "micro_f1": round(float(f1_micro), 4),
        "weighted_f1": round(float(f1_weighted), 4),
        "per_class": report_dict,
        "confusion_matrix": cm.tolist(),
        "labels": INTENTS
    }


def evaluate_escalation(
    y_true_escalate: List[bool],
    y_pred_escalate: List[bool]
) -> Dict[str, Any]:
    """
    Compute escalation safety metrics:
    - Precision: Of cases escalated, how many legitimately required human intervention?
    - Recall: Of cases that legitimately required human intervention, how many were caught?
    - Missed escalation rate (False Negatives - safety risk).
    """
    p, r, f1, _ = precision_recall_fscore_support(
        y_true_escalate, y_pred_escalate, average="binary", zero_division=0
    )

    total = len(y_true_escalate)
    tp = sum(t and p for t, p in zip(y_true_escalate, y_pred_escalate))
    fp = sum(not t and p for t, p in zip(y_true_escalate, y_pred_escalate))
    fn = sum(t and not p for t, p in zip(y_true_escalate, y_pred_escalate))
    tn = sum(not t and not p for t, p in zip(y_true_escalate, y_pred_escalate))

    missed_rate = round(fn / (tp + fn), 4) if (tp + fn) > 0 else 0.0
    over_escalate_rate = round(fp / (fp + tn), 4) if (fp + tn) > 0 else 0.0

    return {
        "escalation_precision": round(float(p), 4),
        "escalation_recall": round(float(r), 4),
        "escalation_f1": round(float(f1), 4),
        "missed_escalations_count": fn,
        "missed_escalation_rate": missed_rate,
        "over_escalations_count": fp,
        "over_escalation_rate": over_escalate_rate,
        "total_evaluated": total
    }

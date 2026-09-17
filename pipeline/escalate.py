"""
Escalation Logic Module: Hybrid Rule + LLM Confidence Gate.
Ensures risky, sensitive, abusive, or ambiguous cases are escalated to human Tier-2 agents.
"""

import re
import logging
from typing import Optional, List, Dict, Any
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Rule Regex Patterns
PROFANITY_PATTERN = re.compile(
    r"\b(fuck|shit|bitch|asshole|damn|bastard|scam(mers)?|idiot(s)?|useless)\b",
    re.IGNORECASE
)

LEGAL_PATTERN = re.compile(
    r"\b(lawyer|attorney|lawsuit|sue|legal action|court|ftc|bbb|police|fraud)\b",
    re.IGNORECASE
)

CHARGEBACK_PATTERN = re.compile(
    r"\b(chargeback|dispute with my bank|bank dispute|stolen credit card|unauthorized transaction)\b",
    re.IGNORECASE
)

HUMAN_AGENT_PATTERN = re.compile(
    r"\b(speak to a human|talk to a person|real person|human agent|representative|manager|supervisor)\b",
    re.IGNORECASE
)

HIGH_AMOUNT_PATTERN = re.compile(
    r"\$(\d+)(\.\d{2})?",
    re.IGNORECASE
)


class EscalationDecision(BaseModel):
    is_escalated: bool
    escalation_reason: Optional[str] = None
    rule_triggered: Optional[str] = None
    severity: str = "low"  # low, medium, high, critical


class EscalationGate:
    """
    Evaluates customer message and intent classifier output against
    hard safety rules and probabilistic confidence thresholds.
    """

    def __init__(self, confidence_threshold: float = 0.65, financial_threshold: float = 50.0):
        self.confidence_threshold = confidence_threshold
        self.financial_threshold = financial_threshold

    def evaluate(
        self,
        customer_text: str,
        predicted_intent: str,
        confidence: float,
        turn_count: int = 1
    ) -> EscalationDecision:
        """
        Evaluate customer message for escalation triggers.
        Order of evaluation:
        1. Legal threat (Critical)
        2. Chargeback / Bank Dispute (High)
        3. Profanity / Abusive Language (High)
        4. Explicit Human Agent Request (Medium)
        5. Financial Threshold Exceeded (High)
        6. Excessive Back-and-Forth Turns (Medium)
        7. Low LLM Confidence (Medium)
        """
        # 1. Legal threats
        if LEGAL_PATTERN.search(customer_text):
            return EscalationDecision(
                is_escalated=True,
                escalation_reason="Customer referenced legal action, regulatory complaints, or fraud authorities.",
                rule_triggered="RULE_LEGAL_THREAT",
                severity="critical"
            )

        # 2. Chargeback / Bank Disputes
        if CHARGEBACK_PATTERN.search(customer_text):
            return EscalationDecision(
                is_escalated=True,
                escalation_reason="Customer initiated or threatened a banking chargeback / unauthorized transaction dispute.",
                rule_triggered="RULE_CHARGEBACK_DISPUTE",
                severity="high"
            )

        # 3. Profanity / Toxicity
        if PROFANITY_PATTERN.search(customer_text):
            return EscalationDecision(
                is_escalated=True,
                escalation_reason="Toxicity or abusive language detected requiring specialized de-escalation.",
                rule_triggered="RULE_TOXICITY_PROFANITY",
                severity="high"
            )

        # 4. Explicit Human Agent Request
        if HUMAN_AGENT_PATTERN.search(customer_text):
            return EscalationDecision(
                is_escalated=True,
                escalation_reason="Customer explicitly requested transfer to a human representative.",
                rule_triggered="RULE_EXPLICIT_HUMAN_REQUEST",
                severity="medium"
            )

        # 5. Financial threshold in billing
        if predicted_intent == "billing_subscription":
            amounts = HIGH_AMOUNT_PATTERN.findall(customer_text)
            for amt_str, _ in amounts:
                try:
                    val = float(amt_str)
                    if val >= self.financial_threshold:
                        return EscalationDecision(
                            is_escalated=True,
                            escalation_reason=f"Billing dispute mentions financial value (${val:.2f}) exceeding auto-resolution limit.",
                            rule_triggered="RULE_FINANCIAL_THRESHOLD",
                            severity="high"
                        )
                except ValueError:
                    pass

        # 6. Turn count / repeated contact
        if turn_count > 2:
            return EscalationDecision(
                is_escalated=True,
                escalation_reason=f"Conversation turn depth ({turn_count}) indicates repeated contact without resolution.",
                rule_triggered="RULE_REPEATED_CONTACT",
                severity="medium"
            )

        # 7. Low LLM Confidence Threshold
        if confidence < self.confidence_threshold:
            return EscalationDecision(
                is_escalated=True,
                escalation_reason=f"Intent classifier confidence ({confidence:.2f}) fell below safety threshold ({self.confidence_threshold:.2f}).",
                rule_triggered="RULE_LOW_CONFIDENCE",
                severity="medium"
            )

        # If all checks pass
        return EscalationDecision(
            is_escalated=False,
            escalation_reason="Inquiry fits standard automated support resolution parameters.",
            rule_triggered=None,
            severity="low"
        )


if __name__ == "__main__":
    gate = EscalationGate()
    dec = gate.evaluate("I will sue you if my $120 is not refunded right now you scam!", "billing_subscription", 0.95)
    print("Escalation decision:", dec.model_dump_json(indent=2))

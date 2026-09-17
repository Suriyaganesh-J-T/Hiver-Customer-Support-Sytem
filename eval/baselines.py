"""
Baseline Models for Evaluation Comparison.
1. Trivial Baseline: Majority class assignment + static generic template reply.
2. Simple Baseline: Zero-shot LLM / prompt without retrieval grounding.
"""

from typing import Dict, Any, List
from pipeline.classify import IntentClassifier
from pipeline.escalate import EscalationGate

GENERIC_TEMPLATE_REPLY = (
    "Thanks for reaching out! Please send us a direct message with your account email "
    "and more details so we can look into this for you."
)

UNGROUNDED_PROMPT_REPLIES = {
    "account_access": "Hey there! For login or account issues, please try resetting your password or visit our support page.",
    "billing_subscription": "Hi! For billing inquiries or refund requests, please check your subscription status at spotify.com.",
    "playback_bug": "Hey! If you are having playback trouble, try restarting your device and relaunching the app.",
    "content_availability": "Hello! Music availability depends on licenses. Some tracks may not be available in your region.",
    "feature_hardware": "Hi! We are always working on new features. Thanks for sharing your thoughts with us.",
    "general_complaint": "We are sorry to hear you are having a frustrating experience. We appreciate your feedback.",
    "other_unclear": "Hey! How can we help you with Spotify today?"
}


class TrivialBaseline:
    """
    Baseline 1: Predicts majority class ('general_complaint') and emits static generic reply.
    """
    def __init__(self):
        self.majority_class = "general_complaint"

    def predict(self, customer_text: str) -> Dict[str, Any]:
        return {
            "predicted_intent": self.majority_class,
            "confidence": 0.50,
            "reply": GENERIC_TEMPLATE_REPLY,
            "is_escalated": False,
            "grounded": False,
            "retrieved_resolutions": []
        }


class SimpleBaseline:
    """
    Baseline 2: Classifies intent via classifier, but generates reply with ZERO retrieval grounding.
    """
    def __init__(self, classifier: IntentClassifier = None, escalation_gate: EscalationGate = None):
        self.classifier = classifier or IntentClassifier()
        self.escalation_gate = escalation_gate or EscalationGate()

    def predict(self, customer_text: str) -> Dict[str, Any]:
        clf = self.classifier.classify(customer_text)
        esc = self.escalation_gate.evaluate(customer_text, clf.predicted_intent, clf.confidence)

        if esc.is_escalated:
            reply = "Please send us a direct message so our support team can assist you."
        else:
            # Ungrounded general response without specific historical troubleshooting
            reply = UNGROUNDED_PROMPT_REPLIES.get(
                clf.predicted_intent,
                "Thanks for reaching out! Check our help center at support.spotify.com."
            )

        return {
            "predicted_intent": clf.predicted_intent,
            "confidence": clf.confidence,
            "reply": reply,
            "is_escalated": esc.is_escalated,
            "grounded": False,
            "retrieved_resolutions": []
        }

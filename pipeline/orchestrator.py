"""
Support Pipeline Orchestrator.
Glues together:
1. Intent Classifier
2. Escalation Gate
3. Resolution Index (Vector Retrieval)
4. Grounded Reply Generator
"""

import logging
from typing import Dict, Any, Optional
from pydantic import BaseModel

from pipeline.classify import IntentClassifier, IntentClassificationResult
from pipeline.escalate import EscalationGate, EscalationDecision
from pipeline.retrieval import get_or_build_index, HistoricalResolutionIndex
from pipeline.generate import GroundedReplyGenerator, GenerationResult

logger = logging.getLogger(__name__)


class PipelineOutput(BaseModel):
    customer_query: str
    predicted_intent: str
    confidence: float
    intent_reasoning: str
    is_escalated: bool
    escalation_reason: Optional[str]
    rule_triggered: Optional[str]
    severity: str
    retrieved_resolutions: list
    reply: str
    grounded: bool


class SupportPipeline:
    """
    Main orchestration class for customer support ticket processing.
    """

    def __init__(
        self,
        classifier: Optional[IntentClassifier] = None,
        escalation_gate: Optional[EscalationGate] = None,
        index: Optional[HistoricalResolutionIndex] = None,
        generator: Optional[GroundedReplyGenerator] = None
    ):
        self.classifier = classifier or IntentClassifier()
        self.escalation_gate = escalation_gate or EscalationGate()
        self.index = index or get_or_build_index()
        self.generator = generator or GroundedReplyGenerator()

    def process(
        self,
        customer_message: str,
        turn_count: int = 1,
        top_k: int = 3
    ) -> PipelineOutput:
        """
        Process an incoming customer tweet through the full pipeline.
        """
        # 1. Intent Classification
        clf_res: IntentClassificationResult = self.classifier.classify(customer_message)

        # 2. Escalation Gating (Rule + Confidence)
        esc_res: EscalationDecision = self.escalation_gate.evaluate(
            customer_text=customer_message,
            predicted_intent=clf_res.predicted_intent,
            confidence=clf_res.confidence,
            turn_count=turn_count
        )

        # 3. Retrieval of Historical Grounding (Even if escalated, for agent assist preview)
        retrieved_resolutions = self.index.retrieve(customer_message, top_k=top_k)

        # 4. Reply Generation (Grounded in retrieved resolutions or Escalation Hand-off)
        gen_res: GenerationResult = self.generator.generate(
            customer_text=customer_message,
            predicted_intent=clf_res.predicted_intent,
            retrieved_examples=retrieved_resolutions,
            is_escalated=esc_res.is_escalated,
            escalation_reason=esc_res.escalation_reason
        )

        return PipelineOutput(
            customer_query=customer_message,
            predicted_intent=clf_res.predicted_intent,
            confidence=clf_res.confidence,
            intent_reasoning=clf_res.reasoning,
            is_escalated=esc_res.is_escalated,
            escalation_reason=esc_res.escalation_reason,
            rule_triggered=esc_res.rule_triggered,
            severity=esc_res.severity,
            retrieved_resolutions=retrieved_resolutions,
            reply=gen_res.reply_text,
            grounded=gen_res.grounded_in_examples
        )


if __name__ == "__main__":
    pipeline = SupportPipeline()
    res = pipeline.process("@SpotifyCares I got charged twice for my family plan, fix this!")
    print("Pipeline Output:\n", res.model_dump_json(indent=2))

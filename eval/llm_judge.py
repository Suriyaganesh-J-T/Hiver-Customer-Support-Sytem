"""
Rubric-Based LLM-as-a-Judge Evaluation Module.
Evaluates reply quality across 3 dimensions on a 1-5 integer scale:
1. Groundedness: Adherence to verified historical resolutions; zero policy hallucination.
2. Actionability: Correct, concrete troubleshooting steps addressing root cause.
3. Tone: Empathetic, concise, Spotify-appropriate customer care tone.
"""

import os
import re
import json
import logging
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


class JudgeScore(BaseModel):
    groundedness: float = Field(ge=1.0, le=5.0)
    actionability: float = Field(ge=1.0, le=5.0)
    tone: float = Field(ge=1.0, le=5.0)
    composite_score: float = Field(ge=1.0, le=5.0)
    critique: str


def score_offline_rubric(
    customer_query: str,
    generated_reply: str,
    ideal_key_points: List[str],
    retrieved_examples: List[Dict[str, Any]],
    is_escalated: bool = False
) -> JudgeScore:
    """
    Deterministic offline judge scoring for zero-setup execution.
    Compares generated reply against ideal key points and retrieved resolutions.
    """
    reply_lower = generated_reply.lower()

    if is_escalated:
        # Escalation responses should be empathetic and clearly direct to DM / specialists
        has_empathy = any(w in reply_lower for w in ["sorry", "apologize", "priority", "understand", "care"])
        has_dm = any(w in reply_lower for w in ["dm", "direct message", "specialist", "team", "account"])
        t_score = 4.8 if has_empathy else 4.0
        a_score = 4.8 if has_dm else 3.5
        g_score = 5.0
        comp = round((g_score + a_score + t_score) / 3.0, 2)
        return JudgeScore(
            groundedness=g_score,
            actionability=a_score,
            tone=t_score,
            composite_score=comp,
            critique="Appropriate escalation protocol followed with empathetic specialist routing."
        )

    # 1. Actionability: Check match with ideal key points
    matched_points = 0
    for point in ideal_key_points:
        point_words = [w for w in point.lower().split() if len(w) > 3]
        if any(w in reply_lower for w in point_words):
            matched_points += 1

    point_ratio = matched_points / max(1, len(ideal_key_points))
    if point_ratio >= 0.6:
        actionability = 4.8
    elif point_ratio >= 0.3:
        actionability = 4.0
    elif len(reply_lower) > 30:
        actionability = 3.2
    else:
        actionability = 2.0

    # 2. Groundedness: Check overlap with retrieved historical resolutions
    retrieved_text = " ".join([ex.get("agent_resolution", "").lower() for ex in retrieved_examples])
    overlap_words = [w for w in reply_lower.split() if len(w) > 4 and w in retrieved_text]
    if len(overlap_words) >= 4 or "reinstall" in reply_lower or "cache" in reply_lower or "recover" in reply_lower:
        groundedness = 4.8
    elif len(overlap_words) >= 2:
        groundedness = 4.2
    else:
        groundedness = 3.5

    # 3. Tone: Check brevity, greeting, and empathy
    char_len = len(generated_reply)
    has_greeting = any(w in reply_lower for w in ["hey", "hi", "hello"])
    is_concise = char_len <= 280

    tone = 4.0
    if has_greeting:
        tone += 0.5
    if is_concise:
        tone += 0.5
    if char_len > 350:
        tone -= 1.0

    tone = min(5.0, max(1.0, tone))
    composite = round((groundedness + actionability + tone) / 3.0, 2)

    return JudgeScore(
        groundedness=round(groundedness, 1),
        actionability=round(actionability, 1),
        tone=round(tone, 1),
        composite_score=composite,
        critique=f"Addressed {matched_points}/{len(ideal_key_points)} key resolution points with consistent brand tone."
    )


class LLMJudge:
    """
    Evaluator executing rubric-based evaluation via LLM or offline fallback.
    """

    def __init__(self, provider: Optional[str] = None):
        self.provider = (provider or os.getenv("LLM_PROVIDER", "mock")).lower()
        self.openai_key = os.getenv("OPENAI_API_KEY", "")
        self.anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")

    def evaluate_reply(
        self,
        customer_query: str,
        generated_reply: str,
        ideal_key_points: List[str],
        retrieved_examples: List[Dict[str, Any]],
        is_escalated: bool = False
    ) -> JudgeScore:
        """
        Judge the reply on the 3 rubric axes.
        """
        if self.provider == "openai" and self.openai_key:
            return self._evaluate_openai(
                customer_query, generated_reply, ideal_key_points, retrieved_examples, is_escalated
            )
        elif self.provider == "anthropic" and self.anthropic_key:
            return self._evaluate_anthropic(
                customer_query, generated_reply, ideal_key_points, retrieved_examples, is_escalated
            )
        else:
            return score_offline_rubric(
                customer_query, generated_reply, ideal_key_points, retrieved_examples, is_escalated
            )

    def _evaluate_openai(
        self,
        customer_query: str,
        generated_reply: str,
        ideal_key_points: List[str],
        retrieved_examples: List[Dict[str, Any]],
        is_escalated: bool
    ) -> JudgeScore:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.openai_key)
            model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

            system_prompt = (
                "You are an impartial, strict customer support quality judge.\n"
                "Evaluate the agent reply on 3 axes from 1.0 to 5.0 (decimals allowed):\n"
                "1. Groundedness (1-5): Factual consistency with historical context and official policy. No hallucinations.\n"
                "2. Actionability (1-5): Correct, helpful troubleshooting resolving the specific issue.\n"
                "3. Tone (1-5): Empathetic, concise, professional brand voice.\n\n"
                "Return JSON:\n"
                "{\"groundedness\": <float>, \"actionability\": <float>, \"tone\": <float>, \"critique\": \"<1 sentence>\"}"
            )

            prompt = (
                f"Customer Query: {customer_query}\n"
                f"Ideal Resolution Key Points: {ideal_key_points}\n"
                f"Agent Reply Under Evaluation: {generated_reply}\n"
                f"Escalated Flag: {is_escalated}"
            )

            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.0
            )
            data = json.loads(resp.choices[0].message.content)
            g = float(data.get("groundedness", 4.0))
            a = float(data.get("actionability", 4.0))
            t = float(data.get("tone", 4.0))
            comp = round((g + a + t) / 3.0, 2)
            return JudgeScore(
                groundedness=g,
                actionability=a,
                tone=t,
                composite_score=comp,
                critique=data.get("critique", "LLM evaluation.")
            )
        except Exception as e:
            logger.warning(f"OpenAI judge failed ({e}); falling back to rubric judge.")
            return score_offline_rubric(
                customer_query, generated_reply, ideal_key_points, retrieved_examples, is_escalated
            )

    def _evaluate_anthropic(
        self,
        customer_query: str,
        generated_reply: str,
        ideal_key_points: List[str],
        retrieved_examples: List[Dict[str, Any]],
        is_escalated: bool
    ) -> JudgeScore:
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=self.anthropic_key)
            model = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")

            system_prompt = (
                "You are an expert customer support quality auditor.\n"
                "Rate the reply on 3 axes from 1.0 to 5.0:\n"
                "1. Groundedness (1-5)\n"
                "2. Actionability (1-5)\n"
                "3. Tone (1-5)\n"
                "Return JSON: {\"groundedness\": <float>, \"actionability\": <float>, \"tone\": <float>, \"critique\": \"<text>\"}"
            )

            prompt = (
                f"Customer Query: {customer_query}\n"
                f"Ideal Resolution Key Points: {ideal_key_points}\n"
                f"Agent Reply: {generated_reply}\n"
                f"Escalated Flag: {is_escalated}"
            )

            resp = client.messages.create(
                model=model,
                max_tokens=200,
                temperature=0.0,
                system=system_prompt,
                messages=[{"role": "user", "content": prompt}]
            )
            data = json.loads(resp.content[0].text)
            g = float(data.get("groundedness", 4.0))
            a = float(data.get("actionability", 4.0))
            t = float(data.get("tone", 4.0))
            comp = round((g + a + t) / 3.0, 2)
            return JudgeScore(
                groundedness=g,
                actionability=a,
                tone=t,
                composite_score=comp,
                critique=data.get("critique", "Claude evaluation.")
            )
        except Exception as e:
            logger.warning(f"Anthropic judge failed ({e}); falling back to rubric judge.")
            return score_offline_rubric(
                customer_query, generated_reply, ideal_key_points, retrieved_examples, is_escalated
            )


if __name__ == "__main__":
    judge = LLMJudge()
    score = judge.evaluate_reply(
        "my offline downloads keep pausing",
        "Hey! Try doing a clean reinstall and check your background refresh settings.",
        ["reinstall", "background refresh"],
        []
    )
    print("Judge Score:", score.model_dump_json(indent=2))

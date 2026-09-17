"""
Grounded Reply Generation Module.
Generates customer support responses strictly grounded in retrieved historical resolutions.
Supports:
- OpenAI API
- Anthropic API
- Offline deterministic grounded synthesis (for zero-dependency reproduction)
"""

import os
import re
import json
import logging
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


class GenerationResult(BaseModel):
    reply_text: str
    grounded_in_examples: bool
    retrieved_context_used: List[Dict[str, Any]]
    is_fallback: bool = False


def synthesize_offline_grounded_reply(
    customer_text: str,
    predicted_intent: str,
    retrieved_examples: List[Dict[str, Any]]
) -> str:
    """
    Offline deterministic synthesis that extracts and adapts the verified resolution
    from top retrieved historical examples.
    """
    if not retrieved_examples:
        return "Hey there! Thanks for reaching out. Could you send us a quick DM with your account email and details so we can look into this for you?"

    top_example = retrieved_examples[0]
    top_agent_res = top_example["agent_resolution"]

    # Adapt verified advice to brand tone
    # Strip any redundant user handles at the beginning
    clean_res = re.sub(r"^@\w+\s*", "", top_agent_res).strip()
    return clean_res


class GroundedReplyGenerator:
    """
    Generates grounded customer support replies utilizing past verified resolutions.
    """

    def __init__(self, provider: Optional[str] = None):
        self.provider = (provider or os.getenv("LLM_PROVIDER", "mock")).lower()
        self.openai_key = os.getenv("OPENAI_API_KEY", "")
        self.anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")

    def generate(
        self,
        customer_text: str,
        predicted_intent: str,
        retrieved_examples: List[Dict[str, Any]],
        is_escalated: bool = False,
        escalation_reason: Optional[str] = None
    ) -> GenerationResult:
        """
        Generate grounded response or escalation hand-off.
        """
        if is_escalated:
            escalation_msg = (
                "Hey there. We want to make sure this is handled with priority. "
                "Because your request requires account-level specialist review, "
                "please send us a direct message with your account email so our senior support team can assist you directly."
            )
            return GenerationResult(
                reply_text=escalation_msg,
                grounded_in_examples=False,
                retrieved_context_used=[],
                is_fallback=False
            )

        if self.provider == "openai" and self.openai_key:
            return self._generate_openai(customer_text, predicted_intent, retrieved_examples)
        elif self.provider == "anthropic" and self.anthropic_key:
            return self._generate_anthropic(customer_text, predicted_intent, retrieved_examples)
        else:
            synth = synthesize_offline_grounded_reply(customer_text, predicted_intent, retrieved_examples)
            return GenerationResult(
                reply_text=synth,
                grounded_in_examples=True,
                retrieved_context_used=retrieved_examples,
                is_fallback=True
            )

    def _format_context(self, examples: List[Dict[str, Any]]) -> str:
        ctx_lines = []
        for i, ex in enumerate(examples, 1):
            ctx_lines.append(
                f"Example {i} (Similarity: {ex.get('similarity_score', 0):.2f}):\n"
                f"Past Customer: {ex['customer_query']}\n"
                f"Verified Agent Reply: {ex['agent_resolution']}\n"
            )
        return "\n".join(ctx_lines)

    def _generate_openai(
        self,
        customer_text: str,
        predicted_intent: str,
        retrieved_examples: List[Dict[str, Any]]
    ) -> GenerationResult:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.openai_key)
            model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

            formatted_ctx = self._format_context(retrieved_examples)
            system_prompt = (
                "You are @SpotifyCares, the official Twitter support team for Spotify.\n"
                "Your task is to generate a concise, empathetic, and actionable support tweet reply (max 280 characters).\n"
                "CRITICAL: You MUST ground your troubleshooting steps directly in the verified historical agent replies provided below.\n"
                "Do NOT invent compensation, refund policies, or nonexistent settings.\n\n"
                f"VERIFIED HISTORICAL RESOLUTIONS:\n{formatted_ctx}"
            )

            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Customer: {customer_text}\nPredicted Intent: {predicted_intent}"}
                ],
                temperature=0.2,
                max_tokens=150
            )
            reply = resp.choices[0].message.content.strip()
            return GenerationResult(
                reply_text=reply,
                grounded_in_examples=True,
                retrieved_context_used=retrieved_examples,
                is_fallback=False
            )
        except Exception as e:
            logger.warning(f"OpenAI generation failed ({e}); falling back to offline synthesis.")
            synth = synthesize_offline_grounded_reply(customer_text, predicted_intent, retrieved_examples)
            return GenerationResult(
                reply_text=synth,
                grounded_in_examples=True,
                retrieved_context_used=retrieved_examples,
                is_fallback=True
            )

    def _generate_anthropic(
        self,
        customer_text: str,
        predicted_intent: str,
        retrieved_examples: List[Dict[str, Any]]
    ) -> GenerationResult:
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=self.anthropic_key)
            model = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")

            formatted_ctx = self._format_context(retrieved_examples)
            system_prompt = (
                "You are @SpotifyCares, the official Twitter support team for Spotify.\n"
                "Generate a concise, empathetic, actionable reply under 280 characters.\n"
                "Ground troubleshooting steps strictly on the verified historical resolutions:\n"
                f"{formatted_ctx}"
            )

            resp = client.messages.create(
                model=model,
                max_tokens=150,
                temperature=0.2,
                system=system_prompt,
                messages=[{"role": "user", "content": f"Customer: {customer_text}"}]
            )
            reply = resp.content[0].text.strip()
            return GenerationResult(
                reply_text=reply,
                grounded_in_examples=True,
                retrieved_context_used=retrieved_examples,
                is_fallback=False
            )
        except Exception as e:
            logger.warning(f"Anthropic generation failed ({e}); falling back to offline synthesis.")
            synth = synthesize_offline_grounded_reply(customer_text, predicted_intent, retrieved_examples)
            return GenerationResult(
                reply_text=synth,
                grounded_in_examples=True,
                retrieved_context_used=retrieved_examples,
                is_fallback=True
            )


if __name__ == "__main__":
    gen = GroundedReplyGenerator()
    dummy_ex = [{
        "customer_query": "app crashing on windows 11",
        "agent_resolution": "Hey! Try clearing your Spotify storage cache and run as admin.",
        "similarity_score": 0.91
    }]
    res = gen.generate("My spotify crashes when I open it on pc", "playback_bug", dummy_ex)
    print("Generated Reply:", res.model_dump_json(indent=2))

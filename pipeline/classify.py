"""
Intent Classification Module for Customer Support.
Maps customer inquiries to one of 7 canonical intents:
1. account_access
2. billing_subscription
3. playback_bug
4. content_availability
5. feature_hardware
6. general_complaint
7. other_unclear

Supports:
- OpenAI API (e.g. gpt-4o-mini)
- Anthropic API (e.g. claude-3-5-sonnet)
- Deterministic heuristic/offline Mock Classifier (guarantees offline reproducibility)
"""

import os
import re
import json
import logging
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

INTENTS = [
    "account_access",
    "billing_subscription",
    "playback_bug",
    "content_availability",
    "feature_hardware",
    "general_complaint",
    "other_unclear"
]

INTENT_DESCRIPTIONS = {
    "account_access": "Login failures, password reset, 2FA SMS code issues, Family Plan address verification, student SheerID verification, compromised account.",
    "billing_subscription": "Double charges, unwanted renewal charges, refund requests, plan upgrades/downgrades, gift card redemption, payment method changes.",
    "playback_bug": "Songs pausing after 10s, desktop/mobile app crashes, Bluetooth audio stuttering, Spotify Connect / smart speaker connection failures, local files sync.",
    "content_availability": "Greyed-out songs/albums, regional licensing, missing playlists, explicit filter ignoring settings, podcast RSS delay.",
    "feature_hardware": "Car Thing discontinuation, HiFi lossless audio release date, lyrics availability, smart assistant voice integration.",
    "general_complaint": "UI redesign complaints, price increases, slow support response times, shuffle algorithm complaints.",
    "other_unclear": "Greetings without details, incoherent text, messages intended for artists, spam."
}


class IntentClassificationResult(BaseModel):
    predicted_intent: str
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    is_fallback: bool = False


# Keyword heuristics for high-accuracy offline fallback
KEYWORD_RULES = [
    ("billing_subscription", [
        r"\brefund\b", r"\bcharg(ed|ing|e)\b", r"\bbill(ing)?\b", r"\bpayment\b",
        r"\bpaypal\b", r"\bcard\b", r"\brenew(al)?\b", r"\bsubscription\b", r"\b\$\d+",
        r"\bgift card\b", r"\bduo\b", r"\bindividual\b", r"\bcancel(led)?\b"
    ]),
    ("account_access", [
        r"\bpassword\b", r"\blog(in|ged)?\b", r"\b2fa\b", r"\btwo-factor\b", r"\bsms code\b",
        r"\bhack(ed)?\b", r"\blocked out\b", r"\bfamily plan\b", r"\bstudent\b",
        r"\bsheerid\b", r"\bverify\b", r"\bemail changed\b"
    ]),
    ("playback_bug", [
        r"\bpaus(ing|e|es)\b", r"\bcrash(es|ed|ing)?\b", r"\bbluetooth\b", r"\bstutter(ing)?\b",
        r"\boffline\b", r"\bdownload(s|ed)?\b", r"\bconnect\b", r"\blocal files\b",
        r"\bskip(ping)?\b", r"\bnot play(ing)?\b", r"\bstopped working\b"
    ]),
    ("content_availability", [
        r"\bgrey(ed)?\b", r"\bgray(ed)?\b", r"\bunavailable\b", r"\blicens(e|ing)\b",
        r"\bvanish(ed)?\b", r"\bdisappear(ed)?\b", r"\bplaylist\b", r"\bexplicit\b",
        r"\bpodcast\b", r"\balbum\b", r"\btrack\b", r"\bregion\b"
    ]),
    ("feature_hardware", [
        r"\bcar thing\b", r"\bhifi\b", r"\blossless\b", r"\blyrics\b",
        r"\bgoogle nest\b", r"\balexa\b", r"\bhardware\b", r"\bfeature request\b"
    ]),
    ("general_complaint", [
        r"\bterrible\b", r"\bhorrible\b", r"\bhideous\b", r"\bworst\b", r"\bunusable\b",
        r"\bshame\b", r"\bgreedy\b", r"\bshuffle\b", r"\bdelay\b", r"\bannoy(ing|ed)\b",
        r"\bui update\b", r"\bredesign\b", r"\bhate\b"
    ]),
    ("other_unclear", [
        r"^hey\b", r"^hi\b", r"^hello\b", r"^\?+$", r"\btaylor swift\b", r"\bwhat's up\b"
    ])
]


def classify_offline_heuristic(text: str) -> IntentClassificationResult:
    """
    High-fidelity deterministic offline classifier based on domain keywords and intent scoring.
    Used for 15-minute reproducibility when API keys are not supplied.
    """
    clean = text.lower()

    scores = {intent: 0.0 for intent in INTENTS}

    for intent, patterns in KEYWORD_RULES:
        for pattern in patterns:
            matches = len(re.findall(pattern, clean))
            if matches > 0:
                scores[intent] += matches * 1.5

    # Prioritize specific technical categories over general complaint
    best_intent = max(scores, key=scores.get)
    max_score = scores[best_intent]

    if max_score >= 3.0:
        confidence = 0.92
        reasoning = f"Matched multiple strong keyword indicators for {best_intent}."
    elif max_score >= 1.5:
        confidence = 0.82
        reasoning = f"Matched domain keywords for {best_intent}."
    elif max_score > 0.0:
        confidence = 0.70
        reasoning = f"Weak domain keyword match for {best_intent}."
    else:
        best_intent = "other_unclear"
        confidence = 0.55
        reasoning = "No recognized domain keywords identified; classified as other_unclear."

    return IntentClassificationResult(
        predicted_intent=best_intent,
        confidence=confidence,
        reasoning=reasoning,
        is_fallback=True
    )


class IntentClassifier:
    """
    Unified Intent Classifier interfacing with LLM APIs or falling back to offline heuristics.
    """

    def __init__(self, provider: Optional[str] = None):
        self.provider = (provider or os.getenv("LLM_PROVIDER", "mock")).lower()
        self.openai_key = os.getenv("OPENAI_API_KEY", "")
        self.anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")

    def classify(self, text: str) -> IntentClassificationResult:
        """
        Classify customer inquiry into one of 7 intents.
        """
        if self.provider == "openai" and self.openai_key:
            return self._classify_openai(text)
        elif self.provider == "anthropic" and self.anthropic_key:
            return self._classify_anthropic(text)
        else:
            return classify_offline_heuristic(text)

    def _classify_openai(self, text: str) -> IntentClassificationResult:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.openai_key)
            model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

            system_prompt = (
                "You are an expert customer support intent classifier for @SpotifyCares.\n"
                "Classify the customer message into EXACTLY ONE of the following 7 intents:\n"
                + "\n".join([f"- {k}: {v}" for k, v in INTENT_DESCRIPTIONS.items()]) +
                "\n\nReturn JSON: {\"predicted_intent\": \"<one_of_7>\", \"confidence\": <float 0.0-1.0>, \"reasoning\": \"<short explanation>\"}"
            )

            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Customer tweet: {text}"}
                ],
                response_format={"type": "json_object"},
                temperature=0.0
            )
            data = json.loads(resp.choices[0].message.content)
            intent = data.get("predicted_intent", "other_unclear")
            if intent not in INTENTS:
                intent = "other_unclear"
            return IntentClassificationResult(
                predicted_intent=intent,
                confidence=float(data.get("confidence", 0.85)),
                reasoning=data.get("reasoning", "LLM classification"),
                is_fallback=False
            )
        except Exception as e:
            logger.warning(f"OpenAI classification failed ({e}); falling back to offline heuristic.")
            return classify_offline_heuristic(text)

    def _classify_anthropic(self, text: str) -> IntentClassificationResult:
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=self.anthropic_key)
            model = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")

            system_prompt = (
                "You are an expert customer support intent classifier for @SpotifyCares.\n"
                "Classify the customer message into EXACTLY ONE of the following 7 intents:\n"
                + "\n".join([f"- {k}: {v}" for k, v in INTENT_DESCRIPTIONS.items()]) +
                "\n\nReturn ONLY a JSON object: {\"predicted_intent\": \"<intent>\", \"confidence\": <float>, \"reasoning\": \"<explanation>\"}"
            )

            response = client.messages.create(
                model=model,
                max_tokens=300,
                temperature=0.0,
                system=system_prompt,
                messages=[{"role": "user", "content": f"Customer tweet: {text}"}]
            )
            raw = response.content[0].text
            json_match = re.search(r"\{.*\}", raw, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))
                intent = data.get("predicted_intent", "other_unclear")
                if intent not in INTENTS:
                    intent = "other_unclear"
                return IntentClassificationResult(
                    predicted_intent=intent,
                    confidence=float(data.get("confidence", 0.85)),
                    reasoning=data.get("reasoning", "Claude classification"),
                    is_fallback=False
                )
        except Exception as e:
            logger.warning(f"Anthropic classification failed ({e}); falling back to offline heuristic.")
        return classify_offline_heuristic(text)


if __name__ == "__main__":
    clf = IntentClassifier()
    res = clf.classify("Why was I billed $10.99 twice this month?")
    print("Classify result:", res.model_dump_json(indent=2))

"""
Intent Classifier — Decides when to use RAG retrieval.

Implements intent-aware retrieval logic:
- RETRIEVE: Questions about career, love, spirituality, planets, traits, predictions
- SKIP: Greetings, summarization requests, meta-questions, casual chat

Uses keyword/heuristic approach (no LLM call wasted on classification).
"""

import re
import logging
from typing import Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class Intent(str, Enum):
    """Detected user intent categories."""
    CAREER = "career"
    LOVE = "love"
    SPIRITUAL = "spiritual"
    PLANETARY = "planetary"
    ZODIAC_TRAITS = "zodiac_traits"
    PREDICTION = "prediction"
    GREETING = "greeting"
    SUMMARY = "summary"
    META = "meta"
    GENERAL = "general"


# Keywords mapped to intents
INTENT_KEYWORDS = {
    Intent.CAREER: [
        "career", "job", "work", "professional", "office", "business",
        "promotion", "salary", "income", "financial", "money", "wealth",
        "profession", "corporate", "interview", "opportunity",
    ],
    Intent.LOVE: [
        "love", "relationship", "partner", "marriage", "romantic",
        "dating", "crush", "soulmate", "spouse", "husband", "wife",
        "boyfriend", "girlfriend", "heart", "emotion", "feelings",
        "compatibility", "breakup", "divorce",
    ],
    Intent.SPIRITUAL: [
        "spiritual", "meditation", "prayer", "soul", "karma",
        "dharma", "enlightenment", "peace", "mantra", "chakra",
        "inner", "divine", "universe", "gratitude", "mindful",
        "yoga", "healing", "energy",
    ],
    Intent.PLANETARY: [
        "planet", "sun", "moon", "mars", "venus", "mercury",
        "jupiter", "saturn", "rahu", "ketu", "retrograde",
        "transit", "conjunction", "aspect", "malefic", "benefic",
        "nakshatra", "house",
    ],
    Intent.ZODIAC_TRAITS: [
        "zodiac", "sign", "aries", "taurus", "gemini", "cancer",
        "leo", "virgo", "libra", "scorpio", "sagittarius",
        "capricorn", "aquarius", "pisces", "personality", "trait",
        "strength", "weakness", "characteristic",
    ],
    Intent.PREDICTION: [
        "today", "tomorrow", "week", "month", "year", "predict",
        "forecast", "horoscope", "future", "will", "going to",
        "stressful", "lucky", "unlucky", "favorable",
    ],
    Intent.GREETING: [
        "hello", "hi", "hey", "good morning", "good evening",
        "namaste", "how are you", "what's up", "greetings",
    ],
    Intent.SUMMARY: [
        "summarize", "summary", "so far", "what have you told",
        "what did you say", "recap", "review", "remind me",
    ],
    Intent.META: [
        "why are you saying", "already told", "repeat", "again",
        "wrong", "mistake", "correct yourself", "shut up",
        "stop", "who are you", "what can you do",
    ],
}

# Intents that should trigger RAG retrieval
RETRIEVAL_INTENTS = {
    Intent.CAREER,
    Intent.LOVE,
    Intent.SPIRITUAL,
    Intent.PLANETARY,
    Intent.ZODIAC_TRAITS,
    Intent.PREDICTION,
}

# Intents that should skip retrieval
SKIP_INTENTS = {
    Intent.GREETING,
    Intent.SUMMARY,
    Intent.META,
}


def classify_intent(message: str) -> Tuple[bool, Intent]:
    """
    Classify the user's message intent and decide whether to retrieve.

    Args:
        message: The user's message text.

    Returns:
        Tuple of (should_retrieve: bool, detected_intent: Intent).
    """
    message_lower = message.lower().strip()

    # Score each intent by keyword matches
    intent_scores = {}
    for intent, keywords in INTENT_KEYWORDS.items():
        score = 0
        for keyword in keywords:
            if keyword in message_lower:
                score += 1
                # Bonus for exact word match (not substring)
                if re.search(rf'\b{re.escape(keyword)}\b', message_lower):
                    score += 1
        if score > 0:
            intent_scores[intent] = score

    if not intent_scores:
        # Default: general question — retrieve to be safe
        logger.info(f"Intent: GENERAL (no keyword match) → retrieve=True")
        return True, Intent.GENERAL

    # Pick the highest-scoring intent
    best_intent = max(intent_scores, key=intent_scores.get)

    should_retrieve = best_intent in RETRIEVAL_INTENTS

    logger.info(
        f"Intent: {best_intent.value} (score={intent_scores[best_intent]}) "
        f"→ retrieve={should_retrieve}"
    )

    return should_retrieve, best_intent

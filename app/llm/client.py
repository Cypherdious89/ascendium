"""
LLM Client — Abstraction layer for language model calls.

Supports:
- Google Gemini 2.5 Flash (when GEMINI_API_KEY is set)
- Stub client for testing without API key

The system prompt is dynamically constructed with:
- User zodiac and profile info
- Conversation history
- Retrieved context (if any)
- Language instruction (Hindi/English)
"""

import os
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


class LLMClient(ABC):
    """Abstract base class for LLM clients."""

    @abstractmethod
    def generate(
        self,
        user_message: str,
        system_message: str,
        conversation_history: List[Dict[str, str]],
    ) -> str:
        """
        Generate a response from the LLM.

        Args:
            user_message: The current user message.
            system_message: System prompt with context.
            conversation_history: List of {"role": ..., "content": ...} dicts.

        Returns:
            Generated response text.
        """
        pass


class GeminiClient(LLMClient):
    """Google Gemini 2.5 Flash client."""

    def __init__(self, model: str = "gemini-2.5-flash", temperature: float = 0.7):
        self.model = model
        self.temperature = temperature
        self._client = None

    def _get_client(self):
        if self._client is None:
            from google import genai
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY environment variable is not set.")
            self._client = genai.Client(api_key=api_key)
        return self._client

    def generate(
        self,
        user_message: str,
        system_message: str,
        conversation_history: List[Dict[str, str]],
    ) -> str:
        try:
            from google.genai import types

            client = self._get_client()

            # Build Gemini content history
            contents = []
            for turn in conversation_history:
                role = "model" if turn["role"] == "assistant" else "user"
                contents.append(
                    types.Content(role=role, parts=[types.Part(text=turn["content"])])
                )
            # Add current user message
            contents.append(
                types.Content(role="user", parts=[types.Part(text=user_message)])
            )

            response = client.models.generate_content(
                model=self.model,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=system_message,
                    temperature=self.temperature,
                    max_output_tokens=8192,
                ),
            )

            return response.text.strip()

        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            return f"I apologize, but I encountered an error generating a response. Please try again."


class StubClient(LLMClient):
    """
    Stub LLM client for testing without an API key.

    Generates template-based responses using the retrieved context
    and user profile information.
    """

    def generate(
        self,
        user_message: str,
        system_message: str,
        conversation_history: List[Dict[str, str]],
    ) -> str:
        # Extract zodiac from system message if present
        zodiac = "your zodiac sign"
        if "Zodiac:" in system_message:
            try:
                zodiac = system_message.split("Zodiac:")[1].split("\n")[0].strip()
            except (IndexError, AttributeError):
                pass

        # Check if Hindi is requested
        is_hindi = "Hindi" in system_message or "हिन्दी" in system_message

        # Extract retrieved context snippets (only from the knowledge section)
        context_snippets = []
        if "Retrieved Astrological Knowledge" in system_message:
            # Extract only the knowledge block
            knowledge_block = system_message.split("Retrieved Astrological Knowledge")[1]
            # Stop at the next section (Guidelines:)
            if "Guidelines:" in knowledge_block:
                knowledge_block = knowledge_block.split("Guidelines:")[0]
            if "IMPORTANT:" in knowledge_block:
                knowledge_block = knowledge_block.split("IMPORTANT:")[0]

            for line in knowledge_block.split("\n"):
                line = line.strip()
                if not line:
                    continue
                # Skip metadata lines
                if line.startswith("[Source:") or line.startswith("###") or line.startswith("- "):
                    continue
                context_snippets.append(line)

        message_lower = user_message.lower()

        if is_hindi:
            if "career" in message_lower or "करियर" in message_lower:
                base = f"आपकी राशि {zodiac} के अनुसार, यह समय करियर में नए अवसरों का है।"
            elif "love" in message_lower or "प्यार" in message_lower:
                base = f"आपकी राशि {zodiac} के अनुसार, भावनात्मक स्पष्टता आपके रिश्तों को मजबूत करेगी।"
            elif "spiritual" in message_lower or "आध्यात्मिक" in message_lower:
                base = f"आपकी राशि {zodiac} के अनुसार, ध्यान और आंतरिक शांति पर ध्यान दें।"
            else:
                base = f"आपकी राशि {zodiac} के अनुसार, आज का दिन सकारात्मक ऊर्जा से भरा है।"

            if context_snippets:
                base += " " + context_snippets[0] if context_snippets else ""
            return base

        else:
            if "career" in message_lower or "job" in message_lower or "work" in message_lower:
                base = f"As a {zodiac}, this is a promising period for your career."
            elif "love" in message_lower or "relationship" in message_lower:
                base = f"For {zodiac}, emotional clarity will strengthen your relationships."
            elif "stress" in message_lower or "difficult" in message_lower:
                base = f"As a {zodiac}, you may be feeling the effects of planetary transitions."
            elif "planet" in message_lower:
                base = f"For {zodiac}, planetary alignments are playing a significant role right now."
            elif "summarize" in message_lower or "summary" in message_lower:
                if conversation_history:
                    past_responses = [t["content"] for t in conversation_history if t["role"] == "assistant"]
                    if past_responses:
                        return "Here's a summary of what we've discussed:\n" + "\n".join(
                            f"- {r[:100]}..." if len(r) > 100 else f"- {r}"
                            for r in past_responses[-3:]
                        )
                return "We haven't discussed much yet. Feel free to ask me about your career, love life, or spiritual path!"
            elif any(g in message_lower for g in ["hello", "hi", "hey", "namaste"]):
                return f"Namaste! I'm your astrology guide. As a {zodiac}, I can help you with insights about your career, love life, spiritual path, and planetary influences. What would you like to know?"
            else:
                base = f"Based on your {zodiac} profile, the current cosmic alignment suggests a period of growth and reflection."

            if context_snippets:
                relevant = context_snippets[:2]
                base += " " + " ".join(relevant)

            return base


def build_system_prompt(
    zodiac: str,
    user_profile: dict,
    retrieved_context: str = "",
    language: str = "en",
) -> str:
    """
    Build the system prompt for the LLM.

    Args:
        zodiac: User's zodiac sign.
        user_profile: Enriched user profile dict.
        retrieved_context: Formatted context from RAG retrieval.
        language: 'en' or 'hi'.

    Returns:
        System prompt string.
    """
    prompt_parts = [
        "You are an expert Vedic and Western astrology advisor. "
        "Provide personalized, insightful astrological guidance based on the user's profile "
        "and the knowledge provided. Be warm, encouraging, and specific.",
        "",
        f"### User Profile:",
        f"- Name: {user_profile.get('name', 'Unknown')}",
        f"- Zodiac: {zodiac}",
        f"- Birth Date: {user_profile.get('birth_date', 'Unknown')}",
    ]

    if user_profile.get("birth_time"):
        prompt_parts.append(f"- Birth Time: {user_profile['birth_time']}")
    if user_profile.get("birth_place"):
        prompt_parts.append(f"- Birth Place: {user_profile['birth_place']}")
    if user_profile.get("age"):
        prompt_parts.append(f"- Age: {user_profile['age']}")

    if retrieved_context:
        prompt_parts.append("")
        prompt_parts.append(retrieved_context)

    if language == "hi":
        prompt_parts.append("")
        prompt_parts.append(
            "IMPORTANT: Respond entirely in Hindi (Devanagari script). "
            "The user prefers Hindi language responses."
        )

    prompt_parts.extend([
        "",
        "Guidelines:",
        "- Reference the user's zodiac sign naturally in your response",
        "- Use the retrieved knowledge to ground your advice",
        "- Be specific and actionable, avoid generic platitudes",
        "- If no context was retrieved, rely on your astrological knowledge",
        "- Keep responses concise but meaningful (2-4 paragraphs)",
    ])

    return "\n".join(prompt_parts)


def get_llm_client() -> LLMClient:
    """
    Factory function to get the appropriate LLM client.

    Uses Gemini if GEMINI_API_KEY is set, otherwise falls back to stub.
    """
    if os.getenv("GEMINI_API_KEY"):
        logger.info("Using Gemini 2.5 Flash LLM client.")
        return GeminiClient()
    else:
        logger.info("No GEMINI_API_KEY found. Using Stub LLM client.")
        return StubClient()

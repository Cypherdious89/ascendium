"""Session-based conversation memory with sliding window control."""

import time
from typing import Dict, List, Optional
from dataclasses import dataclass, field


@dataclass
class ConversationTurn:
    """A single turn in the conversation."""
    role: str  # "user" or "assistant"
    content: str
    timestamp: float = field(default_factory=time.time)
    retrieval_used: bool = False
    context_sources: List[str] = field(default_factory=list)


@dataclass
class Session:
    """A user session with conversation history and profile."""
    session_id: str
    user_profile: dict = field(default_factory=dict)
    history: List[ConversationTurn] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)


class SessionManager:
    """
    Manages multi-turn conversation sessions with memory control.

    Features:
    - In-memory session store keyed by session_id
    - Sliding window (last-N turns) for memory control
    - Session auto-creation on first message
    """

    def __init__(self, max_history: int = 10):
        """
        Args:
            max_history: Maximum number of turns to keep in memory per session.
        """
        self._sessions: Dict[str, Session] = {}
        self.max_history = max_history

    def get_or_create_session(
        self, session_id: str, user_profile: Optional[dict] = None
    ) -> Session:
        """
        Get existing session or create a new one.

        Args:
            session_id: Unique session identifier.
            user_profile: User profile dict (used on creation or update).

        Returns:
            The Session object.
        """
        if session_id not in self._sessions:
            self._sessions[session_id] = Session(
                session_id=session_id,
                user_profile=user_profile or {},
            )
        else:
            # Update profile if provided (user might send updated info)
            if user_profile:
                self._sessions[session_id].user_profile = user_profile
            self._sessions[session_id].last_active = time.time()

        return self._sessions[session_id]

    def add_turn(
        self,
        session_id: str,
        role: str,
        content: str,
        retrieval_used: bool = False,
        context_sources: Optional[List[str]] = None,
    ) -> None:
        """
        Add a conversation turn to the session history.

        Applies sliding window to keep memory bounded.

        Args:
            session_id: Session to add the turn to.
            role: "user" or "assistant".
            content: The message content.
            retrieval_used: Whether RAG retrieval was used for this turn.
            context_sources: List of source names used as context.
        """
        session = self._sessions.get(session_id)
        if not session:
            return

        turn = ConversationTurn(
            role=role,
            content=content,
            retrieval_used=retrieval_used,
            context_sources=context_sources or [],
        )
        session.history.append(turn)

        # Sliding window: trim to max_history
        if len(session.history) > self.max_history:
            session.history = session.history[-self.max_history:]

        session.last_active = time.time()

    def get_history(self, session_id: str) -> List[Dict]:
        """
        Get conversation history formatted for LLM context.

        Returns list of {"role": ..., "content": ...} dicts.
        """
        session = self._sessions.get(session_id)
        if not session:
            return []

        return [
            {"role": turn.role, "content": turn.content}
            for turn in session.history
        ]

    def get_session_summary(self, session_id: str) -> Optional[str]:
        """
        Generate a brief summary of the conversation so far.

        Useful for handling "summarize what you've told me" queries.
        """
        session = self._sessions.get(session_id)
        if not session or not session.history:
            return None

        assistant_turns = [
            t.content for t in session.history if t.role == "assistant"
        ]
        if not assistant_turns:
            return "No responses have been given yet in this session."

        summary_parts = []
        for i, content in enumerate(assistant_turns, 1):
            # Truncate long responses for summary
            snippet = content[:150] + "..." if len(content) > 150 else content
            summary_parts.append(f"{i}. {snippet}")

        return "Here's a summary of our conversation so far:\n" + "\n".join(summary_parts)

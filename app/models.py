"""Pydantic models for the Astro Conversational Agent API."""

from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import date


class UserProfile(BaseModel):
    """User birth details and preferences."""
    name: str
    birth_date: str = Field(..., description="Date of birth in YYYY-MM-DD format")
    birth_time: Optional[str] = Field(None, description="Time of birth in HH:MM format")
    birth_place: Optional[str] = Field(None, description="City, Country")
    preferred_language: str = Field("en", description="'en' for English, 'hi' for Hindi")


class ChatRequest(BaseModel):
    """Incoming chat request."""
    session_id: str
    message: str
    user_profile: UserProfile


class ChatResponse(BaseModel):
    """Outgoing chat response."""
    response: str
    zodiac: str
    context_used: List[str] = Field(default_factory=list)
    retrieval_used: bool = False

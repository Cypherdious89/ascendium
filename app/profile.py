"""User profile management and zodiac sign derivation."""

from datetime import datetime
from typing import Optional


# Standard Western zodiac date ranges (month, day) boundaries
# Each entry: (end_month, end_day, zodiac_sign)
ZODIAC_DATES = [
    (1, 19, "Capricorn"),
    (2, 18, "Aquarius"),
    (3, 20, "Pisces"),
    (4, 19, "Aries"),
    (5, 20, "Taurus"),
    (6, 20, "Gemini"),
    (7, 22, "Cancer"),
    (8, 22, "Leo"),
    (9, 22, "Virgo"),
    (10, 22, "Libra"),
    (11, 21, "Scorpio"),
    (12, 21, "Sagittarius"),
    (12, 31, "Capricorn"),
]


def get_zodiac_sign(birth_date_str: str) -> str:
    """
    Derive zodiac sign from a birth date string (YYYY-MM-DD).

    Args:
        birth_date_str: Date of birth in YYYY-MM-DD format.

    Returns:
        Zodiac sign as a string (e.g., "Leo", "Aries").
    """
    try:
        dt = datetime.strptime(birth_date_str, "%Y-%m-%d")
    except ValueError:
        return "Unknown"

    month, day = dt.month, dt.day

    for end_month, end_day, sign in ZODIAC_DATES:
        if (month < end_month) or (month == end_month and day <= end_day):
            return sign

    return "Capricorn"  # Fallback for Dec 22-31


def get_age(birth_date_str: str) -> Optional[int]:
    """Calculate age from birth date string."""
    try:
        dt = datetime.strptime(birth_date_str, "%Y-%m-%d")
        today = datetime.now()
        age = today.year - dt.year
        if (today.month, today.day) < (dt.month, dt.day):
            age -= 1
        return age
    except ValueError:
        return None


def enrich_profile(user_profile: dict) -> dict:
    """
    Enrich user profile with derived astrological data.

    Adds zodiac sign and age to the user profile.
    """
    birth_date = user_profile.get("birth_date", "")
    enriched = dict(user_profile)
    enriched["zodiac"] = get_zodiac_sign(birth_date)
    enriched["age"] = get_age(birth_date)
    return enriched

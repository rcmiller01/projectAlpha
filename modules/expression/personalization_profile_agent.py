"""Personalization Profile Agent
-------------------------------

Calibrates system responses to align with user-specific sensual
and emotional preferences. Supports multiple profiles and dynamic
re-tuning.
"""

from __future__ import annotations

from typing import Dict, Any


class PersonalizationProfileAgent:
    """Manage user personalization profiles for expression tuning."""

    def __init__(self):
        self.profiles: Dict[str, Dict[str, Any]] = {}

    def set_preferences(self, user_id: str, preferences: Dict[str, Any]) -> None:
        self.profiles[user_id] = preferences

    def get_preferences(self, user_id: str) -> Dict[str, Any]:
        return self.profiles.get(user_id, {})

    def tune_biases(self, user_id: str, dial_agent) -> None:
        """Apply stored preferences to the expression dial agent."""
        prefs = self.get_preferences(user_id)
        adjustments = prefs.get("dial_defaults", {})
        if adjustments:
            dial_agent.apply_adjustments(user_id, adjustments)

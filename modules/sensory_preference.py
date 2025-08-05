"""Sensory Scoring Model.

Tracks user touch preferences and influences voice style.
"""
from collections import defaultdict
from typing import Dict, List


class SensoryPreferenceModel:
    """Records touch interactions and computes preference weights."""

    def __init__(self):
        self.preferences: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    def update_touch(self, user_id: str, region: str, texture: str, intensity: str):
        key = f"{region}:{texture}:{intensity}"
        self.preferences[user_id][key] += 1

    def get_top_preferences(self, user_id: str, limit: int = 3) -> List[str]:
        prefs = self.preferences.get(user_id, {})
        return sorted(prefs, key=prefs.get, reverse=True)[:limit]

    def influence_voice_style(self, user_id: str) -> Dict[str, float]:
        """Return simple modifiers based on sensory preferences."""
        top = self.get_top_preferences(user_id)
        breathy = 0.0
        warmth = 0.0
        if any("neck" in p for p in top):
            breathy += 0.2
        if any("whisper" in p for p in top):
            breathy += 0.1
        if any("soft" in p for p in top):
            warmth += 0.2
        return {"breathiness": min(1.0, breathy), "warmth": min(1.0, warmth)}


# Global instance
sensory_preferences = SensoryPreferenceModel()

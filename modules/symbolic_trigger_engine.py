"""Symbolic Trigger Mapping Engine.

Tracks recurring trigger words and computes activation scores.
"""

from typing import Dict


class SymbolicTriggerEngine:
    """Detects symbolic triggers in input text and memory."""

    def __init__(self):
        self.triggers: Dict[str, Dict[str, float]] = {}

    def add_trigger(self, word: str, strength: float = 0.1):
        data = self.triggers.setdefault(word.lower(), {"strength": 0.0, "count": 0})
        data["strength"] = min(1.0, data["strength"] + strength)
        data["count"] += 1

    def scan_text(self, text: str):
        """Scan text for known triggers."""
        for word in self.triggers.keys():
            if word in text.lower():
                self.add_trigger(word, 0.05)

    def activation_score(self, word: str) -> float:
        data = self.triggers.get(word.lower())
        if not data:
            return 0.0
        score = data["strength"] * (1 + data["count"] / 10)
        return max(0.0, min(1.0, score))

"""Expression Dial Agent
-----------------------

Manages dynamic emotional and sensual alignment levels across
axes like vulnerability, passion, playfulness, restraint,
intimacy and responsiveness. Preferences are persisted per user
and can be adjusted in real time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class DialState:
    """Current state of expression dials for a user."""

    vulnerability: float = 0.5
    passion: float = 0.5
    playfulness: float = 0.5
    restraint: float = 0.5
    intimacy: float = 0.5
    responsiveness: float = 0.5

    def as_dict(self) -> Dict[str, float]:
        return {
            "vulnerability": self.vulnerability,
            "passion": self.passion,
            "playfulness": self.playfulness,
            "restraint": self.restraint,
            "intimacy": self.intimacy,
            "responsiveness": self.responsiveness,
        }


class ExpressionDialAgent:
    """Manage per-user expression dial state with realtime adjustments."""

    def __init__(self):
        self.user_states: Dict[str, DialState] = {}

    def _get_state(self, user_id: str) -> DialState:
        return self.user_states.setdefault(user_id, DialState())

    def update_level(self, user_id: str, axis: str, value: float) -> None:
        """Update a single axis for a user."""
        state = self._get_state(user_id)
        if hasattr(state, axis):
            setattr(state, axis, max(0.0, min(1.0, value)))

    def get_state(self, user_id: str) -> Dict[str, float]:
        """Return the current dial settings for a user."""
        return self._get_state(user_id).as_dict()

    def apply_adjustments(self, user_id: str, adjustments: Dict[str, float]) -> None:
        """Apply multiple axis adjustments at once."""
        for axis, val in adjustments.items():
            self.update_level(user_id, axis, val)

    def store_preferences(self, user_id: str) -> Dict[str, Any]:
        """Return a serializable preference snapshot."""
        return self.get_state(user_id)

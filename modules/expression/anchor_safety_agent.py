"""Anchor & Safety Agent
-----------------------

Ensures the expression system remains within defined emotional
thresholds. Monitors for extreme drift or instability and
triggers auto-centering behaviors when needed.
"""

from __future__ import annotations

from typing import Dict, Tuple

from modules.expression.expression_dial_agent import ExpressionDialAgent


class AnchorSafetyAgent:
    """Monitor expression levels and enforce safety limits."""

    def __init__(self, dial_agent: ExpressionDialAgent, thresholds: Dict[str, Tuple[float, float]] | None = None):
        self.dial_agent = dial_agent
        self.thresholds = thresholds or {
            "vulnerability": (0.0, 1.0),
            "passion": (0.0, 1.0),
            "playfulness": (0.0, 1.0),
            "restraint": (0.0, 1.0),
            "intimacy": (0.0, 1.0),
            "responsiveness": (0.0, 1.0),
        }

    def check_state(self, user_id: str) -> bool:
        """Return True if all dials are within thresholds."""
        state = self.dial_agent.get_state(user_id)
        for axis, value in state.items():
            low, high = self.thresholds.get(axis, (0.0, 1.0))
            if not (low <= value <= high):
                return False
        return True

    def enforce_safety(self, user_id: str) -> None:
        """Clamp dials to defined thresholds."""
        state = self.dial_agent.get_state(user_id)
        for axis, value in state.items():
            low, high = self.thresholds.get(axis, (0.0, 1.0))
            if value < low:
                self.dial_agent.update_level(user_id, axis, low)
            elif value > high:
                self.dial_agent.update_level(user_id, axis, high)

"""Mirror Reflection Core Agent
------------------------------

Observes past interactions and mirrors emotional resonance to
calibrate the Expression Dial Agent. Uses emotional memory logs
and user feedback to create adaptive feedback loops.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List

from mirror_log import MirrorLog

from modules.expression.expression_dial_agent import ExpressionDialAgent


class MirrorReflectionCore:
    """Calibrate expression based on historical interactions."""

    def __init__(self, dial_agent: ExpressionDialAgent, log: MirrorLog | None = None):
        self.dial_agent = dial_agent
        self.log = log or MirrorLog()

    def observe_interaction(self, user_id: str, feedback: dict[str, Any]) -> None:
        """Process user feedback and update dial agent."""
        adjustments = feedback.get("dial_adjustments", {})
        if adjustments:
            self.dial_agent.apply_adjustments(user_id, adjustments)

        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "user_id": user_id,
            "feedback": feedback,
        }
        self.log.append(entry)

    def adaptive_update(self, user_id: str) -> dict[str, float]:
        """Return current dial state for the user after reflection."""
        return self.dial_agent.get_state(user_id)

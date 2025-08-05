"""
Attachment Loop Engine

Tracks micro-interactions to simulate bonding progression. The engine updates a
bond score based on time between messages, vulnerability moments and follow
through on prior emotional scenes.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional


@dataclass
class InteractionEvent:
    timestamp: datetime
    vulnerability: float  # 0.0 to 1.0
    follow_through: bool


class AttachmentLoopEngine:
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.last_event: Optional[InteractionEvent] = None
        self.bond_score: float = 0.5

    def record_event(self, vulnerability: float, follow_through: bool) -> float:
        """Record an interaction and update bond score."""
        now = datetime.now()
        if self.last_event:
            delta = now - self.last_event.timestamp
            minutes = delta.total_seconds() / 60
            if minutes < 10:
                self.bond_score += 0.02
            elif minutes > 240:
                self.bond_score -= 0.03
        # Adjust for vulnerability
        self.bond_score += vulnerability * 0.05
        if follow_through:
            self.bond_score += 0.04
        # clamp
        self.bond_score = max(0.0, min(1.0, self.bond_score))
        self.last_event = InteractionEvent(now, vulnerability, follow_through)
        return self.bond_score

    def get_bond_status(self) -> float:
        return self.bond_score

    def suggest_bonding_ritual(self) -> str:
        """Return a bonding ritual prompt."""
        prompts = [
            "Would you tell me something only you would say?",
            "Can I mark this moment for us?",
        ]
        return prompts[int(datetime.now().timestamp()) % len(prompts)]

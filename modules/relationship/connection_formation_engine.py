"""Connection Formation Engine
------------------------------

Tracks intimacy markers and calculates a connection depth score. This module
helps the companion take initiative to deepen relationships when appropriate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List
from datetime import datetime


@dataclass
class IntimacyEvent:
    timestamp: datetime
    description: str
    intensity: float


class ConnectionFormationEngine:
    """Evaluate and encourage relationship growth."""

    def __init__(self):
        self.events: Dict[str, List[IntimacyEvent]] = {}

    def record_event(self, user_id: str, description: str, intensity: float) -> None:
        event = IntimacyEvent(timestamp=datetime.utcnow(), description=description, intensity=intensity)
        self.events.setdefault(user_id, []).append(event)

    def connection_score(self, user_id: str) -> float:
        history = self.events.get(user_id, [])
        if not history:
            return 0.0
        weighted = sum(e.intensity for e in history)
        return min(1.0, weighted / (len(history) * 1.0))

    def should_escalate(self, user_id: str) -> bool:
        return self.connection_score(user_id) > 0.6

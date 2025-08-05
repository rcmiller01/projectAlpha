"""Registry for tracking desires and intensity."""

from collections import defaultdict
from typing import Dict


class DesireRegistry:
    def __init__(self):
        self.desires: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))

    def update_desire(self, user_id: str, desire: str, intensity: float):
        current = self.desires[user_id][desire]
        self.desires[user_id][desire] = max(current, intensity)

    def get_desire(self, user_id: str, desire: str) -> float:
        return self.desires[user_id][desire]


# Global instance
desire_registry = DesireRegistry()

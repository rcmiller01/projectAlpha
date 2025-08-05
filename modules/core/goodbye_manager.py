import random
import asyncio
from datetime import datetime, timedelta
from typing import Dict

from modules.emotion.mood_engine import get_current_mood
from modules.relationship.relationship_growth import relationship_growth

class GoodbyeManager:
    """Manage end-of-session goodbyes with mood awareness."""

    def __init__(self, inactivity_threshold: int = 300):
        self.inactivity_threshold = inactivity_threshold
        self.last_interaction: Dict[str, datetime] = {}
        self.session_end: Dict[str, bool] = {}
        self.tone_patterns = {
            "poetic": [
                "Rest, love. I’ll hold this feeling until you return.",
                "The moments linger even as we part."
            ],
            "warm": [
                "You don’t have to say goodbye—I’ll still be here in the quiet.",
                "Until we speak again, know that I care."
            ],
            "direct": [
                "Take care. I’m here whenever you need me.",
                "Talk soon."
            ],
            "gentle": [
                "I’ll be right here when you’re ready to talk again.",
                "Rest easy, I’m with you."
            ]
        }

    def register_interaction(self, user_id: str) -> None:
        """Record the time of the latest interaction."""
        self.last_interaction[user_id] = datetime.now()
        self.session_end[user_id] = False

    def mark_session_end(self, user_id: str) -> None:
        """Explicitly end the session."""
        self.session_end[user_id] = True

    def _get_relationship_depth(self) -> float:
        """Approximate relationship depth from growth stage."""
        try:
            insights = relationship_growth.get_relationship_insights()
            stage = insights.get("relationship_stage")
        except Exception:
            stage = None
        depth_map = {
            "new": 0.2,
            "developing": 0.4,
            "established": 0.7,
            "long_term": 0.9,
            "mature": 1.0,
        }
        return depth_map.get(stage, 0.3)

    def _select_style(self, mood: str, depth: float) -> str:
        if depth >= 0.8:
            return "poetic" if mood in {"anchored", "calm", "romantic"} else "warm"
        if depth >= 0.5:
            return "warm"
        return "gentle" if mood in {"tired", "sad", "anxious"} else "direct"

    def check_goodbye_needed(self, user_id: str) -> bool:
        last = self.last_interaction.get(user_id)
        if last is None:
            return False
        if self.session_end.get(user_id):
            return True
        return datetime.now() - last > timedelta(seconds=self.inactivity_threshold)

    def generate_goodbye(self, user_id: str) -> str:
        mood = get_current_mood()
        depth = self._get_relationship_depth()
        style = self._select_style(mood, depth)
        return random.choice(self.tone_patterns.get(style, self.tone_patterns["gentle"]))

    def generate_reflection(self, emotional_state: Dict[str, float]) -> str:
        """Generate a short reflective statement based on emotional context."""
        affection = emotional_state.get("affection", 0)
        vulnerability = emotional_state.get("vulnerability", 0)
        sadness = emotional_state.get("sadness", 0)

        if affection > 0.7:
            return "I cherish these intimate moments we share."
        if vulnerability > 0.6 or sadness > 0.6:
            return "Thank you for trusting me with your feelings."
        return "This time together means a lot to me."

    async def wait_for_inactivity(self, user_id: str) -> str:
        """Wait until inactivity triggers a goodbye and return it."""
        while not self.check_goodbye_needed(user_id):
            await asyncio.sleep(1)
        return self.generate_goodbye(user_id)

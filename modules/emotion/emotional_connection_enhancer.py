"""Emotional Connection Enhancer
--------------------------------

Refines the companion's ability to detect, reflect and adapt to emotional nuance.
This module relies on the existing ``EmotionDetector`` and stores an emotional
arc for each user so that responses can reference prior feelings.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Dict, List, Any

from .emotion_detection import EmotionDetector

logger = logging.getLogger(__name__)


class EmotionalConnectionEnhancer:
    """Track and adapt to user emotions across conversations."""

    def __init__(self):
        self.detector = EmotionDetector()
        self.emotional_history: Dict[str, List[Dict[str, Any]]] = {}

    async def process_message(self, user_id: str, message: str, mode: str) -> Dict[str, Any]:
        """Analyze a message and store emotional context."""
        analysis = await self.detector.analyze_text(message)
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "mode": mode,
            "message": message,
            "analysis": analysis,
        }
        self.emotional_history.setdefault(user_id, []).append(entry)
        logger.debug("Stored emotional entry for %s: %s", user_id, entry)
        return analysis

    def get_recent_emotion(self, user_id: str) -> Dict[str, Any] | None:
        """Return the most recent emotional analysis for a user."""
        history = self.emotional_history.get(user_id)
        if not history:
            return None
        return history[-1]["analysis"]

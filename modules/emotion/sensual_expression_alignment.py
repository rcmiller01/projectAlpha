"""Sensual Expression Alignment System - Phase 1

This module implements a configurable system that allows the AI
companion to express emotions and sensuality while respecting
user preferences and safety constraints.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class ExpressionDial:
    """Tuneable levels for individual emotions."""

    levels: Dict[str, float] = field(default_factory=dict)
    max_level: float = 10.0

    def set_level(self, emotion: str, value: float) -> None:
        value = max(0.0, min(value, self.max_level))
        self.levels[emotion] = value
        logger.debug("Set %s level to %s", emotion, value)

    def get_level(self, emotion: str) -> float:
        return self.levels.get(emotion, 0.0)


@dataclass
class MirrorReflectionCore:
    """Tracks user feedback and adjusts expression levels."""

    expression_dial: ExpressionDial
    feedback_log: Dict[str, list] = field(default_factory=dict)

    def record_feedback(self, emotion: str, user_reaction: float) -> None:
        self.feedback_log.setdefault(emotion, []).append(user_reaction)
        average = sum(self.feedback_log[emotion]) / len(self.feedback_log[emotion])
        new_level = self.expression_dial.get_level(emotion) * (1 + average)
        self.expression_dial.set_level(emotion, new_level)
        logger.debug(
            "Adjusted %s level to %s based on feedback %s",
            emotion,
            new_level,
            user_reaction,
        )


@dataclass
class AnchorSystem:
    """Prevents runaway emotional intensity."""

    expression_dial: ExpressionDial
    safety_threshold: float = 9.0

    def enforce_safety(self) -> None:
        for emotion, level in list(self.expression_dial.levels.items()):
            if level > self.safety_threshold:
                logger.warning("%s level %s exceeded threshold, resetting", emotion, level)
                self.expression_dial.set_level(emotion, self.safety_threshold)


@dataclass
class UserProfile:
    """Stores user preference for expression levels."""

    preferences: Dict[str, float] = field(default_factory=dict)


class PersonalizationEngine:
    """Learns and applies user preferences over time."""

    def __init__(self):
        self.profiles: Dict[str, UserProfile] = {}

    def get_profile(self, user_id: str) -> UserProfile:
        return self.profiles.setdefault(user_id, UserProfile())

    def update_preference(self, user_id: str, emotion: str, value: float) -> None:
        profile = self.get_profile(user_id)
        profile.preferences[emotion] = value
        logger.debug("Updated user %s preference for %s to %s", user_id, emotion, value)


class SensualExpressionAlignmentSystem:
    """Main coordination class."""

    def __init__(self, user_id: str):
        self.user_id = user_id
        self.dial = ExpressionDial()
        self.mirror = MirrorReflectionCore(self.dial)
        self.anchor = AnchorSystem(self.dial)
        self.personalization = PersonalizationEngine()

    def set_emotion_level(self, emotion: str, value: float) -> None:
        self.dial.set_level(emotion, value)
        self.anchor.enforce_safety()

    def record_user_feedback(self, emotion: str, reaction: float) -> None:
        self.mirror.record_feedback(emotion, reaction)
        self.anchor.enforce_safety()

    def apply_user_profile(self) -> None:
        profile = self.personalization.get_profile(self.user_id)
        for emotion, level in profile.preferences.items():
            self.dial.set_level(emotion, level)
        self.anchor.enforce_safety()


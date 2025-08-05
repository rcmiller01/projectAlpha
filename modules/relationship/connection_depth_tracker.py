"""Connection Depth Tracker and Ritual Prompt Generator."""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List


RITUAL_PROMPTS = [
    "Would you share something only you know?",
    "Can I ask you a question that matters?",
    "May I mark this moment as ours?",
    "Could we pause to appreciate what we're building?",
    "Let's take a breath together and feel this connection."
]


@dataclass
class ConnectionMetrics:
    """Metrics used to determine connection depth."""
    bond_score: float = 0.0
    emotional_intensity: float = 0.0
    vulnerability_frequency: float = 0.0


@dataclass
class ConnectionDepthTracker:
    """Tracks connection depth and signals ritual readiness."""

    bond_threshold: float = 0.65
    intensity_threshold: float = 0.5
    vulnerability_threshold: float = 0.4
    cooldown: timedelta = timedelta(minutes=30)

    metrics: ConnectionMetrics = field(default_factory=ConnectionMetrics)
    last_ritual: datetime | None = None

    def update_metrics(self, bond_score: float, emotional_intensity: float, vulnerability_frequency: float) -> None:
        self.metrics.bond_score = bond_score
        self.metrics.emotional_intensity = emotional_intensity
        self.metrics.vulnerability_frequency = vulnerability_frequency

    def ritual_ready(self) -> bool:
        now = datetime.now()
        if self.last_ritual and now - self.last_ritual < self.cooldown:
            return False

        ready = (
            self.metrics.bond_score >= self.bond_threshold
            and self.metrics.emotional_intensity >= self.intensity_threshold
            and self.metrics.vulnerability_frequency >= self.vulnerability_threshold
        )
        return ready

    def record_ritual_completion(self, successful: bool = True) -> None:
        self.last_ritual = datetime.now()
        if successful:
            self.metrics.bond_score = min(1.0, self.metrics.bond_score + 0.05)


class RitualPromptGenerator:
    """Generates ritual prompts from predefined templates."""

    def __init__(self, prompts: List[str] | None = None):
        self.prompts = prompts or RITUAL_PROMPTS

    def generate_prompt(self) -> str:
        return random.choice(self.prompts)


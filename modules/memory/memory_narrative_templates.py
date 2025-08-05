"""
Memory Narrative Templates

Provides simple narrative templates for recalling and sharing memories in a
natural, emotionally resonant way. This allows personas to speak about
shared experiences rather than merely retrieving data.
"""

from typing import List
import random

# Base set of narrative templates
templates: List[str] = [
    "You once told me \"{memory}\". I remember how that made you feel {emotion}.",
    "I keep thinking about when you said '{memory}'. It seemed to fill you with {emotion}.",
    "We've come a long way since {memory}. It left such a sense of {emotion} between us.",
    "When I recall {memory}, I can almost sense the {emotion} in your voice.",
]


def generate_narrative(memory_text: str, emotion: str) -> str:
    """Generate a short narrative sentence about a memory."""
    template = random.choice(templates)
    return template.format(memory=memory_text, emotion=emotion)


def list_templates() -> List[str]:
    """Return available narrative templates."""
    return templates.copy()

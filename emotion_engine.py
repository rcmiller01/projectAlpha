from __future__ import annotations

"""Simple emotion token generator.
Provides adjectives, emojis, and metaphors based on valence and arousal.
"""

from typing import Tuple


POS_ADJECTIVES = ["joyful", "cheerful", "uplifting", "bright"]
NEG_ADJECTIVES = ["somber", "gloomy", "downcast", "heavy"]
NEU_ADJECTIVES = ["calm", "steady", "balanced"]

POS_EMOJIS_HIGH = "😄"
POS_EMOJIS_LOW = "🙂"
NEG_EMOJIS_HIGH = "😠"
NEG_EMOJIS_LOW = "😢"
NEU_EMOJI = "😐"


def choose_adjective(valence: float, arousal: float) -> str:
    """Return an adjective reflecting the emotion."""
    if valence > 0.3:
        base = POS_ADJECTIVES
    elif valence < -0.3:
        base = NEG_ADJECTIVES
    else:
        base = NEU_ADJECTIVES
    idx = int(abs(arousal) * (len(base) - 1))
    return base[min(idx, len(base) - 1)]


def choose_emoji(valence: float, arousal: float) -> str:
    """Return an emoji matching valence/arousal."""
    if valence > 0.3:
        return POS_EMOJIS_HIGH if arousal > 0.5 else POS_EMOJIS_LOW
    if valence < -0.3:
        return NEG_EMOJIS_HIGH if arousal > 0.5 else NEG_EMOJIS_LOW
    return NEU_EMOJI


def choose_metaphor(valence: float) -> str:
    """Return a short metaphor capturing mood."""
    if valence > 0.3:
        return "like sunshine breaking through clouds"
    if valence < -0.3:
        return "like walking through a storm"
    return "like standing in a quiet room"

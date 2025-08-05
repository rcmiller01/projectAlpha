from dataclasses import dataclass
from typing import Dict, Tuple

@dataclass
class MoodStyleProfile:
    """Defines stylistic traits based on mood and mode."""
    metaphor_density: float
    sentence_length_avg: int
    warmth_level: float
    directness_level: float

# Sample style profiles for common mood/mode pairs
MOOD_MODE_STYLE_MAP: Dict[Tuple[str, str], MoodStyleProfile] = {
    ("calm", "personal"): MoodStyleProfile(
        metaphor_density=0.3,
        sentence_length_avg=14,
        warmth_level=0.9,
        directness_level=0.6,
    ),
    ("anxious", "crisis"): MoodStyleProfile(
        metaphor_density=0.1,
        sentence_length_avg=8,
        warmth_level=0.95,
        directness_level=0.95,
    ),
    ("intimate", "hybrid"): MoodStyleProfile(
        metaphor_density=0.6,
        sentence_length_avg=12,
        warmth_level=1.0,
        directness_level=0.7,
    ),
}

DEFAULT_MOOD_STYLE = MoodStyleProfile(
    metaphor_density=0.2,
    sentence_length_avg=12,
    warmth_level=0.8,
    directness_level=0.7,
)


def get_mood_style_profile(mood: str, mode: str) -> MoodStyleProfile:
    """Return the style profile for a given mood/mode pair."""
    return MOOD_MODE_STYLE_MAP.get((mood, mode), DEFAULT_MOOD_STYLE)

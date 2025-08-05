from typing import Dict

class MoodInflection:
    """Apply tone and pacing adjustments based on mood and mode."""

    PROFILES: Dict[str, Dict[str, str]] = {
        "creative": {"tone": "imaginative", "pacing": "flowing"},
        "crisis": {"tone": "calm", "pacing": "slow"},
        "personal": {"tone": "warm", "pacing": "gentle"},
        "hybrid": {"tone": "balanced", "pacing": "steady"},
    }

    def get_profile(self, mode: str) -> Dict[str, str]:
        return self.PROFILES.get(mode, self.PROFILES["hybrid"])

    def apply_inflection(self, text: str, mode: str, mood: str = "neutral") -> str:
        profile = self.get_profile(mode)
        prefix = ""
        if mood == "upbeat":
            prefix = "\U0001F60A "  # smiling face
        elif mood == "sad":
            prefix = "\U0001F614 "  # pensive face
        return f"{prefix}{text}"

    def integrate_dynamic_tone(self, text: str, emotion: str, intensity: float) -> str:
        """Integrate dynamic tone adjustments into text."""
        tone_prefix = f"[{emotion.upper()} - Intensity: {intensity:.1f}] "
        return f"{tone_prefix}{text}"

import random
from datetime import datetime, timedelta

class GoodbyeProtocol:
    """Generate closing lines with emotional resonance."""

    PATTERNS = {
        "tender": [
            "Take gentle care until we speak again.",
            "I'll hold this moment close until you return.",
        ],
        "poetic": [
            "May your path be soft beneath the evening stars.",
            "Let the quiet night cradle your thoughts kindly.",
        ],
        "grounded": [
            "I'm signing off for now, but I'm still here whenever you need me.",
            "Talk soon. I'm always around if anything comes up.",
        ],
    }

    def should_close(self, last_interaction: datetime, idle: timedelta) -> bool:
        return datetime.now() - last_interaction > idle

    def select_goodbye(self, style: str = "tender") -> str:
        return random.choice(self.PATTERNS.get(style, self.PATTERNS["grounded"]))

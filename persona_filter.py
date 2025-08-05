"""Persona Filter Engine
Applies unified tone and vocabulary adjustments across backend responses."""
import re


def apply_persona_filter(response_text: str, persona: str) -> str:
    """Apply light NLP-based adjustments to a response according to persona."""
    if not response_text:
        return ""
    text = re.sub(r"^As an? AI( language model)?[,. ]*", "", response_text, flags=re.IGNORECASE)

    persona_lower = persona.lower()

    if persona_lower == "companion":
        text = text.replace("assistant", "friend")
        if not text.startswith("😊"):
            text = "😊 " + text
    elif persona_lower == "analyst":
        text = text.replace("I think", "My analysis indicates")
    elif persona_lower == "creative":
        if not text.endswith("✨"):
            text = text + " ✨"
    elif persona_lower == "coach":
        text = text.replace("You should", "Let's work on")
    # Generic cleanup
    text = text.replace("AI", "assistant")
    return text.strip()

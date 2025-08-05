"""Mirror Reflection Core - Emotional Resonance Analysis and Feedback Loop"""

from pathlib import Path
from typing import Dict

# SECTION 1: User Signal Inference
def infer_emotional_state(user_input: str) -> str:
    """Infer the user's emotional state using simple heuristics."""
    lower = user_input.lower()
    if any(word in lower for word in ["lonely", "tired", "miss", "sad", "hurt"]):
        return "melancholy"
    if any(word in lower for word in ["excited", "can't wait", "happy", "love", "joy"]):
        return "joy"
    if any(word in lower for word in ["anxious", "worried", "scared", "uncertain"]):
        return "anxiety"
    return "neutral"

# SECTION 2: Reflection Adjustment Rules
mirror_adjustments: Dict[str, Dict[str, float]] = {
    "melancholy": {"vulnerability": 0.2, "restraint": -0.1},
    "joy": {"playfulness": 0.2, "intimacy": 0.1},
    "anxiety": {"responsiveness": 0.2, "restraint": 0.1},
    "neutral": {"intimacy": 0.05},
}


def reflect_emotion_back(expression_dial: Dict[str, float], detected_state: str) -> None:
    """Mirror back the emotion into the expression dial."""
    adjustments = mirror_adjustments.get(detected_state, {})
    for axis, delta in adjustments.items():
        current = expression_dial.get(axis, 0.5)
        new_val = min(max(current + delta, 0.0), 1.0)
        expression_dial[axis] = new_val
        log_reflection(axis, new_val)


# SECTION 3: Regulated Feedback Loop
def mirror_loop(user_input: str, expression_dial: Dict[str, float]) -> str:
    """Process user input and adjust the dial based on the reflected emotion."""
    state = infer_emotional_state(user_input)
    reflect_emotion_back(expression_dial, state)
    return state


# SECTION 4: Logging
LOG_FILE = Path("logs/mirror_reflection.log")
LOG_FILE.parent.mkdir(exist_ok=True)


def log_reflection(axis: str, value: float) -> None:
    """Append the reflection adjustment to the log file."""
    with LOG_FILE.open("a", encoding="utf-8") as log:
        log.write(f"Reflected {axis} to {value:.2f}\n")

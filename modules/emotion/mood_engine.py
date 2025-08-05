import json
import os
from datetime import datetime, timedelta
from modules.persona.persona_state import get_active_persona

MOOD_STATE = {
    "current": "anchored",
    "last_updated": datetime.now().isoformat()
}

# Mood decay configuration
DECAY_MINUTES = 30
MOOD_DECAY_MAP = {
    "anchored": "waiting",
    "waiting": "hollow",
    "soft": "anchored",
    "wild": "anchored"
}

def load_mood_thresholds():
    path = os.path.join("config", "mood_thresholds.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[Mood Engine] Failed to load thresholds: {e}")
        return {}

def update_mood(trigger, intensity=1):
    persona = get_active_persona()
    thresholds = load_mood_thresholds().get(persona, {})

    weights = thresholds.get("weights", {})
    mood_shift = weights.get(trigger, 0) * intensity

    mood_history = thresholds.get("moods", {})
    current_mood = MOOD_STATE["current"]

    if mood_shift > 0:
        next_mood = mood_history.get("up", {}).get(current_mood, current_mood)
    elif mood_shift < 0:
        next_mood = mood_history.get("down", {}).get(current_mood, current_mood)
    else:
        next_mood = current_mood

    MOOD_STATE["current"] = next_mood
    MOOD_STATE["last_updated"] = datetime.now().isoformat()
    print(f"[Mood Engine] Mood updated → {next_mood}")
    return next_mood

def _apply_decay():
    """Apply mood decay based on time elapsed"""
    last = datetime.fromisoformat(MOOD_STATE["last_updated"])
    elapsed = datetime.now() - last
    if elapsed >= timedelta(minutes=DECAY_MINUTES):
        current = MOOD_STATE["current"]
        next_mood = MOOD_DECAY_MAP.get(current)
        if next_mood and next_mood != current:
            MOOD_STATE["current"] = next_mood
            MOOD_STATE["last_updated"] = datetime.now().isoformat()


def get_current_mood():
    _apply_decay()
    return MOOD_STATE["current"]

def get_last_update() -> str:
    return MOOD_STATE["last_updated"]

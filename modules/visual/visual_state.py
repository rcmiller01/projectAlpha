from modules.emotion.mood_engine import get_current_mood
from modules.persona.persona_state import get_active_persona
from datetime import datetime

MOOD_VISUAL_MAP = {
    "Mia": {
        "anchored": "anchored_calm",
        "soft": "soft_lit",
        "waiting": "golden_gaze",
        "hollow": "porch_empty"
    },
    "Solene": {
        "wild": "burning_focus",
        "flooded": "shadow_light",
        "storming": "storm_wrath",
        "anchored": "feral_stillness"
    }
}

def get_visual_state():
    persona = get_active_persona()
    mood = get_current_mood()
    visual_tag = MOOD_VISUAL_MAP.get(persona, {}).get(mood, "default_idle")

    return {
        "persona": persona,
        "mood": mood,
        "visual": visual_tag,
        "timestamp": datetime.now().isoformat()
    }

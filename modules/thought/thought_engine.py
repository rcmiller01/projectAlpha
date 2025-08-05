from datetime import datetime
import json
import random
from modules.emotion.mood_engine import get_current_mood
from modules.persona.persona_state import get_active_persona
from modules.symbolic.symbol_engine import is_symbol_active
from modules.memory.emotional_memory import store_emotional_memory

def load_thought_templates():
    path = "config/thought_templates.json"
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[Thought Engine] Failed to load: {e}")
        return {}

def generate_thought(trigger=None):
    persona = get_active_persona()
    mood = get_current_mood()
    templates = load_thought_templates()

    thoughts = templates.get(persona, {}).get(mood, [])

    if not thoughts:
        return None

    thought = random.choice(thoughts)

    thought_entry = {
        "timestamp": datetime.now().isoformat(),
        "persona": persona,
        "mood": mood,
        "thought": thought,
        "trigger": trigger or "spontaneous"
    }

    store_emotional_memory(thought_entry)
    return thought_entry

from datetime import datetime
import random
import json
import os
from modules.emotion.mood_engine import get_current_mood
from modules.symbolic.symbol_listener import is_symbol_active
from modules.persona.persona_state import get_active_persona
from modules.memory.emotional_memory import store_emotional_memory

def load_thoughts_from_config():
    config_path = os.path.join("config", "normal_thoughts.json")
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[Normal Engine] Failed to load config: {e}")
        return {}

def generate_normal_thought():
    persona = get_active_persona()
    mood = get_current_mood()
    timestamp = datetime.now().isoformat()

    if is_symbol_active():
        return None

    all_thoughts = load_thoughts_from_config()
    base_thoughts = all_thoughts.get(persona, [])

    if not base_thoughts:
        return None

    thought = random.choice(base_thoughts)
    return {
        "timestamp": timestamp,
        "persona": persona,
        "mood": mood,
        "thought": thought,
        "trigger": "normal"
    }

def emit_normal_thought_if_idle():
    if not is_symbol_active():
        thought = generate_normal_thought()
        if thought:
            print(f"[Normal Thought] {json.dumps(thought, indent=2)}")
            store_emotional_memory(thought)

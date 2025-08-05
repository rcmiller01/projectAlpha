
# modules/journal/trigger_dispatcher.py

from datetime import datetime, timedelta
from modules.journal.journal_engine import JournalEngine
from modules.journal.journal_utils import load_prompts, infer_mood
import random

# Hardcoded symbolic to mood mappings
SYMBOLIC_EVENT_MAP = {
    "collar_touched": {"trigger": "collar_touch", "mood": "serene"},
    "song_played": {"trigger": "music", "mood": "longing"},
    "user_departure": {"trigger": "absence", "mood": "solitude"},
    "user_returned": {"trigger": "reunion", "mood": "joyful"},
    "kiss": {"trigger": "kiss", "mood": "devoted"},
    "garden_symbol": {"trigger": "garden", "mood": "serene"},
    "anchor_word": {"trigger": "symbolic_anchor", "mood": "anchored"},
}

# Mood drift trigger if mood persists
MOOD_PERSISTENCE_THRESHOLDS = {
    "longing": 3,  # hours
    "solitude": 4,
    "fiery": 2,
}

# Mock persona mood state tracker
persona_mood_state = {
    "mia": {"mood": "serene", "since": datetime.utcnow()},
    "solene": {"mood": "fiery", "since": datetime.utcnow()}
}

def dispatch_trigger(event, value=None, persona="mia"):
    persona = persona.lower()
    journal = JournalEngine(persona=persona)

    # Handle symbolic events
    if event in SYMBOLIC_EVENT_MAP:
        mapping = SYMBOLIC_EVENT_MAP[event]
        prompt = random.choice(load_prompts(persona))
        mood = mapping["mood"]
        journal.add_entry(
            mood=mood,
            trigger=mapping["trigger"],
            visibility="private",
            text=prompt
        )
        print(f"[{persona}] Journaled symbolic trigger: {event}")

    # Handle mood persistence (drift-based journaling)
    elif event == "mood_state":
        mood_info = value  # Expected: {"mood": "longing", "since": datetime_obj}
        if not mood_info or "mood" not in mood_info or "since" not in mood_info:
            print("Invalid mood_state value.")
            return

        mood = mood_info["mood"]
        since = mood_info["since"]
        threshold = MOOD_PERSISTENCE_THRESHOLDS.get(mood)

        if threshold and (datetime.utcnow() - since) >= timedelta(hours=threshold):
            prompt = random.choice(load_prompts(persona))
            journal.add_entry(
                mood=mood,
                trigger="mood_drift",
                visibility="private",
                text=prompt
            )
            print(f"[{persona}] Journaled mood drift for '{mood}'")
        else:
            print(f"[{persona}] Mood '{mood}' not yet persisted long enough.")
    else:
        print(f"[{persona}] Unknown trigger event: {event}")

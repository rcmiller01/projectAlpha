from modules.emotion.mood_engine import update_mood
from modules.symbolic.symbol_engine import (
    clear_symbol,
    is_symbol_active,
    load_symbol_definitions,
    trigger_symbol,
)
from modules.thought.thought_engine import generate_thought
from modules.voice.voice_output import speak


def process_symbol(symbol_name):
    if not trigger_symbol(symbol_name):
        return None

    definitions = load_symbol_definitions()
    symbol = definitions.get(symbol_name, {})

    override_mood = symbol.get("override_mood")
    effect_strength = symbol.get("priority", 1)

    if override_mood:
        trigger_key = f"symbol:{symbol_name}"
        update_mood(trigger=trigger_key, intensity=effect_strength)
        print(f"[Symbol Bridge] {symbol_name} → Mood override → {override_mood}")

        # Generate and speak related thought
        thought = generate_thought(trigger=trigger_key)
        if thought:
            print(f"[Symbol Bridge Thought] {thought['thought']}")
            speak(thought["thought"])

    return override_mood

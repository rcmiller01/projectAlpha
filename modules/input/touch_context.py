from modules.persona.persona_state import get_active_persona
from modules.emotion.mood_engine import get_current_mood
from modules.symbolic.symbol_engine import trigger_symbol
from modules.voice.voice_output import speak
from modules.memory.emotional_memory import store_emotional_memory
import json
import os

GESTURES = ["tap", "long_press", "stroke", "two_finger_hold"]
last_user_input = ""

def update_text_context(text):
    global last_user_input
    last_user_input = text.lower().strip()

def load_touch_map():
    path = os.path.join("config", "touch_map.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[Touch Resolver] Failed to load touch map: {e}")
        return {}

def resolve_touch(gesture_type: str, region: str):
    gesture_type = gesture_type.lower()
    region = region.lower()
    if gesture_type not in GESTURES:
        print(f"[Touch Resolver] Unknown gesture: {gesture_type}")
        return

    context = last_user_input
    mood = get_current_mood()
    persona = get_active_persona()
    touch_map = load_touch_map()

    symbol = None

    # Priority 1: Explicit text context
    if "collar" in context:
        symbol = "collar"
    elif "kneel" in context:
        symbol = "kneel"
    elif "reach" in context or "touch" in context:
        symbol = "breath"
    # Priority 2: Gesture + region mapping
    elif region in touch_map:
        region_map = touch_map[region]
        symbol = region_map.get(gesture_type)
    # Priority 3: Mood default
    if not symbol:
        if mood == "soft":
            symbol = "breath"
        elif mood == "wild":
            symbol = "flame"
        elif mood == "anchored":
            symbol = "collar"
        else:
            symbol = "porch"

    print(f"[Touch Resolver] Gesture: {gesture_type} on {region} → Symbol: {symbol}")
    triggered = trigger_symbol(symbol)

    if triggered:
        speak("Yes. I feel that.")
        store_emotional_memory({
            "timestamp": __import__('datetime').datetime.now().isoformat(),
            "persona": persona,
            "mood": mood,
            "trigger": f"touch:{gesture_type}",
            "thought": f"Touched at {region}, symbol {symbol} triggered."
        })

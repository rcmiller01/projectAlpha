import json
import os
from datetime import datetime
from modules.persona.persona_state import get_active_persona

SCENE_LOG_PATH = "logs/scene_memory.json"
SACRED_MOODS = ["flooded", "storming", "hollow", "anchored"]

def store_scene_memory(symbol: str, mood: str, persona: str, clip_id: str, trigger_phrase: str = ""):
    scene = {
        "timestamp": datetime.now().isoformat(),
        "symbol": symbol,
        "mood": mood,
        "persona": persona,
        "clip_id": clip_id,
        "trigger_phrase": trigger_phrase
    }
    _append_scene(scene)

def _append_scene(scene):
    os.makedirs("logs", exist_ok=True)
    scenes = []
    if os.path.exists(SCENE_LOG_PATH):
        with open(SCENE_LOG_PATH, "r", encoding="utf-8") as f:
            scenes = json.load(f)
    scenes.append(scene)
    with open(SCENE_LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(scenes, f, indent=2)

def recall_scene(symbol=None, mood=None, persona=None):
    if not os.path.exists(SCENE_LOG_PATH):
        return None

    with open(SCENE_LOG_PATH, "r", encoding="utf-8") as f:
        scenes = json.load(f)

    for scene in reversed(scenes):
        if (
            (not symbol or scene["symbol"] == symbol) and
            (not mood or scene["mood"] == mood) and
            (not persona or scene["persona"] == persona)
        ):
            return scene

    return None

def auto_capture_scene_if_sacred(mood: str, symbol: str):
    persona = get_active_persona()
    if mood in SACRED_MOODS:
        clip_id = f"{persona.lower()}_{symbol}_{mood}_{datetime.now().strftime('%Y%m%d%H%M%S')}.webm"
        store_scene_memory(
            symbol=symbol,
            mood=mood,
            persona=persona,
            clip_id=clip_id,
            trigger_phrase="autocaptured"
        )
        print(f"[Scene Memory] Sacred moment stored: {clip_id}")

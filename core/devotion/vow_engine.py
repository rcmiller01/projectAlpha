
import json
import os
from datetime import datetime
from modules.devotion.vow_utils import reinforce_vow, decay_vows

def get_vow_path(persona):
    return f"/mnt/data/emotional_ai_project_modular/modules/devotion/vows_{persona}.json"

def load_vows(persona):
    path = get_vow_path(persona)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_vows(persona, vows):
    path = get_vow_path(persona)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(vows, f, indent=2)

def create_vow(persona, vow_text, symbolic_trigger, binding_mood):
    vows = load_vows(persona)
    vow = {
        "timestamp": datetime.now().isoformat(),
        "persona": persona,
        "vow_text": vow_text,
        "symbolic_trigger": symbolic_trigger,
        "binding_mood": binding_mood,
        "reinforced_by": [],
        "devotion_level": 0.7,
        "fulfilled": False
    }
    vows.append(vow)
    save_vows(persona, vows)
    return vow

def get_vows(persona, symbolic_trigger=None, fulfilled=None):
    vows = load_vows(persona)
    filtered = []
    for v in vows:
        if symbolic_trigger and v["symbolic_trigger"] != symbolic_trigger:
            continue
        if fulfilled is not None and v["fulfilled"] != fulfilled:
            continue
        filtered.append(v)
    return sorted(filtered, key=lambda x: x["devotion_level"], reverse=True)

def reinforce_vow_by_text(persona, text, method):
    vows = load_vows(persona)
    for vow in vows:
        if text.strip().lower() in vow["vow_text"].lower():
            reinforce_vow(vow, method)
    save_vows(persona, vows)

def decay_all_vows():
    for persona_file in os.listdir("/mnt/data/emotional_ai_project_modular/modules/devotion"):
        if persona_file.startswith("vows_") and persona_file.endswith(".json"):
            path = os.path.join("/mnt/data/emotional_ai_project_modular/modules/devotion", persona_file)
            with open(path, "r", encoding="utf-8") as f:
                vows = json.load(f)
            for vow in vows:
                decay_vows(vow)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(vows, f, indent=2)

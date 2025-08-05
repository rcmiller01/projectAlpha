
import os
import json
from datetime import datetime

def get_journal_path(persona):
    return f"/mnt/data/emotional_ai_project_modular/journals/{persona.lower()}_journal.json"

def load_journal(persona):
    path = get_journal_path(persona)
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_journal(persona, data):
    path = get_journal_path(persona)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def create_journal_entry(persona, text, mood="neutral", trigger="manual", visibility="private"):
    data = load_journal(persona)
    entry = {
        "timestamp": datetime.now().isoformat(),
        "mood": mood,
        "trigger": trigger,
        "visibility": visibility,
        "persona": persona,
        "text": text
    }
    data.append(entry)
    save_journal(persona, data)
    return entry

def journal_vow_echo(persona, vow_text, mood, symbol, image_path, visibility="private"):
    data = load_journal(persona)
    entry = {
        "timestamp": datetime.now().isoformat(),
        "mood": mood,
        "trigger": f"vow_echo::{symbol}",
        "visibility": visibility,
        "persona": persona,
        "text": f"Echoed vow: '{vow_text}' with symbol: {symbol}",
        "image_path": image_path
    }
    data.append(entry)
    save_journal(persona, data)
    return entry

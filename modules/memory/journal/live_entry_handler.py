import datetime
import json
import os

def save_journal_entry(persona_name, entry_text, mood_tags, symbols):
    timestamp = datetime.datetime.now().isoformat()
    entry = {
        "timestamp": timestamp,
        "mood": mood_tags,
        "symbols_inspired": symbols,
        "entry": entry_text
    }

    file_path = f"journal/{persona_name}_journal_log.json"
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    else:
        data = []

    data.append(entry)

    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)
    return entry

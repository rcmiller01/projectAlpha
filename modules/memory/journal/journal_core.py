"""
journal_core.py

Handles journal entry creation, tagging, emotional weighting,
and private memory blooms for personas.
"""

import json
from datetime import datetime
from typing import List, Optional, Dict
from pathlib import Path

JOURNAL_PATH = "memory/journal/"
Path(JOURNAL_PATH).mkdir(parents=True, exist_ok=True)

class JournalCore:
    def __init__(self, persona_name: str):
        self.persona_name = persona_name
        self.journal_file = f"{JOURNAL_PATH}{persona_name.lower()}_journal_log.json"
        self.entries = self.load_entries()

    def load_entries(self) -> List[Dict]:
        try:
            with open(self.journal_file, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            return []

    def save_entries(self):
        with open(self.journal_file, "w") as f:
            json.dump(self.entries, f, indent=2)

    def log_entry(self, content: str, tags: List[str], emotion: str, private: bool = False, bloom: bool = False):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "content": content,
            "tags": tags,
            "emotion": emotion,
            "private": private,
            "bloom": bloom
        }
        self.entries.append(entry)
        self.save_entries()

    def auto_bloom(self, emotion: str, content: str):
        """Automatically trigger private bloom entry from emotional spike"""
        self.log_entry(content=content, tags=["bloom"], emotion=emotion, private=True, bloom=True)

    def tag_entry(self, index: int, new_tags: List[str]):
        if 0 <= index < len(self.entries):
            self.entries[index]["tags"].extend(new_tags)
            self.save_entries()

    def retrieve_recent(self, count: int = 5) -> List[Dict]:
        return self.entries[-count:]

# Example:
# jc = JournalCore("Mia")
# jc.log_entry("Felt safe under his touch.", ["anchor", "comfort"], "warmth")
# jc.auto_bloom("longing", "I need him. I need to feel him.")

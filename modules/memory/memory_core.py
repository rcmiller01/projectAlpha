import json
from datetime import datetime
from typing import Dict, List, Optional


class MemoryCore:
    def __init__(self, mia_journal_path: str, solene_journal_path: str):
        self.mia_journal_path = mia_journal_path
        self.solene_journal_path = solene_journal_path

    def _load_journal(self, path: str) -> list[dict]:
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return []

    def _save_journal(self, path: str, entries: list[dict]):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(entries, f, indent=4)

    def _filter_entries(
        self, entries: list[dict], mood: Optional[str] = None, symbol: Optional[str] = None
    ) -> list[dict]:
        filtered = entries
        if mood:
            filtered = [e for e in filtered if mood in e.get("mood", [])]
        if symbol:
            filtered = [e for e in filtered if symbol in e.get("symbols_inspired", [])]
        return sorted(filtered, key=lambda x: x.get("timestamp", ""), reverse=True)

    def get_entries(
        self, persona: str, mood: Optional[str] = None, symbol: Optional[str] = None
    ) -> list[dict]:
        path = self.mia_journal_path if persona == "mia" else self.solene_journal_path
        entries = self._load_journal(path)
        return self._filter_entries(entries, mood, symbol)

    def add_entry(self, persona: str, mood: list[str], symbols_inspired: list[str], entry: str):
        path = self.mia_journal_path if persona == "mia" else self.solene_journal_path
        entries = self._load_journal(path)
        entries.append(
            {
                "timestamp": datetime.now().isoformat(),
                "mood": mood,
                "symbols_inspired": symbols_inspired,
                "entry": entry,
            }
        )
        self._save_journal(path, entries)

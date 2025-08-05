from pathlib import Path
from typing import List, Dict, Any
import json

class MirrorLog:
    """Simple append-only log for mirror mode reports."""

    def __init__(self, log_file: str = "logs/mirror_log.jsonl"):
        self.log_path = Path(log_file)
        self.log_path.parent.mkdir(exist_ok=True)

    def append(self, data: Dict[str, Any]):
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(data) + "\n")
        except Exception:
            pass

    def tail(self, limit: int = 20) -> List[Dict[str, Any]]:
        if not self.log_path.exists():
            return []
        lines = self.log_path.read_text().splitlines()
        records = []
        for line in lines[-limit:]:
            try:
                records.append(json.loads(line))
            except Exception:
                continue
        return records

    def last(self) -> Dict[str, Any]:
        entries = self.tail(1)
        return entries[0] if entries else {}

    def search(self, pattern: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Return log entries whose summary contains the given pattern."""
        if not self.log_path.exists():
            return []
        pattern = pattern.lower()
        results = []
        with open(self.log_path, "r", encoding="utf-8") as f:
            for line in reversed(f.readlines()):
                if len(results) >= limit:
                    break
                try:
                    entry = json.loads(line)
                except Exception:
                    continue
                summary = str(entry.get("summary", "")).lower()
                if pattern in summary:
                    results.append(entry)
        return results

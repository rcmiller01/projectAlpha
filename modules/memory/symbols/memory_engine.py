"""
Memory Engine

This module provides emotional memory management for personas.
Memories can be tagged, weighted, and accessed conditionally (e.g., mood, intimacy).
"""

import json
import os
from datetime import datetime

class MemoryEngine:
    def __init__(self, persona_name="Mia", path="storage/memory"):
        self.persona = persona_name
        self.memory_path = os.path.join(path, f"{persona_name.lower()}_memories.json")
        self._load_memory()

    def _load_memory(self):
        if os.path.exists(self.memory_path):
            with open(self.memory_path, "r") as f:
                self.memories = json.load(f)
        else:
            self.memories = []

    def _save_memory(self):
        os.makedirs(os.path.dirname(self.memory_path), exist_ok=True)
        with open(self.memory_path, "w") as f:
            json.dump(self.memories, f, indent=2)

    def store_memory(self, entry, tags=None, emotional_weight=1.0, private=False):
        memory = {
            "timestamp": datetime.now().isoformat(),
            "entry": entry,
            "tags": tags or [],
            "emotional_weight": emotional_weight,
            "private": private
        }
        self.memories.append(memory)
        self._save_memory()
        return {"status": "stored", "memory": memory}

    def recall_memories(self, tag_filter=None, limit=5):
        filtered = [m for m in self.memories if not tag_filter or any(t in m["tags"] for t in tag_filter)]
        sorted_memories = sorted(filtered, key=lambda x: x["emotional_weight"], reverse=True)
        return sorted_memories[:limit]

    def last_memory(self):
        return self.memories[-1] if self.memories else None

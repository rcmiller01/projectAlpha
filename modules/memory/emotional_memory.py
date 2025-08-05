import json
import os
from datetime import datetime

MEMORY_LOG = []

def store_emotional_memory(entry):
    MEMORY_LOG.append(entry)
    _persist(entry)

def _persist(entry):
    path = "logs/emotional_log.jsonl"
    os.makedirs("logs", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")

def get_recent_memories(limit=10):
    return MEMORY_LOG[-limit:]

def get_memories_by_filter(persona=None, mood=None, trigger=None):
    return [
        m for m in MEMORY_LOG
        if (not persona or m["persona"] == persona) and
           (not mood or m["mood"] == mood) and
           (not trigger or m["trigger"] == trigger)
    ]

def reset_memory():
    global MEMORY_LOG
    MEMORY_LOG = []
    print("[Memory] In-memory log cleared.")

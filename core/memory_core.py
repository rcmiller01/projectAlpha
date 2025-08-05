"""
Status: Todo
Note: Central memory access, sort, retrieval
"""
_MEMORIES = []


def add_memory(memory: dict):
    _MEMORIES.append(memory)


def get_recent_memories(persona: str, limit: int = 5):
    filtered = [m for m in _MEMORIES if m.get("persona") == persona]
    return filtered[-limit:]


"""High level memory interface using MemoryManager."""

from .memory_manager import memory_manager


def record_interaction(user_id: str, closeness: float = 0.05):
    memory_manager.record_closeness(user_id, closeness)


def update_trust(user_id: str, trust: float):
    memory_manager.set_trust_score(user_id, trust)

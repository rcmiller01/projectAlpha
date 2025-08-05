"""Deep Memory Framework
-----------------------

Stores factual, symbolic, emotional and temporal aspects of memories.
Each memory fragment can hold thematic tags and an emotional intensity score.
This simplified implementation uses in-memory storage but can be extended to a
persistent database.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any


@dataclass
class MemoryFragment:
    user_id: str
    content: str
    tags: List[str] = field(default_factory=list)
    emotional_state: Dict[str, float] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    symbolic_links: List[str] = field(default_factory=list)


class DeepMemoryFramework:
    """Manage multi-dimensional memory fragments."""

    def __init__(self):
        self.fragments: List[MemoryFragment] = []

    def store_memory(self, fragment: MemoryFragment) -> None:
        self.fragments.append(fragment)

    def recall_by_tag(self, tag: str) -> List[MemoryFragment]:
        return [f for f in self.fragments if tag in f.tags]

    def latest_fragment(self, user_id: str) -> MemoryFragment | None:
        user_fragments = [f for f in self.fragments if f.user_id == user_id]
        return user_fragments[-1] if user_fragments else None

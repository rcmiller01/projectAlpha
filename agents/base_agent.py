"""Base classes for Dolphin specialized agents."""

from typing import Any, List, Optional


class BaseAgent:
    """Common functionality for specialized agents."""

    def __init__(self, memory_system: Optional[Any] = None) -> None:
        self.memory_system = memory_system
        self.emotion_scores = {}  # TODO: Connect to emotion engine for scoring
        # TODO: Include future training scaffolds

    def recall(self, query: str) -> List[Any]:
        """Return memories related to the query."""
        if self.memory_system:
            return self.memory_system.recall(query)
        # TODO: integrate with memory recall system
        return []

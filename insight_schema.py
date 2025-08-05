from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any
import uuid

@dataclass
class Insight:
    """Data model for a reflection insight."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    type: str = ""
    context: str = ""
    emotion_vector: Dict[str, float] = field(default_factory=dict)
    details: str = ""
    intensity: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "type": self.type,
            "context": self.context,
            "emotion_vector": self.emotion_vector,
            "details": self.details,
            "intensity": self.intensity,
        }

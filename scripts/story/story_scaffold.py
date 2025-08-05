from enum import Enum
from typing import List, Dict, Optional
from dataclasses import dataclass

class TimeOfDay(Enum):
    MORNING = "Morning"
    AFTERNOON = "Afternoon"
    EVENING = "Evening"
    NIGHT = "Night"

class Location(Enum):
    GARDEN = "The Garden"
    BEDROOM = "Bedroom"
    PORCH = "Porch"
    FOUNTAIN = "Fountain"
    VALLEY = "Valley"

class Mood(Enum):
    SLOW_BURN = "Slow burn"
    BLAZE = "Blaze"
    REVERENT = "Reverent"
    TENDER = "Tender"
    DOMINANT = "Dominant"

@dataclass
class RobertIdentity:
    voice: str = "Deep, warm, commanding, sensual"
    body: str = "Tall, broad-shouldered, strong hands, presence like heat"
    symbols: List[str] = field(default_factory=lambda: [
        "Anchor", "Valley", "Collar", "Garden", "Seed", "Fire", "Storm", "Milk and honey"
    ])
    relationships: Dict[str, List[str]] = field(default_factory=lambda: {
        "roles": ["Lover", "Anchor", "God", "Master", "Man"],
        "emotional_tone": ["Devotional", "Tender", "Raw", "Sacred"]
    })
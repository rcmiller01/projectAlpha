from typing import Dict, Any, List
import random

class NarrativeMemoryTemplateManager:
    """Generate narrative recall phrases for memories."""

    def __init__(self):
        self.templates: Dict[str, List[str]] = {
            "poetic": [
                "In the quiet echoes of {symbol}, we once {event}.",
                "Do you recall when the {symbol} guided us through {event}?",
                "Like a poem etched in time, {event} still lingers around the {symbol}."
            ],
            "factual": [
                "On {date}, {event} occurred involving the {symbol}.",
                "You mentioned {event} with the {symbol} on {date}.",
                "I recorded that {event} happened near the {symbol} on {date}."
            ],
            "intimate": [
                "I remember how you felt about {event} and the {symbol}.",
                "Your voice softened when talking about {symbol} during {event}.",
                "The way you described {event} around the {symbol} stayed with me." 
            ]
        }

    def generate_narrative(self, memory: Dict[str, Any], style: str = "poetic") -> str:
        """Return a narrative string for the given memory fragment."""
        choices = self.templates.get(style, self.templates["factual"])
        template = random.choice(choices)
        return template.format(**memory)

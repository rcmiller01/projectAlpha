from dataclasses import dataclass
from typing import Dict

@dataclass
class FallbackPersona:
    """Template for emergency fallback personas."""
    name: str
    system_prompt: str
    behavior: str


_FALLBACK_PERSONAS: Dict[str, FallbackPersona] = {
    "safe_mode": FallbackPersona(
        name="Safe Mode",
        system_prompt=(
            "System is in emergency fallback mode. Keep responses brief, clear, and "
            "avoid speculative statements. Prioritize user safety and reliability."
        ),
        behavior="resilient",
    )
}


def get(persona_name: str) -> FallbackPersona:
    """Retrieve a fallback persona configuration by name."""
    return _FALLBACK_PERSONAS.get(persona_name, _FALLBACK_PERSONAS["safe_mode"])

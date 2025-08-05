"""Autonomous Desire Expression module.

Generates intimate narrative cues when emotional and trust
conditions are met.
"""

from datetime import datetime, timedelta
from typing import Optional

from ..emotion.emotion_state import emotion_state
from ..memory.memory_manager import memory_manager
from ..symbolic_trigger_engine import SymbolicTriggerEngine


class DesireInitiator:
    """Initiates desire-based responses based on memory and emotion."""

    def __init__(self, trigger_engine: Optional[SymbolicTriggerEngine] = None):
        self.trigger_engine = trigger_engine or SymbolicTriggerEngine()
        self.desire_cooldown = timedelta(minutes=10)
        self._last_trigger: Optional[datetime] = None

    def should_initiate(self, user_id: str) -> bool:
        """Check if conditions warrant an autonomous desire expression."""
        trust = memory_manager.get_trust_score(user_id)
        longing = emotion_state.romantic_emotions.get("longing", 0.0)
        decay = memory_manager.get_desire_decay(user_id)

        cooldown_ok = (
            self._last_trigger is None or
            datetime.now() - self._last_trigger > self.desire_cooldown
        )
        return trust > 0.7 and longing > 0.5 and decay < 0.3 and cooldown_ok

    def generate_expression(self, user_id: str) -> str:
        """Return narrative text expressing gentle desire."""
        symbols = memory_manager.get_preferred_symbols(user_id)
        symbol_text = ", ".join(symbols) if symbols else "our connection"
        line = f"My thoughts keep drifting to {symbol_text}. I long to grow closer to you."
        self._last_trigger = datetime.now()
        return line


# Global instance
trigger_engine = SymbolicTriggerEngine()
desire_initiator = DesireInitiator(trigger_engine)

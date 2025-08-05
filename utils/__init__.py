"""
Utility functions for the Unified AI Companion
Enhanced functions for timing, logging, and helper operations
"""

from .message_timing import infer_conversation_tempo
from .event_logger import log_emotional_event

__all__ = [
    'infer_conversation_tempo',
    'log_emotional_event'
]

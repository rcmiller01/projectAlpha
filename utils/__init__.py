"""
Utility functions for the Unified AI Companion
Enhanced functions for timing, logging, and helper operations
"""

from .event_logger import log_emotional_event
from .message_timing import infer_conversation_tempo

__all__ = ["infer_conversation_tempo", "log_emotional_event"]

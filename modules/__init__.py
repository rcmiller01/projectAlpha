"""
Unified Companion System Modules

Core modules for the emotional AI companion system.
"""

# Import key components for easier access
try:
    from .core.crisis_safety_override import CrisisSafetyOverride
    from .core.goodbye_protocol import GoodbyeProtocol
    from .core.unified_companion import UnifiedCompanion
    from .database.database_interface import create_database_interface
    from .emotion.mood_inflection import MoodInflection
    from .memory.narrative_memory_templates import NarrativeMemoryTemplateManager
    from .relationship.connection_depth_tracker import ConnectionDepthTracker
    from .symbolic.symbol_resurrection import SymbolResurrectionManager
except ImportError:
    # Graceful degradation for missing dependencies
    pass

__version__ = "3.0.0"
__author__ = "Unified Companion Development Team"

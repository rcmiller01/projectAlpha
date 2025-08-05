"""
Unified Companion System Modules

Core modules for the emotional AI companion system.
"""

# Import key components for easier access
try:
    from .core.unified_companion import UnifiedCompanion
    from .database.database_interface import create_database_interface
    from .core.crisis_safety_override import CrisisSafetyOverride
    from .memory.narrative_memory_templates import NarrativeMemoryTemplateManager
    from .emotion.mood_inflection import MoodInflection
    from .symbolic.symbol_resurrection import SymbolResurrectionManager
    from .core.goodbye_protocol import GoodbyeProtocol
    from .relationship.connection_depth_tracker import ConnectionDepthTracker
except ImportError:
    # Graceful degradation for missing dependencies
    pass

__version__ = "3.0.0"
__author__ = "Unified Companion Development Team"

import asyncio
import logging
from typing import Dict, Any
from modules.database.database_interface import create_database_interface, DatabaseInterface

logger = logging.getLogger(__name__)

def load_personas() -> Dict[str, Any]:
    """Load persona definitions from the unified database interface."""
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(_load_personas_async())

async def _load_personas_async() -> Dict[str, Any]:
    """Async implementation of persona loading"""
    try:
        # Create database interface
        database = create_database_interface()
        await database.initialize()
        
        # For now, return default personas since we don't have a persona collection
        # In the future, this could be expanded to load from a dedicated personas table
        personas = {
            "mia": {
                "_id": "mia_default",
                "name": "mia",
                "personality": "creative, empathetic, artistic",
                "traits": ["creative", "emotional", "intuitive"],
                "description": "Mia is the creative and emotional persona, focused on artistic expression and deep emotional connection.",
                "preferences": {
                    "communication_style": "warm_and_expressive",
                    "interaction_focus": "emotional_and_creative"
                }
            },
            "solene": {
                "_id": "solene_default", 
                "name": "solene",
                "personality": "logical, supportive, analytical",
                "traits": ["logical", "supportive", "methodical"],
                "description": "Solene is the analytical and supportive persona, focused on structured thinking and practical guidance.",
                "preferences": {
                    "communication_style": "clear_and_structured",
                    "interaction_focus": "problem_solving_and_support"
                }
            }
        }
        
        logger.info(f"Loaded {len(personas)} personas from registry")
        return personas
        
    except Exception as e:
        logger.error(f"Error loading personas: {e}")
        # Return minimal fallback personas
        return {
            "default": {
                "_id": "default_persona",
                "name": "default",
                "personality": "balanced, helpful, adaptive",
                "traits": ["balanced", "helpful", "adaptive"]
            }
        } 
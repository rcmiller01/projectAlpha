import asyncio
import logging
from typing import Optional, Dict, Any
from ..database.database_interface import create_database_interface, DatabaseInterface

logger = logging.getLogger(__name__)

class PersonaState:
    """
    Unified persona state management using the database interface
    """
    
    def __init__(self, database: Optional[DatabaseInterface] = None):
        if database:
            self.database = database
        else:
            # Create default database interface
            self.database = create_database_interface()
        
    async def initialize(self):
        """Initialize the database connection"""
        await self.database.initialize()

async def get_active_persona() -> Optional[str]:
    """Get the currently active persona"""
    try:
        persona_state = PersonaState()
        await persona_state.initialize()
        
        # For now, we'll use user profile to store active persona
        # This could be expanded to a dedicated configuration system
        user_profile = await persona_state.database.get_user_profile("system")
        if user_profile and hasattr(user_profile, 'preferences'):
            return user_profile.preferences.get("active_persona")
        return None
    except Exception as e:
        logger.error(f"Error getting active persona: {e}")
        return None

async def set_active_persona(name: str):
    """Set the active persona"""
    try:
        persona_state = PersonaState()
        await persona_state.initialize()
        
        # Update system user profile with active persona
        await persona_state.database.update_user_profile("system", {
            "preferences": {"active_persona": name}
        })
    except Exception as e:
        logger.error(f"Error setting active persona: {e}")

async def load_personas() -> Dict[str, Any]:
    """Load available personas from configuration"""
    try:
        persona_state = PersonaState()
        await persona_state.initialize()
        
        # For now, return default personas
        # This could be expanded to load from database/configuration
        personas = {
            "mia": {
                "name": "mia",
                "personality": "creative, empathetic, artistic",
                "traits": ["creative", "emotional", "intuitive"]
            },
            "solene": {
                "name": "solene", 
                "personality": "logical, supportive, analytical",
                "traits": ["logical", "supportive", "methodical"]
            }
        }
        return personas
    except Exception as e:
        logger.error(f"Error loading personas: {e}")
        return {}
    return personas 
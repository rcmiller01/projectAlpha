"""
MongoDB Database Implementation

Production-ready persistent database implementation for the unified companion system.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.errors import DuplicateKeyError, PyMongoError

from .database_interface import (
    DatabaseInterface, UserProfile, InteractionRecord, SessionRecord,
    PsychologicalState, MemoryFragment, EmotionalRiskEntry
)


class MongoDatabase:
    """
    MongoDB implementation for persistent storage
    """
    
    def __init__(self, connection_string: str, database_name: str = "unified_companion"):
        self.connection_string = connection_string
        self.database_name = database_name
        self.client = None
        self.db = None
        self.logger = logging.getLogger(__name__)
        
        # Collection names
        self.collections = {
            "users": "user_profiles",
            "interactions": "interactions", 
            "sessions": "sessions",
            "psychological_states": "psychological_states",
            "memory_fragments": "memory_fragments",
            "emotional_patterns": "emotional_patterns",
            "crisis_logs": "crisis_logs",
            "emotional_risk": "emotional_risk"
        }
    
    async def initialize(self) -> None:
        """Initialize the MongoDB connection and ensure indexes"""
        try:
            self.client = AsyncIOMotorClient(self.connection_string)
            self.db = self.client[self.database_name]
            
            # Test connection
            await self.client.admin.command('ping')
            self.logger.info(f"Connected to MongoDB: {self.database_name}")
            
            # Create indexes for performance
            await self._create_indexes()
            
        except Exception as e:
            self.logger.error(f"Failed to initialize MongoDB: {e}")
            raise
    
    async def _create_indexes(self):
        """Create database indexes for optimal performance"""
        try:
            # User profiles
            await self.db[self.collections["users"]].create_index("user_id", unique=True)
            
            # Interactions
            await self.db[self.collections["interactions"]].create_index([
                ("user_id", 1), ("timestamp", -1)
            ])
            await self.db[self.collections["interactions"]].create_index("session_id")
            
            # Sessions
            await self.db[self.collections["sessions"]].create_index([
                ("user_id", 1), ("start_time", -1)
            ])
            
            # Psychological states
            await self.db[self.collections["psychological_states"]].create_index([
                ("user_id", 1), ("timestamp", -1)
            ])
            
            # Memory fragments
            await self.db[self.collections["memory_fragments"]].create_index([
                ("user_id", 1), ("importance_score", -1)
            ])
            await self.db[self.collections["memory_fragments"]].create_index([
                ("user_id", 1), ("memory_type", 1)
            ])
            await self.db[self.collections["memory_fragments"]].create_index("tags")

            # Crisis logs
            await self.db[self.collections["crisis_logs"]].create_index([
                ("user_id", 1), ("timestamp", -1)
            ])

            # Emotional risk registry
            await self.db[self.collections["emotional_risk"]].create_index([
                ("user_id", 1), ("timestamp", -1)
            ])
            
            self.logger.info("Database indexes created successfully")
            
        except Exception as e:
            self.logger.error(f"Error creating indexes: {e}")
    
    async def create_user_profile(self, user_profile: UserProfile) -> bool:
        """Create a new user profile"""
        try:
            await self.db[self.collections["users"]].insert_one(user_profile.to_dict())
            self.logger.info(f"Created user profile: {user_profile.user_id}")
            return True
        except DuplicateKeyError:
            self.logger.warning(f"User profile already exists: {user_profile.user_id}")
            return False
        except Exception as e:
            self.logger.error(f"Error creating user profile: {e}")
            return False
    
    async def get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """Get user profile by user_id"""
        try:
            doc = await self.db[self.collections["users"]].find_one({"user_id": user_id})
            if doc:
                # Remove MongoDB _id field
                doc.pop("_id", None)
                return UserProfile.from_dict(doc)
            return None
        except Exception as e:
            self.logger.error(f"Error getting user profile: {e}")
            return None
    
    async def update_user_profile(self, user_id: str, updates: Dict[str, Any]) -> bool:
        """Update user profile"""
        try:
            updates["last_active"] = datetime.now().isoformat()
            result = await self.db[self.collections["users"]].update_one(
                {"user_id": user_id},
                {"$set": updates}
            )
            return result.modified_count > 0
        except Exception as e:
            self.logger.error(f"Error updating user profile: {e}")
            return False
    
    async def save_interaction(self, interaction: InteractionRecord) -> bool:
        """Save an interaction record"""
        try:
            await self.db[self.collections["interactions"]].insert_one(interaction.to_dict())
            return True
        except Exception as e:
            self.logger.error(f"Error saving interaction: {e}")
            return False
    
    async def get_recent_interactions(self, user_id: str, limit: int = 20) -> List[InteractionRecord]:
        """Get recent interactions for a user"""
        try:
            cursor = self.db[self.collections["interactions"]].find(
                {"user_id": user_id}
            ).sort("timestamp", -1).limit(limit)
            
            interactions = []
            async for doc in cursor:
                doc.pop("_id", None)
                interactions.append(InteractionRecord.from_dict(doc))
            
            return interactions
        except Exception as e:
            self.logger.error(f"Error getting recent interactions: {e}")
            return []
    
    async def save_psychological_state(self, state: PsychologicalState) -> bool:
        """Save psychological state snapshot"""
        try:
            await self.db[self.collections["psychological_states"]].insert_one(state.to_dict())
            return True
        except Exception as e:
            self.logger.error(f"Error saving psychological state: {e}")
            return False
    
    async def get_latest_psychological_state(self, user_id: str) -> Optional[PsychologicalState]:
        """Get latest psychological state for user"""
        try:
            doc = await self.db[self.collections["psychological_states"]].find_one(
                {"user_id": user_id},
                sort=[("timestamp", -1)]
            )
            if doc:
                doc.pop("_id", None)
                return PsychologicalState.from_dict(doc)
            return None
        except Exception as e:
            self.logger.error(f"Error getting psychological state: {e}")
            return None
    
    async def save_memory_fragment(self, memory: MemoryFragment) -> bool:
        """Save a memory fragment"""
        try:
            await self.db[self.collections["memory_fragments"]].insert_one(memory.to_dict())
            return True
        except Exception as e:
            self.logger.error(f"Error saving memory fragment: {e}")
            return False
    
    async def get_relevant_memories(self, user_id: str, memory_type: Optional[str] = None, 
                                  tags: Optional[List[str]] = None, limit: int = 10) -> List[MemoryFragment]:
        """Get relevant memory fragments for context"""
        try:
            query = {"user_id": user_id}
            
            if memory_type:
                query["memory_type"] = memory_type
            
            if tags:
                query["tags"] = {"$in": tags}
            
            cursor = self.db[self.collections["memory_fragments"]].find(query).sort(
                "importance_score", -1
            ).limit(limit)
            
            memories = []
            async for doc in cursor:
                doc.pop("_id", None)
                memories.append(MemoryFragment.from_dict(doc))
            
            return memories
        except Exception as e:
            self.logger.error(f"Error getting relevant memories: {e}")
            return []
    
    async def update_memory_access(self, memory_id: str) -> bool:
        """Update memory access tracking"""
        try:
            result = await self.db[self.collections["memory_fragments"]].update_one(
                {"memory_id": memory_id},
                {
                    "$set": {"last_accessed": datetime.now().isoformat()},
                    "$inc": {"access_count": 1}
                }
            )
            return result.modified_count > 0
        except Exception as e:
            self.logger.error(f"Error updating memory access: {e}")
            return False
    
    async def log_crisis_event(self, user_id: str, crisis_data: Dict[str, Any]) -> bool:
        """Log crisis intervention event"""
        try:
            crisis_log = {
                "user_id": user_id,
                "timestamp": datetime.now().isoformat(),
                "crisis_level": crisis_data.get("level"),
                "crisis_score": crisis_data.get("score"),
                "detected_types": crisis_data.get("detected_types", []),
                "intervention_taken": crisis_data.get("intervention_taken"),
                "user_input": crisis_data.get("user_input"),
                "system_response": crisis_data.get("system_response"),
                "follow_up_needed": crisis_data.get("follow_up_needed", False)
            }
            
            await self.db[self.collections["crisis_logs"]].insert_one(crisis_log)
            self.logger.warning(f"Crisis event logged for user {user_id}: {crisis_data.get('level')}")
            return True
        except Exception as e:
            self.logger.error(f"Error logging crisis event: {e}")
            return False
    
    async def get_crisis_history(self, user_id: str, days: int = 30) -> List[Dict[str, Any]]:
        """Get crisis intervention history"""
        try:
            start_date = datetime.now() - timedelta(days=days)
            
            cursor = self.db[self.collections["crisis_logs"]].find({
                "user_id": user_id,
                "timestamp": {"$gte": start_date.isoformat()}
            }).sort("timestamp", -1)
            
            crisis_logs = []
            async for doc in cursor:
                doc.pop("_id", None)
                crisis_logs.append(doc)
            
            return crisis_logs
        except Exception as e:
            self.logger.error(f"Error getting crisis history: {e}")
            return []

    async def log_emotional_risk(self, entry: EmotionalRiskEntry) -> bool:
        """Log emotional risk entry"""
        try:
            await self.db[self.collections["emotional_risk"]].insert_one(entry.to_dict())
            return True
        except Exception as e:
            self.logger.error(f"Error logging emotional risk: {e}")
            return False

    async def get_emotional_risk_history(self, user_id: str, limit: int = 20) -> List[EmotionalRiskEntry]:
        """Retrieve emotional risk history"""
        try:
            cursor = self.db[self.collections["emotional_risk"]].find({"user_id": user_id}).sort(
                "timestamp", -1
            ).limit(limit)

            entries: List[EmotionalRiskEntry] = []
            async for doc in cursor:
                doc.pop("_id", None)
                entries.append(EmotionalRiskEntry.from_dict(doc))

            return entries
        except Exception as e:
            self.logger.error(f"Error retrieving emotional risk history: {e}")
            return []
    
    async def save_emotional_pattern(self, user_id: str, pattern_data: Dict[str, Any]) -> bool:
        """Save long-term emotional pattern analysis"""
        try:
            pattern_doc = {
                "user_id": user_id,
                "timestamp": datetime.now().isoformat(),
                "pattern_type": pattern_data.get("pattern_type"),
                "emotional_weights": pattern_data.get("emotional_weights", {}),
                "attachment_indicators": pattern_data.get("attachment_indicators", {}),
                "stress_patterns": pattern_data.get("stress_patterns", {}),
                "growth_indicators": pattern_data.get("growth_indicators", {}),
                "recommendations": pattern_data.get("recommendations", [])
            }
            
            await self.db[self.collections["emotional_patterns"]].insert_one(pattern_doc)
            return True
        except Exception as e:
            self.logger.error(f"Error saving emotional pattern: {e}")
            return False
    
    async def get_emotional_patterns(self, user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get long-term emotional patterns for user"""
        try:
            cursor = self.db[self.collections["emotional_patterns"]].find(
                {"user_id": user_id}
            ).sort("timestamp", -1).limit(limit)
            
            patterns = []
            async for doc in cursor:
                doc.pop("_id", None)
                patterns.append(doc)
            
            return patterns
        except Exception as e:
            self.logger.error(f"Error getting emotional patterns: {e}")
            return []
    
    async def cleanup_old_data(self, days_to_keep: int = 90) -> Dict[str, int]:
        """Clean up old interaction data while preserving important memories"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            cutoff_iso = cutoff_date.isoformat()
            
            # Clean up old interactions (keep recent and crisis-related)
            interaction_result = await self.db[self.collections["interactions"]].delete_many({
                "timestamp": {"$lt": cutoff_iso},
                "interaction_type": {"$ne": "crisis_support"}
            })
            
            # Clean up old psychological states (keep monthly snapshots)
            # This is more complex - we'd keep one state per month for long-term tracking
            
            return {
                "interactions_cleaned": interaction_result.deleted_count,
                "cleanup_date": cutoff_iso
            }
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
            return {"error": str(e)}
    
    async def close(self):
        """Close database connection"""
        if self.client:
            self.client.close()
            self.logger.info("MongoDB connection closed")

"""
Unified Companion Database Schema

MongoDB schema design for the unified companion system with collections
for user data, interactions, psychological state, and system memory.
"""

from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict, field
from enum import Enum
import asyncio
import logging

# Note: motor package will be installed in production
# from motor.motor_asyncio import AsyncIOMotorClient

class InteractionType(Enum):
    EMOTIONAL_SUPPORT = "emotional_support"
    TECHNICAL_ASSISTANCE = "technical_assistance"
    CREATIVE_COLLABORATION = "creative_collaboration"
    INTEGRATED_SUPPORT = "integrated_support"
    GENERAL_CONVERSATION = "general_conversation"
    CRISIS_SUPPORT = "crisis_support"

class EmotionalState(Enum):
    STABLE = "stable"
    STRESSED = "stressed"
    ANXIOUS = "anxious"
    DEPRESSED = "depressed"
    EXCITED = "excited"
    HAPPY = "happy"
    OVERWHELMED = "overwhelmed"
    CONFUSED = "confused"

@dataclass
class UserProfile:
    """Core user profile information"""
    user_id: str
    created_at: datetime
    last_active: datetime
    display_name: Optional[str] = None
    preferences: Optional[Dict[str, Any]] = field(default_factory=dict)
    adaptive_profile: Optional[Dict[str, Any]] = field(default_factory=dict)
    
    def to_dict(self):
        return {
            "user_id": self.user_id,
            "created_at": self.created_at,
            "last_active": self.last_active,
            "display_name": self.display_name,
            "preferences": self.preferences or {},
            "adaptive_profile": self.adaptive_profile or {}
        }

@dataclass
class InteractionRecord:
    """Individual interaction record"""
    interaction_id: str
    user_id: str
    session_id: str
    timestamp: datetime
    user_input: str
    companion_response: str
    interaction_type: InteractionType
    context_analysis: Dict[str, Any]
    emotional_state: Dict[str, float]
    technical_context: Dict[str, Any]
    creative_context: Dict[str, Any]
    guidance_used: Dict[str, Any]
    response_metrics: Dict[str, Any]
    
    def to_dict(self):
        return {
            "interaction_id": self.interaction_id,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "timestamp": self.timestamp,
            "user_input": self.user_input,
            "companion_response": self.companion_response,
            "interaction_type": self.interaction_type.value,
            "context_analysis": self.context_analysis,
            "emotional_state": self.emotional_state,
            "technical_context": self.technical_context,
            "creative_context": self.creative_context,
            "guidance_used": self.guidance_used,
            "response_metrics": self.response_metrics
        }

@dataclass
class SessionRecord:
    """Session-level interaction data"""
    session_id: str
    user_id: str
    start_time: datetime
    end_time: Optional[datetime]
    total_interactions: int
    primary_focuses: List[str]
    emotional_trajectory: List[Dict[str, Any]]
    session_summary: Dict[str, Any]
    
    def to_dict(self):
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "total_interactions": self.total_interactions,
            "primary_focuses": self.primary_focuses,
            "emotional_trajectory": self.emotional_trajectory,
            "session_summary": self.session_summary
        }

@dataclass
class PsychologicalState:
    """Comprehensive psychological state tracking"""
    user_id: str
    timestamp: datetime
    emotional_indicators: Dict[str, float]
    stress_levels: Dict[str, float]
    attachment_patterns: Dict[str, Any]
    creative_state: Dict[str, Any]
    technical_confidence: Dict[str, float]
    support_needs: List[str]
    risk_factors: Dict[str, float]
    growth_indicators: Dict[str, float]
    
    def to_dict(self):
        return {
            "user_id": self.user_id,
            "timestamp": self.timestamp,
            "emotional_indicators": self.emotional_indicators,
            "stress_levels": self.stress_levels,
            "attachment_patterns": self.attachment_patterns,
            "creative_state": self.creative_state,
            "technical_confidence": self.technical_confidence,
            "support_needs": self.support_needs,
            "risk_factors": self.risk_factors,
            "growth_indicators": self.growth_indicators
        }

@dataclass
class MemoryFragment:
    """Individual memory fragment for context continuity"""
    memory_id: str
    user_id: str
    content: str
    memory_type: str  # "emotional", "technical", "creative", "personal", "contextual"
    importance_score: float
    created_at: datetime
    last_accessed: datetime
    access_count: int
    related_interactions: List[str]
    tags: List[str]
    tone: Optional[str] = None
    sentiment: float = 0.0
    time_of_day: Optional[str] = None
    
    def to_dict(self):
        return {
            "memory_id": self.memory_id,
            "user_id": self.user_id,
            "content": self.content,
            "memory_type": self.memory_type,
            "importance_score": self.importance_score,
            "created_at": self.created_at,
            "last_accessed": self.last_accessed,
            "access_count": self.access_count,
            "related_interactions": self.related_interactions,
            "tags": self.tags,
            "tone": self.tone,
            "sentiment": self.sentiment,
            "time_of_day": self.time_of_day
        }

class UnifiedCompanionDatabase:
    """
    Database interface for the unified companion system using MongoDB
    """
    
    def __init__(self, connection_string: str, database_name: str = "unified_companion"):
        self.connection_string = connection_string
        self.database_name = database_name
        self.client: Optional[Any] = None
        self.db: Optional[Any] = None
        self.logger = logging.getLogger(__name__)
        
        # Collection names
        self.collections = {
            "users": "user_profiles",
            "interactions": "interaction_records",
            "sessions": "session_records",
            "psychology": "psychological_states",
            "memory": "memory_fragments",
            "analytics": "system_analytics"
        }
    
    async def initialize(self):
        """Initialize database connection and create indexes"""
        try:
            # Import motor here to avoid import errors in development
            from motor.motor_asyncio import AsyncIOMotorClient
            
            self.client = AsyncIOMotorClient(self.connection_string)
            self.db = self.client[self.database_name]
            
            # Create indexes for optimal performance
            await self._create_indexes()
            
            self.logger.info("Database connection initialized successfully")
            
        except ImportError:
            self.logger.warning("Motor package not installed - database functionality disabled")
            # In development mode, create mock database interface
            self.db = MockDatabase()
            
        except Exception as e:
            self.logger.error(f"Failed to initialize database: {e}")
            raise
    
    async def _create_indexes(self):
        """Create database indexes for optimal query performance"""
        
        # User profiles indexes
        await self.db[self.collections["users"]].create_index("user_id", unique=True)
        await self.db[self.collections["users"]].create_index("last_active")
        
        # Interaction records indexes
        await self.db[self.collections["interactions"]].create_index("user_id")
        await self.db[self.collections["interactions"]].create_index("session_id")
        await self.db[self.collections["interactions"]].create_index("timestamp")
        await self.db[self.collections["interactions"]].create_index([("user_id", 1), ("timestamp", -1)])
        await self.db[self.collections["interactions"]].create_index("interaction_type")
        
        # Session records indexes
        await self.db[self.collections["sessions"]].create_index("user_id")
        await self.db[self.collections["sessions"]].create_index("session_id", unique=True)
        await self.db[self.collections["sessions"]].create_index("start_time")
        
        # Psychological states indexes
        await self.db[self.collections["psychology"]].create_index("user_id")
        await self.db[self.collections["psychology"]].create_index("timestamp")
        await self.db[self.collections["psychology"]].create_index([("user_id", 1), ("timestamp", -1)])
        
        # Memory fragments indexes
        await self.db[self.collections["memory"]].create_index("user_id")
        await self.db[self.collections["memory"]].create_index("memory_type")
        await self.db[self.collections["memory"]].create_index("importance_score")
        await self.db[self.collections["memory"]].create_index("last_accessed")
        await self.db[self.collections["memory"]].create_index("tags")
        await self.db[self.collections["memory"]].create_index([("user_id", 1), ("importance_score", -1)])
    
    # User Profile Operations
    async def create_user_profile(self, user_profile: UserProfile) -> bool:
        """Create a new user profile"""
        try:
            result = await self.db[self.collections["users"]].insert_one(user_profile.to_dict())
            return result.inserted_id is not None
        except Exception as e:
            self.logger.error(f"Error creating user profile: {e}")
            return False
    
    async def get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """Get user profile by user_id"""
        try:
            doc = await self.db[self.collections["users"]].find_one({"user_id": user_id})
            if doc:
                return UserProfile(
                    user_id=doc["user_id"],
                    created_at=doc["created_at"],
                    last_active=doc["last_active"],
                    display_name=doc.get("display_name"),
                    preferences=doc.get("preferences", {}),
                    adaptive_profile=doc.get("adaptive_profile", {})
                )
            return None
        except Exception as e:
            self.logger.error(f"Error getting user profile: {e}")
            return None
    
    async def update_user_profile(self, user_id: str, updates: Dict[str, Any]) -> bool:
        """Update user profile"""
        try:
            updates["last_active"] = datetime.now()
            result = await self.db[self.collections["users"]].update_one(
                {"user_id": user_id},
                {"$set": updates}
            )
            return result.modified_count > 0
        except Exception as e:
            self.logger.error(f"Error updating user profile: {e}")
            return False
    
    # Interaction Records Operations
    async def save_interaction(self, interaction: InteractionRecord) -> bool:
        """Save an interaction record"""
        try:
            result = await self.db[self.collections["interactions"]].insert_one(interaction.to_dict())
            return result.inserted_id is not None
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
                interaction = InteractionRecord(
                    interaction_id=doc["interaction_id"],
                    user_id=doc["user_id"],
                    session_id=doc["session_id"],
                    timestamp=doc["timestamp"],
                    user_input=doc["user_input"],
                    companion_response=doc["companion_response"],
                    interaction_type=InteractionType(doc["interaction_type"]),
                    context_analysis=doc["context_analysis"],
                    emotional_state=doc["emotional_state"],
                    technical_context=doc["technical_context"],
                    creative_context=doc["creative_context"],
                    guidance_used=doc["guidance_used"],
                    response_metrics=doc["response_metrics"]
                )
                interactions.append(interaction)
            
            return interactions
        except Exception as e:
            self.logger.error(f"Error getting recent interactions: {e}")
            return []
    
    async def get_session_interactions(self, session_id: str) -> List[InteractionRecord]:
        """Get all interactions for a specific session"""
        try:
            cursor = self.db[self.collections["interactions"]].find(
                {"session_id": session_id}
            ).sort("timestamp", 1)
            
            interactions = []
            async for doc in cursor:
                interaction = InteractionRecord(
                    interaction_id=doc["interaction_id"],
                    user_id=doc["user_id"],
                    session_id=doc["session_id"],
                    timestamp=doc["timestamp"],
                    user_input=doc["user_input"],
                    companion_response=doc["companion_response"],
                    interaction_type=InteractionType(doc["interaction_type"]),
                    context_analysis=doc["context_analysis"],
                    emotional_state=doc["emotional_state"],
                    technical_context=doc["technical_context"],
                    creative_context=doc["creative_context"],
                    guidance_used=doc["guidance_used"],
                    response_metrics=doc["response_metrics"]
                )
                interactions.append(interaction)
            
            return interactions
        except Exception as e:
            self.logger.error(f"Error getting session interactions: {e}")
            return []
    
    # Session Records Operations
    async def create_session(self, session: SessionRecord) -> bool:
        """Create a new session record"""
        try:
            result = await self.db[self.collections["sessions"]].insert_one(session.to_dict())
            return result.inserted_id is not None
        except Exception as e:
            self.logger.error(f"Error creating session: {e}")
            return False
    
    async def update_session(self, session_id: str, updates: Dict[str, Any]) -> bool:
        """Update session record"""
        try:
            result = await self.db[self.collections["sessions"]].update_one(
                {"session_id": session_id},
                {"$set": updates}
            )
            return result.modified_count > 0
        except Exception as e:
            self.logger.error(f"Error updating session: {e}")
            return False
    
    async def get_session(self, session_id: str) -> Optional[SessionRecord]:
        """Get session record by session_id"""
        try:
            doc = await self.db[self.collections["sessions"]].find_one({"session_id": session_id})
            if doc:
                return SessionRecord(
                    session_id=doc["session_id"],
                    user_id=doc["user_id"],
                    start_time=doc["start_time"],
                    end_time=doc.get("end_time"),
                    total_interactions=doc["total_interactions"],
                    primary_focuses=doc["primary_focuses"],
                    emotional_trajectory=doc["emotional_trajectory"],
                    session_summary=doc["session_summary"]
                )
            return None
        except Exception as e:
            self.logger.error(f"Error getting session: {e}")
            return None
    
    # Psychological State Operations
    async def save_psychological_state(self, state: PsychologicalState) -> bool:
        """Save psychological state snapshot"""
        try:
            result = await self.db[self.collections["psychology"]].insert_one(state.to_dict())
            return result.inserted_id is not None
        except Exception as e:
            self.logger.error(f"Error saving psychological state: {e}")
            return False
    
    async def get_latest_psychological_state(self, user_id: str) -> Optional[PsychologicalState]:
        """Get latest psychological state for user"""
        try:
            doc = await self.db[self.collections["psychology"]].find_one(
                {"user_id": user_id},
                sort=[("timestamp", -1)]
            )
            if doc:
                return PsychologicalState(
                    user_id=doc["user_id"],
                    timestamp=doc["timestamp"],
                    emotional_indicators=doc["emotional_indicators"],
                    stress_levels=doc["stress_levels"],
                    attachment_patterns=doc["attachment_patterns"],
                    creative_state=doc["creative_state"],
                    technical_confidence=doc["technical_confidence"],
                    support_needs=doc["support_needs"],
                    risk_factors=doc["risk_factors"],
                    growth_indicators=doc["growth_indicators"]
                )
            return None
        except Exception as e:
            self.logger.error(f"Error getting psychological state: {e}")
            return None
    
    async def get_psychological_trend(self, user_id: str, days: int = 30) -> List[PsychologicalState]:
        """Get psychological state trend over time"""
        try:
            start_date = datetime.now() - timedelta(days=days)
            cursor = self.db[self.collections["psychology"]].find(
                {"user_id": user_id, "timestamp": {"$gte": start_date}}
            ).sort("timestamp", 1)
            
            states = []
            async for doc in cursor:
                state = PsychologicalState(
                    user_id=doc["user_id"],
                    timestamp=doc["timestamp"],
                    emotional_indicators=doc["emotional_indicators"],
                    stress_levels=doc["stress_levels"],
                    attachment_patterns=doc["attachment_patterns"],
                    creative_state=doc["creative_state"],
                    technical_confidence=doc["technical_confidence"],
                    support_needs=doc["support_needs"],
                    risk_factors=doc["risk_factors"],
                    growth_indicators=doc["growth_indicators"]
                )
                states.append(state)
            
            return states
        except Exception as e:
            self.logger.error(f"Error getting psychological trend: {e}")
            return []
    
    # Memory Fragment Operations
    async def save_memory_fragment(self, memory: MemoryFragment) -> bool:
        """Save a memory fragment"""
        try:
            result = await self.db[self.collections["memory"]].insert_one(memory.to_dict())
            return result.inserted_id is not None
        except Exception as e:
            self.logger.error(f"Error saving memory fragment: {e}")
            return False
    
    async def get_relevant_memories(self, user_id: str, memory_type: str = None, 
                                  tags: List[str] = None, limit: int = 10) -> List[MemoryFragment]:
        """Get relevant memory fragments for context"""
        try:
            query = {"user_id": user_id}
            
            if memory_type:
                query["memory_type"] = memory_type
            
            if tags:
                query["tags"] = {"$in": tags}
            
            cursor = self.db[self.collections["memory"]].find(query).sort(
                "importance_score", -1
            ).limit(limit)
            
            memories = []
            async for doc in cursor:
                memory = MemoryFragment(
                    memory_id=doc["memory_id"],
                    user_id=doc["user_id"],
                    content=doc["content"],
                    memory_type=doc["memory_type"],
                    importance_score=doc["importance_score"],
                    created_at=doc["created_at"],
                    last_accessed=doc["last_accessed"],
                    access_count=doc["access_count"],
                    related_interactions=doc["related_interactions"],
                    tags=doc["tags"]
                )
                memories.append(memory)
            
            return memories
        except Exception as e:
            self.logger.error(f"Error getting relevant memories: {e}")
            return []
    
    async def update_memory_access(self, memory_id: str) -> bool:
        """Update memory access tracking"""
        try:
            result = await self.db[self.collections["memory"]].update_one(
                {"memory_id": memory_id},
                {
                    "$set": {"last_accessed": datetime.now()},
                    "$inc": {"access_count": 1}
                }
            )
            return result.modified_count > 0
        except Exception as e:
            self.logger.error(f"Error updating memory access: {e}")
            return False
    
    # Analytics Operations
    async def get_user_analytics(self, user_id: str, days: int = 30) -> Dict[str, Any]:
        """Get user interaction analytics"""
        try:
            start_date = datetime.now() - timedelta(days=days)
            
            # Interaction count by type
            interaction_pipeline = [
                {"$match": {"user_id": user_id, "timestamp": {"$gte": start_date}}},
                {"$group": {"_id": "$interaction_type", "count": {"$sum": 1}}}
            ]
            
            interaction_stats = {}
            async for doc in self.db[self.collections["interactions"]].aggregate(interaction_pipeline):
                interaction_stats[doc["_id"]] = doc["count"]
            
            # Emotional state trends
            emotional_pipeline = [
                {"$match": {"user_id": user_id, "timestamp": {"$gte": start_date}}},
                {"$sort": {"timestamp": 1}},
                {"$project": {"timestamp": 1, "emotional_indicators": 1}}
            ]
            
            emotional_trend = []
            async for doc in self.db[self.collections["psychology"]].aggregate(emotional_pipeline):
                emotional_trend.append({
                    "timestamp": doc["timestamp"],
                    "emotional_indicators": doc["emotional_indicators"]
                })
            
            return {
                "user_id": user_id,
                "analysis_period_days": days,
                "interaction_statistics": interaction_stats,
                "emotional_trend": emotional_trend,
                "total_interactions": sum(interaction_stats.values()),
                "generated_at": datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting user analytics: {e}")
            return {}
    
    async def close(self):
        """Close database connection"""
        if self.client:
            self.client.close()

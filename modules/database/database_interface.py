"""
Unified Companion Database Interface

Database interface definitions for the unified companion system.
This provides the schema and interface without requiring specific database dependencies.
"""

from typing import Dict, List, Any, Optional, Protocol
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import logging
import json
import uuid

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
    preferences: Dict[str, Any] = field(default_factory=dict)
    adaptive_profile: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "created_at": self.created_at.isoformat(),
            "last_active": self.last_active.isoformat(),
            "display_name": self.display_name,
            "preferences": self.preferences,
            "adaptive_profile": self.adaptive_profile
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UserProfile':
        return cls(
            user_id=data["user_id"],
            created_at=datetime.fromisoformat(data["created_at"]),
            last_active=datetime.fromisoformat(data["last_active"]),
            display_name=data.get("display_name"),
            preferences=data.get("preferences", {}),
            adaptive_profile=data.get("adaptive_profile", {})
        )

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
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "interaction_id": self.interaction_id,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "timestamp": self.timestamp.isoformat(),
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
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'InteractionRecord':
        return cls(
            interaction_id=data["interaction_id"],
            user_id=data["user_id"],
            session_id=data["session_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            user_input=data["user_input"],
            companion_response=data["companion_response"],
            interaction_type=InteractionType(data["interaction_type"]),
            context_analysis=data["context_analysis"],
            emotional_state=data["emotional_state"],
            technical_context=data["technical_context"],
            creative_context=data["creative_context"],
            guidance_used=data["guidance_used"],
            response_metrics=data["response_metrics"]
        )

@dataclass
class SessionRecord:
    """Session-level interaction data"""
    session_id: str
    user_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    total_interactions: int = 0
    primary_focuses: List[str] = field(default_factory=list)
    emotional_trajectory: List[Dict[str, Any]] = field(default_factory=list)
    session_summary: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "total_interactions": self.total_interactions,
            "primary_focuses": self.primary_focuses,
            "emotional_trajectory": self.emotional_trajectory,
            "session_summary": self.session_summary
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SessionRecord':
        return cls(
            session_id=data["session_id"],
            user_id=data["user_id"],
            start_time=datetime.fromisoformat(data["start_time"]),
            end_time=datetime.fromisoformat(data["end_time"]) if data.get("end_time") else None,
            total_interactions=data["total_interactions"],
            primary_focuses=data["primary_focuses"],
            emotional_trajectory=data["emotional_trajectory"],
            session_summary=data["session_summary"]
        )

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
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "timestamp": self.timestamp.isoformat(),
            "emotional_indicators": self.emotional_indicators,
            "stress_levels": self.stress_levels,
            "attachment_patterns": self.attachment_patterns,
            "creative_state": self.creative_state,
            "technical_confidence": self.technical_confidence,
            "support_needs": self.support_needs,
            "risk_factors": self.risk_factors,
            "growth_indicators": self.growth_indicators
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PsychologicalState':
        return cls(
            user_id=data["user_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            emotional_indicators=data["emotional_indicators"],
            stress_levels=data["stress_levels"],
            attachment_patterns=data["attachment_patterns"],
            creative_state=data["creative_state"],
            technical_confidence=data["technical_confidence"],
            support_needs=data["support_needs"],
            risk_factors=data["risk_factors"],
            growth_indicators=data["growth_indicators"]
        )

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
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "user_id": self.user_id,
            "content": self.content,
            "memory_type": self.memory_type,
            "importance_score": self.importance_score,
            "created_at": self.created_at.isoformat(),
            "last_accessed": self.last_accessed.isoformat(),
            "access_count": self.access_count,
            "related_interactions": self.related_interactions,
            "tags": self.tags,
            "tone": self.tone,
            "sentiment": self.sentiment,
            "time_of_day": self.time_of_day
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryFragment':
        return cls(
            memory_id=data["memory_id"],
            user_id=data["user_id"],
            content=data["content"],
            memory_type=data["memory_type"],
            importance_score=data["importance_score"],
            created_at=datetime.fromisoformat(data["created_at"]),
            last_accessed=datetime.fromisoformat(data["last_accessed"]),
            access_count=data["access_count"],
            related_interactions=data["related_interactions"],
            tags=data["tags"],
            tone=data.get("tone"),
            sentiment=data.get("sentiment", 0.0),
            time_of_day=data.get("time_of_day")
        )


@dataclass
class EmotionalRiskEntry:
    """Log of emotionally vulnerable interactions"""
    entry_id: str
    user_id: str
    timestamp: datetime
    user_input: str
    risk_tags: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "user_id": self.user_id,
            "timestamp": self.timestamp.isoformat(),
            "user_input": self.user_input,
            "risk_tags": self.risk_tags,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EmotionalRiskEntry':
        return cls(
            entry_id=data["entry_id"],
            user_id=data["user_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            user_input=data.get("user_input", ""),
            risk_tags=data.get("risk_tags", []),
        )

class DatabaseInterface(Protocol):
    """Protocol defining the database interface for the unified companion system"""
    
    async def initialize(self) -> None:
        """Initialize the database connection"""
        ...
    
    async def create_user_profile(self, user_profile: UserProfile) -> bool:
        """Create a new user profile"""
        ...
    
    async def get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """Get user profile by user_id"""
        ...
    
    async def update_user_profile(self, user_id: str, updates: Dict[str, Any]) -> bool:
        """Update user profile"""
        ...
    
    async def save_interaction(self, interaction: InteractionRecord) -> bool:
        """Save an interaction record"""
        ...
    
    async def get_recent_interactions(self, user_id: str, limit: int = 20) -> List[InteractionRecord]:
        """Get recent interactions for a user"""
        ...
    
    async def save_psychological_state(self, state: PsychologicalState) -> bool:
        """Save psychological state snapshot"""
        ...
    
    async def get_latest_psychological_state(self, user_id: str) -> Optional[PsychologicalState]:
        """Get latest psychological state for user"""
        ...
    
    async def save_memory_fragment(self, memory: MemoryFragment) -> bool:
        """Save a memory fragment"""
        ...
    
    async def get_relevant_memories(self, user_id: str, memory_type: Optional[str] = None,
                                  tags: Optional[List[str]] = None, limit: int = 10) -> List[MemoryFragment]:
        """Get relevant memory fragments for context"""
        ...

    async def log_emotional_risk(self, entry: EmotionalRiskEntry) -> bool:
        """Log an emotionally vulnerable interaction"""
        ...

    async def get_emotional_risk_history(self, user_id: str, limit: int = 20) -> List[EmotionalRiskEntry]:
        """Retrieve emotional risk history"""
        ...

class InMemoryDatabase:
    """
    In-memory database implementation for development and testing
    """
    
    def __init__(self):
        self.users: Dict[str, UserProfile] = {}
        self.interactions: List[InteractionRecord] = []
        self.sessions: Dict[str, SessionRecord] = {}
        self.psychological_states: List[PsychologicalState] = []
        self.memory_fragments: List[MemoryFragment] = []
        self.emotional_risk_registry: List[EmotionalRiskEntry] = []
        self.logger = logging.getLogger(__name__)
    
    async def initialize(self) -> None:
        """Initialize the in-memory database"""
        self.logger.info("In-memory database initialized")
    
    async def create_user_profile(self, user_profile: UserProfile) -> bool:
        """Create a new user profile"""
        try:
            if user_profile.user_id in self.users:
                return False
            self.users[user_profile.user_id] = user_profile
            return True
        except Exception as e:
            self.logger.error(f"Error creating user profile: {e}")
            return False
    
    async def get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """Get user profile by user_id"""
        return self.users.get(user_id)
    
    async def update_user_profile(self, user_id: str, updates: Dict[str, Any]) -> bool:
        """Update user profile"""
        try:
            if user_id not in self.users:
                return False
            
            user = self.users[user_id]
            for key, value in updates.items():
                if hasattr(user, key):
                    setattr(user, key, value)
            
            user.last_active = datetime.now()
            return True
        except Exception as e:
            self.logger.error(f"Error updating user profile: {e}")
            return False
    
    async def save_interaction(self, interaction: InteractionRecord) -> bool:
        """Save an interaction record"""
        try:
            self.interactions.append(interaction)
            return True
        except Exception as e:
            self.logger.error(f"Error saving interaction: {e}")
            return False
    
    async def get_recent_interactions(self, user_id: str, limit: int = 20) -> List[InteractionRecord]:
        """Get recent interactions for a user"""
        try:
            user_interactions = [i for i in self.interactions if i.user_id == user_id]
            user_interactions.sort(key=lambda x: x.timestamp, reverse=True)
            return user_interactions[:limit]
        except Exception as e:
            self.logger.error(f"Error getting recent interactions: {e}")
            return []
    
    async def get_session_interactions(self, session_id: str) -> List[InteractionRecord]:
        """Get all interactions for a specific session"""
        try:
            session_interactions = [i for i in self.interactions if i.session_id == session_id]
            session_interactions.sort(key=lambda x: x.timestamp)
            return session_interactions
        except Exception as e:
            self.logger.error(f"Error getting session interactions: {e}")
            return []
    
    async def create_session(self, session: SessionRecord) -> bool:
        """Create a new session record"""
        try:
            self.sessions[session.session_id] = session
            return True
        except Exception as e:
            self.logger.error(f"Error creating session: {e}")
            return False
    
    async def update_session(self, session_id: str, updates: Dict[str, Any]) -> bool:
        """Update session record"""
        try:
            if session_id not in self.sessions:
                return False
            
            session = self.sessions[session_id]
            for key, value in updates.items():
                if hasattr(session, key):
                    setattr(session, key, value)
            
            return True
        except Exception as e:
            self.logger.error(f"Error updating session: {e}")
            return False
    
    async def get_session(self, session_id: str) -> Optional[SessionRecord]:
        """Get session record by session_id"""
        return self.sessions.get(session_id)
    
    async def save_psychological_state(self, state: PsychologicalState) -> bool:
        """Save psychological state snapshot"""
        try:
            self.psychological_states.append(state)
            return True
        except Exception as e:
            self.logger.error(f"Error saving psychological state: {e}")
            return False
    
    async def get_latest_psychological_state(self, user_id: str) -> Optional[PsychologicalState]:
        """Get latest psychological state for user"""
        try:
            user_states = [s for s in self.psychological_states if s.user_id == user_id]
            if user_states:
                return max(user_states, key=lambda x: x.timestamp)
            return None
        except Exception as e:
            self.logger.error(f"Error getting psychological state: {e}")
            return None
    
    async def get_psychological_trend(self, user_id: str, days: int = 30) -> List[PsychologicalState]:
        """Get psychological state trend over time"""
        try:
            start_date = datetime.now() - timedelta(days=days)
            user_states = [s for s in self.psychological_states 
                          if s.user_id == user_id and s.timestamp >= start_date]
            user_states.sort(key=lambda x: x.timestamp)
            return user_states
        except Exception as e:
            self.logger.error(f"Error getting psychological trend: {e}")
            return []
    
    async def save_memory_fragment(self, memory: MemoryFragment) -> bool:
        """Save a memory fragment"""
        try:
            self.memory_fragments.append(memory)
            return True
        except Exception as e:
            self.logger.error(f"Error saving memory fragment: {e}")
            return False
    
    async def get_relevant_memories(self, user_id: str, memory_type: Optional[str] = None, 
                                  tags: Optional[List[str]] = None, limit: int = 10) -> List[MemoryFragment]:
        """Get relevant memory fragments for context"""
        try:
            user_memories = [m for m in self.memory_fragments if m.user_id == user_id]
            
            if memory_type:
                user_memories = [m for m in user_memories if m.memory_type == memory_type]
            
            if tags:
                user_memories = [m for m in user_memories if any(tag in m.tags for tag in tags)]
            
            # Sort by importance score
            user_memories.sort(key=lambda x: x.importance_score, reverse=True)
            return user_memories[:limit]
        except Exception as e:
            self.logger.error(f"Error getting relevant memories: {e}")
            return []
    
    async def update_memory_access(self, memory_id: str) -> bool:
        """Update memory access tracking"""
        try:
            for memory in self.memory_fragments:
                if memory.memory_id == memory_id:
                    memory.last_accessed = datetime.now()
                    memory.access_count += 1
                    return True
            return False
        except Exception as e:
            self.logger.error(f"Error updating memory access: {e}")
            return False

    async def log_emotional_risk(self, entry: EmotionalRiskEntry) -> bool:
        """Log an emotional risk entry"""
        try:
            self.emotional_risk_registry.append(entry)
            return True
        except Exception as e:
            self.logger.error(f"Error logging emotional risk: {e}")
            return False

    async def get_emotional_risk_history(self, user_id: str, limit: int = 20) -> List[EmotionalRiskEntry]:
        """Retrieve emotional risk history"""
        try:
            entries = [e for e in self.emotional_risk_registry if e.user_id == user_id]
            entries.sort(key=lambda x: x.timestamp, reverse=True)
            return entries[:limit]
        except Exception as e:
            self.logger.error(f"Error getting emotional risk history: {e}")
            return []
    
    async def get_user_analytics(self, user_id: str, days: int = 30) -> Dict[str, Any]:
        """Get user interaction analytics"""
        try:
            start_date = datetime.now() - timedelta(days=days)
            
            # Get interactions in timeframe
            user_interactions = [i for i in self.interactions 
                               if i.user_id == user_id and i.timestamp >= start_date]
            
            # Count by interaction type
            interaction_stats = {}
            for interaction in user_interactions:
                interaction_type = interaction.interaction_type.value
                interaction_stats[interaction_type] = interaction_stats.get(interaction_type, 0) + 1
            
            # Get emotional trend
            user_states = [s for s in self.psychological_states 
                          if s.user_id == user_id and s.timestamp >= start_date]
            user_states.sort(key=lambda x: x.timestamp)
            
            emotional_trend = [
                {
                    "timestamp": state.timestamp.isoformat(),
                    "emotional_indicators": state.emotional_indicators
                }
                for state in user_states
            ]
            
            return {
                "user_id": user_id,
                "analysis_period_days": days,
                "interaction_statistics": interaction_stats,
                "emotional_trend": emotional_trend,
                "total_interactions": len(user_interactions),
                "generated_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting user analytics: {e}")
            return {}
    
    async def close(self):
        """Close database connection (no-op for in-memory)"""
        self.logger.info("In-memory database closed")

def create_database_interface(connection_string: Optional[str] = None, database_type: str = "auto") -> DatabaseInterface:
    """
    Factory function to create appropriate database interface
    Auto-detects MongoDB when connection string is provided
    Supports local JSON persistence when `database_type` is ``jsonfile``.
    """
    import os
    
    # Auto-detect database type based on environment and parameters
    if database_type == "auto":
        if connection_string or os.getenv('MONGO_CONNECTION_STRING'):
            database_type = "mongodb"
            logging.info("Auto-detected MongoDB from connection string or environment")
        else:
            database_type = "inmemory"
            logging.info("No connection string found, defaulting to in-memory database")
    
    # Use environment variable if no connection string provided
    if not connection_string:
        connection_string = os.getenv('MONGO_CONNECTION_STRING')
    
    if database_type == "inmemory":
        logging.info("Using in-memory database")
        return InMemoryDatabase()
    elif database_type == "jsonfile":
        json_path = connection_string or os.getenv('JSON_DB_PATH', 'companion_db.json')
        logging.info(f"Using JSON database file: {json_path}")
        from .json_database import JSONDatabase
        db = JSONDatabase(json_path)
        awaitable = getattr(db, 'initialize', None)
        if awaitable:
            try:
                import asyncio
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(db.initialize())
                else:
                    loop.run_until_complete(db.initialize())
            except RuntimeError:
                asyncio.run(db.initialize())
        return db
    elif database_type == "mongodb":
        if not connection_string:
            raise ValueError("MongoDB connection string required but not provided")
            
        try:
            from .mongodb_database import MongoDatabase
            logging.info(f"Initializing MongoDB with connection: {connection_string[:20] if len(connection_string) > 20 else connection_string}...")
            db = MongoDatabase(connection_string)
            # Test basic functionality to ensure MongoDB is working
            logging.info("MongoDB interface created successfully")
            return db
        except ImportError as e:
            error_msg = f"MongoDB dependencies not available: {e}. Install with 'pip install motor pymongo'"
            logging.error(error_msg)
            raise ImportError(error_msg)
        except Exception as e:
            error_msg = f"Failed to create MongoDB interface: {e}"
            logging.error(error_msg)
            raise RuntimeError(error_msg)
    else:
        raise ValueError(
            f"Unsupported database type: {database_type}. Supported types: 'auto', 'inmemory', 'mongodb', 'jsonfile'"
        )

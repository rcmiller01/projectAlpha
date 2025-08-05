#!/usr/bin/env python3
"""
Memory System for Dolphin AI Orchestrator

Handles session memory, long-term storage, sentiment analysis,
and context management for enhanced AI interactions.
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import hashlib
import os
import re
from collections import defaultdict

logger = logging.getLogger(__name__)

class MemorySystem:
    """
    Manages short-term and long-term memory for AI interactions
    """
    
    def __init__(self, memory_dir: str = "memory"):
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(exist_ok=True)
        
        self.short_term_file = self.memory_dir / "short_term_memory.json"
        self.long_term_file = self.memory_dir / "long_term_memory.json"
        self.sessions_dir = self.memory_dir / "sessions"
        self.sessions_dir.mkdir(exist_ok=True)
        
        # Memory stores
        self.short_term_memory = self._load_short_term()
        self.long_term_memory = self._load_long_term()
        self.current_session_id = None

        allowed = os.getenv("ALLOWED_PERSONAS", "")
        self.authorized_personas = [p.strip().lower() for p in allowed.split(',') if p.strip()]
        self.persona_token = os.getenv("PERSONA_TOKEN", "")
        
        # Sentiment keywords for basic analysis
        self.positive_keywords = {
            'happy', 'joy', 'excited', 'love', 'amazing', 'wonderful', 'great',
            'fantastic', 'awesome', 'brilliant', 'perfect', 'excellent', 'good',
            'pleased', 'satisfied', 'grateful', 'thankful', 'delighted'
        }
        
        self.negative_keywords = {
            'sad', 'angry', 'frustrated', 'annoyed', 'upset', 'disappointed',
            'worried', 'anxious', 'stressed', 'terrible', 'awful', 'bad',
            'horrible', 'hate', 'disgusted', 'depressed', 'lonely', 'scared'
        }

        logger.info("🧠 Memory System initialized")

    def _is_authorized(self, persona: Optional[str], token: Optional[str]) -> bool:
        """Check if a persona is authorized to access memory"""
        if token and token == self.persona_token:
            return True
        if persona:
            return persona.lower() in self.authorized_personas
        return False
    
    def _load_short_term(self) -> Dict[str, Any]:
        """Load short-term memory from file"""
        if self.short_term_file.exists():
            try:
                with open(self.short_term_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading short-term memory: {e}")
        
        return {
            "active_sessions": {},
            "recent_interactions": [],
            "current_context": {},
            "last_updated": datetime.now().isoformat()
        }
    
    def _load_long_term(self) -> Dict[str, Any]:
        """Load long-term memory from file"""
        if self.long_term_file.exists():
            try:
                with open(self.long_term_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading long-term memory: {e}")
        
        return {
            "user_preferences": {},
            "personality_traits": {},
            "goals": [],
            "achievements": [],
            "relationships": {},
            "important_events": [],
            "emotional_patterns": {},
            "learned_behaviors": {},
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat()
        }
    
    def _save_short_term(self):
        """Save short-term memory to file"""
        try:
            self.short_term_memory["last_updated"] = datetime.now().isoformat()
            with open(self.short_term_file, 'w') as f:
                json.dump(self.short_term_memory, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving short-term memory: {e}")
    
    def _save_long_term(self):
        """Save long-term memory to file"""
        try:
            self.long_term_memory["last_updated"] = datetime.now().isoformat()
            with open(self.long_term_file, 'w') as f:
                json.dump(self.long_term_memory, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving long-term memory: {e}")
    
    def create_session(self, session_id: Optional[str] = None) -> str:
        """Create a new session and return session ID"""
        if session_id is None:
            session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hashlib.md5(str(datetime.now()).encode()).hexdigest()[:8]}"
        
        self.current_session_id = session_id
        
        session_data = {
            "session_id": session_id,
            "created_at": datetime.now().isoformat(),
            "messages": [],
            "context": {},
            "sentiment_history": [],
            "judgments": [],
            "persona_used": "companion",
            "status": "active"
        }
        
        # Add to short-term memory
        self.short_term_memory["active_sessions"][session_id] = session_data
        self._save_short_term()
        
        logger.info(f"New session created: {session_id}")
        return session_id
    
    def add_message(self, session_id: str, message: str, role: str,
                   handler: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None,
                   persona: Optional[str] = None, persona_token: Optional[str] = None) -> Dict[str, Any]:
        """Add a message to session memory with sentiment analysis"""

        if not self._is_authorized(persona, persona_token):
            logger.warning("Unauthorized memory write attempt")
            return {}
        
        # Ensure session exists
        if session_id not in self.short_term_memory["active_sessions"]:
            self.create_session(session_id)
        
        # Analyze sentiment
        sentiment_score, emotion_tags = self._analyze_sentiment(message)
        
        # Create message entry
        message_entry = {
            "timestamp": datetime.now().isoformat(),
            "role": role,  # "user" or "assistant"
            "content": message,
            "handler": handler,
            "sentiment_score": sentiment_score,
            "emotion_tags": emotion_tags,
            "metadata": metadata or {}
        }
        
        # Add to session
        session = self.short_term_memory["active_sessions"][session_id]
        session["messages"].append(message_entry)
        session["sentiment_history"].append({
            "timestamp": message_entry["timestamp"],
            "score": sentiment_score,
            "tags": emotion_tags
        })
        
        # Update recent interactions
        self.short_term_memory["recent_interactions"].append({
            "session_id": session_id,
            "timestamp": message_entry["timestamp"],
            "role": role,
            "preview": message[:100] + "..." if len(message) > 100 else message,
            "sentiment": sentiment_score
        })
        
        # Keep only last 50 recent interactions
        self.short_term_memory["recent_interactions"] = \
            self.short_term_memory["recent_interactions"][-50:]
        
        self._save_short_term()
        
        # Update long-term patterns if significant
        if abs(sentiment_score) > 0.6:  # Strong emotional content
            self._update_emotional_patterns(emotion_tags, sentiment_score)
        
        return message_entry
    
    def _analyze_sentiment(self, text: str) -> Tuple[float, List[str]]:
        """Basic sentiment analysis using keyword matching"""
        text_lower = text.lower()
        words = re.findall(r'\b\w+\b', text_lower)
        
        positive_count = sum(1 for word in words if word in self.positive_keywords)
        negative_count = sum(1 for word in words if word in self.negative_keywords)
        
        total_emotional_words = positive_count + negative_count
        
        if total_emotional_words == 0:
            sentiment_score = 0.0
        else:
            sentiment_score = (positive_count - negative_count) / len(words)
            # Normalize to -1 to 1 range
            sentiment_score = max(-1.0, min(1.0, sentiment_score * 10))
        
        # Extract emotion tags
        emotion_tags = []
        for word in words:
            if word in self.positive_keywords:
                emotion_tags.append(f"positive:{word}")
            elif word in self.negative_keywords:
                emotion_tags.append(f"negative:{word}")
        
        return sentiment_score, emotion_tags
    
    def _update_emotional_patterns(self, emotion_tags: List[str], sentiment_score: float):
        """Update long-term emotional patterns"""
        if "emotional_patterns" not in self.long_term_memory:
            self.long_term_memory["emotional_patterns"] = {}
        
        patterns = self.long_term_memory["emotional_patterns"]
        
        # Update emotion frequency
        for tag in emotion_tags:
            if tag not in patterns:
                patterns[tag] = {"count": 0, "recent_scores": []}
            patterns[tag]["count"] += 1
            patterns[tag]["recent_scores"].append({
                "score": sentiment_score,
                "timestamp": datetime.now().isoformat()
            })
            # Keep only last 20 scores
            patterns[tag]["recent_scores"] = patterns[tag]["recent_scores"][-20:]
        
        self._save_long_term()

    def add_judgment(self, session_id: str, judgment: Dict[str, Any],
                     persona: Optional[str] = None, persona_token: Optional[str] = None) -> None:
        """Store a judgment entry for later introspection."""

        if not self._is_authorized(persona, persona_token):
            logger.warning("Unauthorized memory write attempt")
            return

        if session_id not in self.short_term_memory["active_sessions"]:
            self.create_session(session_id)

        session = self.short_term_memory["active_sessions"][session_id]
        if "judgments" not in session:
            session["judgments"] = []

        session["judgments"].append({
            "timestamp": datetime.now().isoformat(),
            **judgment
        })

        # Keep only last 20 judgments
        session["judgments"] = session["judgments"][-20:]
        self._save_short_term()

    def get_session_context(self, session_id: str, last_n_messages: int = 10,
                            persona: Optional[str] = None, persona_token: Optional[str] = None) -> Dict[str, Any]:
        """Get recent context for a session"""

        if not self._is_authorized(persona, persona_token):
            logger.warning("Unauthorized memory read attempt")
            return {"messages": [], "context": {}, "sentiment_trend": 0.0}
        if session_id not in self.short_term_memory["active_sessions"]:
            return {"messages": [], "context": {}, "sentiment_trend": 0.0}
        
        session = self.short_term_memory["active_sessions"][session_id]
        recent_messages = session["messages"][-last_n_messages:]
        
        # Calculate sentiment trend
        recent_sentiments = [msg["sentiment_score"] for msg in recent_messages if "sentiment_score" in msg]
        sentiment_trend = sum(recent_sentiments) / len(recent_sentiments) if recent_sentiments else 0.0
        
        return {
            "messages": recent_messages,
            "context": session.get("context", {}),
            "sentiment_trend": sentiment_trend,
            "total_messages": len(session["messages"]),
            "session_duration": self._calculate_session_duration(session)
        }
    
    def _calculate_session_duration(self, session: Dict[str, Any]) -> str:
        """Calculate how long a session has been active"""
        try:
            created_at = datetime.fromisoformat(session["created_at"])
            duration = datetime.now() - created_at
            
            if duration.days > 0:
                return f"{duration.days} days"
            elif duration.seconds > 3600:
                hours = duration.seconds // 3600
                return f"{hours} hours"
            else:
                minutes = duration.seconds // 60
                return f"{minutes} minutes"
        except:
            return "unknown"
    
    def update_long_term_trait(self, trait_type: str, trait_name: str, value: Any):
        """Update a long-term trait or preference"""
        if trait_type not in self.long_term_memory:
            self.long_term_memory[trait_type] = {}
        
        self.long_term_memory[trait_type][trait_name] = {
            "value": value,
            "updated_at": datetime.now().isoformat()
        }
        
        self._save_long_term()
        logger.info(f"Updated long-term trait: {trait_type}.{trait_name}")
    
    def add_goal(self, goal_text: str, priority: str = "medium") -> str:
        """Add a user goal to long-term memory"""
        goal_id = f"goal_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        goal = {
            "id": goal_id,
            "text": goal_text,
            "priority": priority,
            "status": "active",
            "created_at": datetime.now().isoformat(),
            "progress_notes": []
        }
        
        self.long_term_memory["goals"].append(goal)
        self._save_long_term()
        
        logger.info(f"New goal added: {goal_text}")
        return goal_id
    
    def update_goal_progress(self, goal_id: str, progress_note: str):
        """Update progress on a goal"""
        for goal in self.long_term_memory["goals"]:
            if goal["id"] == goal_id:
                goal["progress_notes"].append({
                    "note": progress_note,
                    "timestamp": datetime.now().isoformat()
                })
                self._save_long_term()
                return True
        return False
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """Get a comprehensive memory summary"""
        active_sessions = len(self.short_term_memory["active_sessions"])
        total_messages = sum(len(session["messages"]) for session in self.short_term_memory["active_sessions"].values())
        
        # Recent sentiment trend
        recent_sentiments = []
        for interaction in self.short_term_memory["recent_interactions"][-10:]:
            if "sentiment" in interaction:
                recent_sentiments.append(interaction["sentiment"])
        
        avg_sentiment = sum(recent_sentiments) / len(recent_sentiments) if recent_sentiments else 0.0
        
        return {
            "short_term": {
                "active_sessions": active_sessions,
                "total_messages": total_messages,
                "recent_sentiment_avg": round(avg_sentiment, 3),
                "last_updated": self.short_term_memory.get("last_updated")
            },
            "long_term": {
                "goals_count": len(self.long_term_memory.get("goals", [])),
                "traits_count": len(self.long_term_memory.get("personality_traits", {})),
                "emotional_patterns_count": len(self.long_term_memory.get("emotional_patterns", {})),
                "achievements_count": len(self.long_term_memory.get("achievements", [])),
                "created_at": self.long_term_memory.get("created_at"),
                "last_updated": self.long_term_memory.get("last_updated")
            }
        }
    
    def flush_short_term(self) -> bool:
        """Clear all short-term memory"""
        try:
            # Archive current sessions before clearing
            archive_data = {
                "archived_at": datetime.now().isoformat(),
                "sessions": self.short_term_memory["active_sessions"]
            }
            
            archive_file = self.memory_dir / f"archive_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(archive_file, 'w') as f:
                json.dump(archive_data, f, indent=2)
            
            # Reset short-term memory
            self.short_term_memory = {
                "active_sessions": {},
                "recent_interactions": [],
                "current_context": {},
                "last_updated": datetime.now().isoformat()
            }
            
            self._save_short_term()
            logger.info("Short-term memory flushed and archived")
            return True
            
        except Exception as e:
            logger.error(f"Error flushing short-term memory: {e}")
            return False
    
    def close_session(self, session_id: str):
        """Close and archive a session"""
        if session_id in self.short_term_memory["active_sessions"]:
            session = self.short_term_memory["active_sessions"][session_id]
            session["status"] = "closed"
            session["closed_at"] = datetime.now().isoformat()
            
            # Save to individual session file
            session_file = self.sessions_dir / f"{session_id}.json"
            with open(session_file, 'w') as f:
                json.dump(session, f, indent=2)
            
            # Remove from active sessions
            del self.short_term_memory["active_sessions"][session_id]
            self._save_short_term()
            
            logger.info(f"Session closed and archived: {session_id}")
    
    def search_memories(self, query: str, memory_type: str = "both") -> List[Dict[str, Any]]:
        """Search through memories for relevant content"""
        results = []
        query_lower = query.lower()
        
        if memory_type in ["both", "short_term"]:
            # Search recent interactions
            for interaction in self.short_term_memory["recent_interactions"]:
                if query_lower in interaction["preview"].lower():
                    results.append({
                        "type": "short_term",
                        "source": "recent_interaction",
                        "content": interaction,
                        "relevance": "high" if query_lower in interaction["preview"][:50].lower() else "medium"
                    })
        
        if memory_type in ["both", "long_term"]:
            # Search goals
            for goal in self.long_term_memory.get("goals", []):
                if query_lower in goal["text"].lower():
                    results.append({
                        "type": "long_term",
                        "source": "goal",
                        "content": goal,
                        "relevance": "high"
                    })
            
            # Search achievements
            for achievement in self.long_term_memory.get("achievements", []):
                if query_lower in str(achievement).lower():
                    results.append({
                        "type": "long_term",
                        "source": "achievement",
                        "content": achievement,
                        "relevance": "medium"
                    })
        
        return results[:20]  # Limit results

    def get_last_memory_session(self) -> Optional[Dict[str, Any]]:
        """Return context for the most recently active session."""
        sessions = self.short_term_memory.get("active_sessions", {})
        latest_id = None
        latest_time = None

        for sid, sess in sessions.items():
            ts = sess.get("messages", [])
            if ts:
                timestamp = ts[-1].get("timestamp", sess.get("created_at"))
            else:
                timestamp = sess.get("created_at")
            try:
                dt = datetime.fromisoformat(timestamp)
            except Exception:
                continue
            if not latest_time or dt > latest_time:
                latest_time = dt
                latest_id = sid

        if latest_id:
            info = self.get_session_context(latest_id)
            info["session_id"] = latest_id
            return info
        return None

    def search_recent_events(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Return the most recent memory interactions."""
        interactions = self.short_term_memory.get("recent_interactions", [])
        return interactions[-limit:]

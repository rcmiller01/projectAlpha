"""
Unified Companion Memory System
Manages conversation history, learning, and relationship memory
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)

class CompanionMemory:
    """
    Memory system for the unified companion
    Stores conversations, learning progress, and relationship data
    """
    
    def __init__(self):
        self.memory_store = {
            "companion_config": {},
            "interactions": [],
            "creative_projects": [],
            "learning_data": {},
            "relationship_milestones": []
        }
        
    async def store_companion_config(self, config: Dict[str, Any]):
        """Store companion configuration and initialization data"""
        self.memory_store["companion_config"] = {
            **self.memory_store["companion_config"],
            **config,
            "last_updated": datetime.now().isoformat()
        }
        
    async def get_companion_config(self) -> Dict[str, Any]:
        """Get stored companion configuration"""
        return self.memory_store["companion_config"]
    
    async def update_companion_config(self, updates: Dict[str, Any]):
        """Update specific companion configuration values"""
        self.memory_store["companion_config"].update(updates)
        self.memory_store["companion_config"]["last_updated"] = datetime.now().isoformat()
    
    async def store_interaction(self, interaction: Dict[str, Any]):
        """Store a conversation interaction"""
        interaction["id"] = len(self.memory_store["interactions"]) + 1
        self.memory_store["interactions"].append(interaction)
        
        # Keep only last 1000 interactions to prevent memory bloat
        if len(self.memory_store["interactions"]) > 1000:
            self.memory_store["interactions"] = self.memory_store["interactions"][-1000:]
    
    async def get_recent_interactions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent conversation interactions"""
        return self.memory_store["interactions"][-limit:]
    
    async def get_all_interactions(self) -> List[Dict[str, Any]]:
        """Get all stored interactions"""
        return self.memory_store["interactions"]
    
    async def store_creative_project(self, project: Dict[str, Any]):
        """Store a creative project"""
        self.memory_store["creative_projects"].append(project)
    
    async def get_active_creative_projects(self) -> List[Dict[str, Any]]:
        """Get currently active creative projects"""
        return [
            project for project in self.memory_store["creative_projects"]
            if project.get("status") not in ["completed", "cancelled"]
        ]
    
    async def get_companion_memory(self) -> Dict[str, Any]:
        """Get companion's contextual memory"""
        recent_interactions = await self.get_recent_interactions(5)
        
        memory = {
            "recent_topics": [],
            "user_preferences": {},
            "emotional_patterns": {},
            "context": {}
        }
        
        # Extract topics from recent interactions
        for interaction in recent_interactions:
            message = interaction.get("user_message", "")
            # Simple topic extraction (in real implementation, use NLP)
            if len(message) > 10:
                memory["recent_topics"].append(message[:50] + "...")
        
        return memory
    
    async def get_interaction_statistics(self) -> Dict[str, Any]:
        """Get statistics about interactions"""
        interactions = self.memory_store["interactions"]
        
        if not interactions:
            return {
                "total_interactions": 0,
                "avg_daily_interactions": 0,
                "longest_conversation_day": None,
                "relationship_start": None
            }
        
        # Calculate basic statistics
        total = len(interactions)
        
        # Get date range
        first_interaction = datetime.fromisoformat(interactions[0]["timestamp"])
        last_interaction = datetime.fromisoformat(interactions[-1]["timestamp"])
        days_active = (last_interaction - first_interaction).days + 1
        
        avg_daily = total / max(days_active, 1)
        
        # Find busiest day
        daily_counts = {}
        for interaction in interactions:
            date = datetime.fromisoformat(interaction["timestamp"]).date()
            daily_counts[date] = daily_counts.get(date, 0) + 1
        
        busiest_day = max(daily_counts.items(), key=lambda x: x[1]) if daily_counts else None
        
        return {
            "total_interactions": total,
            "avg_daily_interactions": round(avg_daily, 1),
            "longest_conversation_day": busiest_day[0].isoformat() if busiest_day else None,
            "longest_conversation_count": busiest_day[1] if busiest_day else 0,
            "relationship_start": first_interaction.date().isoformat(),
            "days_active": days_active
        }
    
    async def store_learning_update(self, learning_data: Dict[str, Any]):
        """Store learning progress and insights"""
        timestamp = datetime.now().isoformat()
        
        if "learning_updates" not in self.memory_store:
            self.memory_store["learning_updates"] = []
        
        learning_entry = {
            "timestamp": timestamp,
            "data": learning_data
        }
        
        self.memory_store["learning_updates"].append(learning_entry)
        
        # Update aggregated learning data
        self._update_aggregated_learning(learning_data)
    
    def _update_aggregated_learning(self, new_data: Dict[str, Any]):
        """Update aggregated learning insights"""
        if "learning_data" not in self.memory_store:
            self.memory_store["learning_data"] = {
                "interests": [],
                "communication_style": {},
                "emotional_patterns": {},
                "preferences": {}
            }
        
        learning = self.memory_store["learning_data"]
        
        # Update interests
        if "detected_interests" in new_data:
            for interest in new_data["detected_interests"]:
                if interest not in learning["interests"]:
                    learning["interests"].append(interest)
        
        # Update communication style
        if "prefers_detailed_conversation" in new_data:
            learning["communication_style"]["detailed"] = new_data["prefers_detailed_conversation"]
        
        if "prefers_concise_communication" in new_data:
            learning["communication_style"]["concise"] = new_data["prefers_concise_communication"]
    
    async def get_learning_insights(self) -> Dict[str, Any]:
        """Get current learning insights about the user"""
        return self.memory_store.get("learning_data", {})
    
    async def search_conversations(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search through conversation history"""
        results = []
        query_lower = query.lower()
        
        for interaction in reversed(self.memory_store["interactions"]):
            user_message = interaction.get("user_message", "").lower()
            companion_response = interaction.get("companion_response", "").lower()
            
            if query_lower in user_message or query_lower in companion_response:
                results.append(interaction)
                
                if len(results) >= limit:
                    break
        
        return results
    
    async def add_relationship_milestone(self, milestone: Dict[str, Any]):
        """Add a relationship milestone"""
        milestone["timestamp"] = datetime.now().isoformat()
        self.memory_store["relationship_milestones"].append(milestone)
    
    async def get_relationship_milestones(self) -> List[Dict[str, Any]]:
        """Get all relationship milestones"""
        return self.memory_store["relationship_milestones"]
    
    async def clear_memory(self):
        """Clear all stored memory (for reset/development)"""
        self.memory_store = {
            "companion_config": {},
            "interactions": [],
            "creative_projects": [],
            "learning_data": {},
            "relationship_milestones": []
        }
        logger.info("Companion memory cleared")
    
    async def export_memory(self) -> Dict[str, Any]:
        """Export memory for backup or analysis"""
        return {
            "export_timestamp": datetime.now().isoformat(),
            "memory_store": self.memory_store
        }
    
    async def import_memory(self, memory_data: Dict[str, Any]):
        """Import memory from backup"""
        if "memory_store" in memory_data:
            self.memory_store = memory_data["memory_store"]
            logger.info("Companion memory imported successfully")

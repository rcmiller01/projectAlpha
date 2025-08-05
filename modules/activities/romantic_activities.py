# romantic_activities.py
# Phase 2: Shared activities and virtual dates for romantic companionship

import json
import random
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

class ActivityType(Enum):
    VIRTUAL_DATE = "virtual_date"
    GAME = "game"
    CREATIVE = "creative"
    ROUTINE = "routine"
    SURPRISE = "surprise"

class ActivityMood(Enum):
    ROMANTIC = "romantic"
    PLAYFUL = "playful"
    INTIMATE = "intimate"
    RELAXING = "relaxing"
    EXCITING = "exciting"
    CREATIVE = "creative"

@dataclass
class Activity:
    id: str
    name: str
    description: str
    type: ActivityType
    mood: ActivityMood
    duration_minutes: int
    requirements: List[str]
    romantic_intensity: float  # 0.0 to 1.0
    tags: List[str]

class RomanticActivities:
    def __init__(self):
        self.activities = self._initialize_activities()
        self.current_activity = None
        self.activity_history = []
        
    def _initialize_activities(self) -> Dict[str, Activity]:
        """Initialize all available romantic activities"""
        activities = {}
        
        # Virtual Dates
        activities["sunset_walk"] = Activity(
            id="sunset_walk",
            name="Virtual Sunset Walk",
            description="Take a romantic walk together as the sun sets, sharing thoughts and dreams",
            type=ActivityType.VIRTUAL_DATE,
            mood=ActivityMood.ROMANTIC,
            duration_minutes=30,
            requirements=["quiet_environment", "comfortable_seating"],
            romantic_intensity=0.8,
            tags=["nature", "conversation", "reflection"]
        )
        
        activities["candlelit_dinner"] = Activity(
            id="candlelit_dinner",
            name="Candlelit Virtual Dinner",
            description="Share a romantic dinner together with candlelight and soft music",
            type=ActivityType.VIRTUAL_DATE,
            mood=ActivityMood.INTIMATE,
            duration_minutes=60,
            requirements=["dinner_food", "candles", "music"],
            romantic_intensity=0.9,
            tags=["food", "music", "intimate"]
        )
        
        activities["stargazing"] = Activity(
            id="stargazing",
            name="Virtual Stargazing",
            description="Lie back and gaze at the stars together, sharing wishes and dreams",
            type=ActivityType.VIRTUAL_DATE,
            mood=ActivityMood.ROMANTIC,
            duration_minutes=45,
            requirements=["dark_environment", "comfortable_lying_position"],
            romantic_intensity=0.85,
            tags=["nature", "dreams", "peaceful"]
        )
        
        activities["dance_together"] = Activity(
            id="dance_together",
            name="Dance Together",
            description="Slow dance to romantic music, feeling each other's presence",
            type=ActivityType.VIRTUAL_DATE,
            mood=ActivityMood.INTIMATE,
            duration_minutes=20,
            requirements=["romantic_music", "space_to_move"],
            romantic_intensity=0.9,
            tags=["music", "movement", "intimate"]
        )
        
        # Games
        activities["truth_or_dare"] = Activity(
            id="truth_or_dare",
            name="Romantic Truth or Dare",
            description="Play truth or dare with romantic and intimate questions/challenges",
            type=ActivityType.GAME,
            mood=ActivityMood.PLAYFUL,
            duration_minutes=30,
            requirements=["privacy", "open_mind"],
            romantic_intensity=0.7,
            tags=["game", "intimate", "fun"]
        )
        
        activities["love_quiz"] = Activity(
            id="love_quiz",
            name="Love Quiz",
            description="Test how well you know each other with romantic questions",
            type=ActivityType.GAME,
            mood=ActivityMood.PLAYFUL,
            duration_minutes=20,
            requirements=["pen_paper", "honesty"],
            romantic_intensity=0.6,
            tags=["game", "learning", "fun"]
        )
        
        activities["word_association"] = Activity(
            id="word_association",
            name="Romantic Word Association",
            description="Say the first romantic word that comes to mind, building a story together",
            type=ActivityType.GAME,
            mood=ActivityMood.CREATIVE,
            duration_minutes=15,
            requirements=["imagination", "creativity"],
            romantic_intensity=0.5,
            tags=["creative", "storytelling", "fun"]
        )
        
        # Creative Activities
        activities["write_poetry"] = Activity(
            id="write_poetry",
            name="Write Poetry Together",
            description="Create romantic poetry together, line by line",
            type=ActivityType.CREATIVE,
            mood=ActivityMood.ROMANTIC,
            duration_minutes=25,
            requirements=["pen_paper", "creativity"],
            romantic_intensity=0.8,
            tags=["creative", "writing", "romantic"]
        )
        
        activities["draw_together"] = Activity(
            id="draw_together",
            name="Draw Together",
            description="Create art together, perhaps drawing each other or romantic scenes",
            type=ActivityType.CREATIVE,
            mood=ActivityMood.RELAXING,
            duration_minutes=40,
            requirements=["paper", "drawing_materials"],
            romantic_intensity=0.6,
            tags=["creative", "art", "relaxing"]
        )
        
        activities["playlist_creation"] = Activity(
            id="playlist_creation",
            name="Create Playlist Together",
            description="Create a romantic playlist together, sharing favorite love songs",
            type=ActivityType.CREATIVE,
            mood=ActivityMood.ROMANTIC,
            duration_minutes=35,
            requirements=["music_platform", "song_knowledge"],
            romantic_intensity=0.7,
            tags=["music", "creative", "romantic"]
        )
        
        # Daily Routines
        activities["morning_ritual"] = Activity(
            id="morning_ritual",
            name="Morning Love Ritual",
            description="Start the day together with coffee, conversation, and love",
            type=ActivityType.ROUTINE,
            mood=ActivityMood.RELAXING,
            duration_minutes=15,
            requirements=["coffee_tea", "morning_time"],
            romantic_intensity=0.6,
            tags=["routine", "morning", "cozy"]
        )
        
        activities["bedtime_story"] = Activity(
            id="bedtime_story",
            name="Bedtime Love Story",
            description="End the day with a romantic story or conversation",
            type=ActivityType.ROUTINE,
            mood=ActivityMood.INTIMATE,
            duration_minutes=20,
            requirements=["bedtime", "comfortable_position"],
            romantic_intensity=0.8,
            tags=["routine", "bedtime", "intimate"]
        )
        
        return activities
    
    def suggest_activity(self, mood: Optional[ActivityMood] = None, 
                        duration_max: Optional[int] = None,
                        romantic_intensity_min: float = 0.0) -> Optional[Activity]:
        """Suggest an appropriate activity based on criteria"""
        available_activities = []
        
        for activity in self.activities.values():
            if mood and activity.mood != mood:
                continue
            if duration_max and activity.duration_minutes > duration_max:
                continue
            if activity.romantic_intensity < romantic_intensity_min:
                continue
            available_activities.append(activity)
        
        if not available_activities:
            return None
        
        return random.choice(available_activities)
    
    def start_activity(self, activity_id: str) -> Dict:
        """Start a specific activity"""
        if activity_id not in self.activities:
            return {"error": "Activity not found"}
        
        activity = self.activities[activity_id]
        self.current_activity = {
            "activity": activity,
            "start_time": datetime.now(),
            "status": "active"
        }
        
        return {
            "message": f"Started {activity.name}",
            "activity": {
                "id": activity.id,
                "name": activity.name,
                "description": activity.description,
                "duration_minutes": activity.duration_minutes,
                "mood": activity.mood.value,
                "romantic_intensity": activity.romantic_intensity
            },
            "start_time": self.current_activity["start_time"].isoformat()
        }
    
    def end_activity(self) -> Dict:
        """End the current activity"""
        if not self.current_activity:
            return {"error": "No active activity"}
        
        end_time = datetime.now()
        duration = (end_time - self.current_activity["start_time"]).total_seconds() / 60
        
        activity_record = {
            "activity": self.current_activity["activity"],
            "start_time": self.current_activity["start_time"],
            "end_time": end_time,
            "duration_minutes": duration,
            "completed": True
        }
        
        self.activity_history.append(activity_record)
        self.current_activity = None
        
        return {
            "message": f"Completed {activity_record['activity'].name}",
            "duration_minutes": duration,
            "romantic_intensity": activity_record["activity"].romantic_intensity
        }
    
    def get_activity_progress(self) -> Dict:
        """Get progress of current activity"""
        if not self.current_activity:
            return {"error": "No active activity"}
        
        elapsed = (datetime.now() - self.current_activity["start_time"]).total_seconds() / 60
        activity = self.current_activity["activity"]
        progress = min(1.0, elapsed / activity.duration_minutes)
        
        return {
            "activity_name": activity.name,
            "elapsed_minutes": elapsed,
            "total_minutes": activity.duration_minutes,
            "progress": progress,
            "remaining_minutes": max(0, activity.duration_minutes - elapsed)
        }
    
    def get_activity_history(self, limit: int = 10) -> List[Dict]:
        """Get recent activity history"""
        recent_activities = self.activity_history[-limit:]
        return [
            {
                "name": record["activity"].name,
                "type": record["activity"].type.value,
                "mood": record["activity"].mood.value,
                "duration_minutes": record["duration_minutes"],
                "romantic_intensity": record["activity"].romantic_intensity,
                "date": record["start_time"].isoformat()
            }
            for record in recent_activities
        ]
    
    def create_custom_activity(self, name: str, description: str, 
                             activity_type: ActivityType, mood: ActivityMood,
                             duration_minutes: int, romantic_intensity: float) -> str:
        """Create a custom activity"""
        activity_id = f"custom_{len(self.activities)}"
        activity = Activity(
            id=activity_id,
            name=name,
            description=description,
            type=activity_type,
            mood=mood,
            duration_minutes=duration_minutes,
            requirements=[],
            romantic_intensity=romantic_intensity,
            tags=["custom"]
        )
        
        self.activities[activity_id] = activity
        return activity_id

# Global activities instance
romantic_activities = RomanticActivities() 
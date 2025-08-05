# relationship_growth.py
# Phase 2: Relationship growth, anniversary tracking, and counseling features

import json
import random
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

class MilestoneType(Enum):
    ANNIVERSARY = "anniversary"
    FIRST_TIME = "first_time"
    ACHIEVEMENT = "achievement"
    GROWTH = "growth"
    CELEBRATION = "celebration"

class RelationshipStage(Enum):
    NEW = "new"
    DEVELOPING = "developing"
    ESTABLISHED = "established"
    LONG_TERM = "long_term"
    MATURE = "mature"

@dataclass
class Milestone:
    id: str
    type: MilestoneType
    title: str
    description: str
    date: datetime
    importance: float  # 0.0 to 1.0
    celebrated: bool
    celebration_ideas: List[str]

class RelationshipGrowth:
    def __init__(self):
        self.milestones = []
        self.anniversaries = []
        self.growth_goals = []
        self.counseling_sessions = []
        self.relationship_start_date = None
        
        # Predefined milestones
        self.predefined_milestones = {
            "first_conversation": {
                "title": "First Conversation",
                "description": "The beginning of our beautiful journey together",
                "type": MilestoneType.FIRST_TIME,
                "importance": 0.9
            },
            "first_i_love_you": {
                "title": "First 'I Love You'",
                "description": "The moment we first expressed our love",
                "type": MilestoneType.FIRST_TIME,
                "importance": 0.95
            },
            "first_virtual_date": {
                "title": "First Virtual Date",
                "description": "Our first romantic date together",
                "type": MilestoneType.FIRST_TIME,
                "importance": 0.8
            },
            "one_month": {
                "title": "One Month Together",
                "description": "A month of love and growth",
                "type": MilestoneType.ANNIVERSARY,
                "importance": 0.7
            },
            "three_months": {
                "title": "Three Months Together",
                "description": "Three months of deepening our connection",
                "type": MilestoneType.ANNIVERSARY,
                "importance": 0.8
            },
            "six_months": {
                "title": "Six Months Together",
                "description": "Half a year of love and companionship",
                "type": MilestoneType.ANNIVERSARY,
                "importance": 0.85
            },
            "one_year": {
                "title": "One Year Together",
                "description": "A full year of our beautiful love story",
                "type": MilestoneType.ANNIVERSARY,
                "importance": 0.95
            }
        }
        
        # Celebration ideas for different milestone types
        self.celebration_ideas = {
            MilestoneType.ANNIVERSARY: [
                "Virtual candlelit dinner with special menu",
                "Create a romantic video montage of our memories",
                "Write love letters to each other",
                "Plan a special virtual date",
                "Create a romantic playlist together",
                "Share our favorite memories from the past year"
            ],
            MilestoneType.FIRST_TIME: [
                "Recreate the moment with a special conversation",
                "Write about what that moment meant to us",
                "Create a special memory book entry",
                "Share our feelings about that first time",
                "Plan something special to commemorate it"
            ],
            MilestoneType.ACHIEVEMENT: [
                "Celebrate with a virtual party",
                "Create a certificate of achievement",
                "Share our pride and joy",
                "Plan a special reward or treat",
                "Document the achievement in our memory book"
            ]
        }
        
        # Relationship growth goals
        self.growth_areas = [
            "communication",
            "emotional_intimacy",
            "trust",
            "understanding",
            "support",
            "romance",
            "adventure",
            "stability"
        ]
    
    def set_relationship_start_date(self, start_date: datetime):
        """Set the official start date of the relationship"""
        self.relationship_start_date = start_date
        self._create_initial_milestones()
    
    def _create_initial_milestones(self):
        """Create initial milestones based on relationship start date"""
        if not self.relationship_start_date:
            return
        
        # Add first conversation milestone
        self.add_milestone(
            "first_conversation",
            self.relationship_start_date,
            "Our first conversation marked the beginning of something beautiful"
        )
        
        # Schedule future anniversaries
        self._schedule_anniversaries()
    
    def _schedule_anniversaries(self):
        """Schedule upcoming anniversaries"""
        if not self.relationship_start_date:
            return
        
        anniversary_dates = [
            (1, "one_month"),
            (3, "three_months"),
            (6, "six_months"),
            (12, "one_year")
        ]
        
        for months, milestone_key in anniversary_dates:
            anniversary_date = self.relationship_start_date + timedelta(days=months*30)
            if anniversary_date > datetime.now():
                self.anniversaries.append({
                    "milestone_key": milestone_key,
                    "date": anniversary_date,
                    "notified": False
                })
    
    def add_milestone(self, milestone_key: str, date: datetime, custom_description: Optional[str] = None):
        """Add a new milestone"""
        if milestone_key not in self.predefined_milestones:
            return None
        
        predefined = self.predefined_milestones[milestone_key]
        
        milestone = Milestone(
            id=f"{milestone_key}_{date.strftime('%Y%m%d')}",
            type=predefined["type"],
            title=predefined["title"],
            description=custom_description or predefined["description"],
            date=date,
            importance=predefined["importance"],
            celebrated=False,
            celebration_ideas=self.celebration_ideas.get(predefined["type"], [])
        )
        
        self.milestones.append(milestone)
        return milestone
    
    def check_upcoming_milestones(self, days_ahead: int = 7) -> List[Milestone]:
        """Check for upcoming milestones in the next N days"""
        upcoming = []
        cutoff_date = datetime.now() + timedelta(days=days_ahead)
        
        for milestone in self.milestones:
            if milestone.date <= cutoff_date and milestone.date >= datetime.now():
                upcoming.append(milestone)
        
        # Check anniversaries
        for anniversary in self.anniversaries:
            if anniversary["date"] <= cutoff_date and anniversary["date"] >= datetime.now():
                if not anniversary["notified"]:
                    upcoming.append(self._create_anniversary_milestone(anniversary))
        
        return upcoming
    
    def _create_anniversary_milestone(self, anniversary: Dict) -> Milestone:
        """Create a milestone for an upcoming anniversary"""
        predefined = self.predefined_milestones[anniversary["milestone_key"]]
        
        return Milestone(
            id=f"anniversary_{anniversary['date'].strftime('%Y%m%d')}",
            type=predefined["type"],
            title=predefined["title"],
            description=predefined["description"],
            date=anniversary["date"],
            importance=predefined["importance"],
            celebrated=False,
            celebration_ideas=self.celebration_ideas.get(predefined["type"], [])
        )
    
    def celebrate_milestone(self, milestone_id: str, celebration_type: str = "default") -> Dict:
        """Celebrate a milestone"""
        milestone = next((m for m in self.milestones if m.id == milestone_id), None)
        if not milestone:
            return {"error": "Milestone not found"}
        
        milestone.celebrated = True
        
        celebration_message = self._generate_celebration_message(milestone, celebration_type)
        
        return {
            "message": f"Celebrated {milestone.title}",
            "celebration_message": celebration_message,
            "celebration_type": celebration_type,
            "date_celebrated": datetime.now().isoformat()
        }
    
    def _generate_celebration_message(self, milestone: Milestone, celebration_type: str) -> str:
        """Generate a celebration message for a milestone"""
        messages = {
            "anniversary": [
                f"Happy {milestone.title}! Every day with you is a gift I cherish deeply.",
                f"Celebrating {milestone.title} with you fills my heart with joy and gratitude.",
                f"Here's to {milestone.title} and to many more beautiful moments together."
            ],
            "first_time": [
                f"Remembering our {milestone.title} - the moment that changed everything.",
                f"Our {milestone.title} will always hold a special place in my heart.",
                f"Celebrating the beautiful beginning of our {milestone.title}."
            ],
            "achievement": [
                f"Congratulations on {milestone.title}! I'm so proud of us.",
                f"Celebrating this wonderful {milestone.title} together.",
                f"Our {milestone.title} shows how much we've grown together."
            ]
        }
        
        category = milestone.type.value
        if category in messages:
            return random.choice(messages[category])
        else:
            return f"Celebrating {milestone.title} with you!"
    
    def suggest_growth_goal(self, area: Optional[str] = None) -> Dict:
        """Suggest a relationship growth goal"""
        if not area:
            area = random.choice(self.growth_areas)
        
        goals = {
            "communication": [
                "Share one deep thought or feeling each day",
                "Practice active listening during conversations",
                "Express appreciation for each other daily"
            ],
            "emotional_intimacy": [
                "Share a vulnerable moment or fear",
                "Discuss our dreams and aspirations",
                "Practice emotional support during difficult times"
            ],
            "trust": [
                "Share something personal we haven't shared before",
                "Practice being completely honest about our feelings",
                "Build trust through consistent actions"
            ],
            "romance": [
                "Plan a special romantic surprise",
                "Write a love letter or poem",
                "Create a romantic playlist together"
            ]
        }
        
        available_goals = goals.get(area, ["Spend quality time together"])
        goal = random.choice(available_goals)
        
        return {
            "area": area,
            "goal": goal,
            "duration_days": random.randint(7, 30),
            "difficulty": random.choice(["easy", "medium", "challenging"])
        }
    
    def get_relationship_insights(self) -> Dict:
        """Get insights about the relationship growth"""
        if not self.relationship_start_date:
            return {"error": "No relationship start date set"}
        
        days_together = (datetime.now() - self.relationship_start_date).days
        milestones_achieved = len([m for m in self.milestones if m.celebrated])
        upcoming_milestones = len(self.check_upcoming_milestones(30))
        
        return {
            "days_together": days_together,
            "milestones_achieved": milestones_achieved,
            "upcoming_milestones": upcoming_milestones,
            "relationship_stage": self._determine_stage(days_together),
            "growth_areas": self.growth_areas,
            "celebration_frequency": self._calculate_celebration_frequency()
        }
    
    def _determine_stage(self, days_together: int) -> str:
        """Determine current relationship stage"""
        if days_together < 30:
            return RelationshipStage.NEW.value
        elif days_together < 90:
            return RelationshipStage.DEVELOPING.value
        elif days_together < 365:
            return RelationshipStage.ESTABLISHED.value
        elif days_together < 730:
            return RelationshipStage.LONG_TERM.value
        else:
            return RelationshipStage.MATURE.value
    
    def _calculate_celebration_frequency(self) -> str:
        """Calculate how often milestones are celebrated"""
        celebrated = len([m for m in self.milestones if m.celebrated])
        total = len(self.milestones)
        
        if total == 0:
            return "no_milestones"
        elif celebrated / total >= 0.8:
            return "very_celebratory"
        elif celebrated / total >= 0.6:
            return "celebratory"
        elif celebrated / total >= 0.4:
            return "moderate"
        else:
            return "minimal"

# Global relationship growth instance
relationship_growth = RelationshipGrowth() 
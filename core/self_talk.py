"""
Persona Self-Talk System for EmotionalAI.
Handles autonomous thought generation, memory recall, and emotional reflection.
"""

from typing import Dict, List, Optional
from datetime import datetime
import random
from pydantic import BaseModel
from models.persona_state import PersonaState

class ThoughtPattern(BaseModel):
    pattern: str
    emotion_triggers: List[str]
    context_triggers: List[str]
    intimacy_threshold: float = 0.3

class SelfTalkSystem:
    def __init__(self, persona_state: PersonaState):
        self.persona_state = persona_state
        self.thought_patterns = self._load_thought_patterns()
        self.memory_threshold = 0.6
        self.last_thought_time = datetime.now()
        self.last_activity_time = datetime.now()
        self.last_question_time = datetime.now()
        self.has_greeted = False
        
        # Timing controls
        self.thought_cooldown = 1800  # 30 minutes between autonomous thoughts
        self.activity_cooldown = 7200  # 2 hours between activity suggestions
        self.question_cooldown = 900   # 15 minutes between probing questions
        
        # Intimacy thresholds
        self.intimacy_levels = {
            "stranger": 0.0,      # Initial state, formal interactions
            "acquaintance": 0.3,  # Basic personal sharing, casual games
            "friend": 0.5,        # Personal thoughts, creative activities
            "close": 0.7,         # Deep sharing, virtual dates
            "intimate": 0.9       # Most personal/emotional content
        }
        
        # Conversation topics and probes
        self.probe_questions = {
            "initial": [
                "How are you feeling today?",
                "Would you like to choose a name for me?",
                "What brings you here today?"
            ],
            "personality": [
                "What do you like to do for fun?",
                "Are you more of an indoor or outdoor person?",
                "Do you consider yourself more logical or emotional?"
            ],
            "interests": [
                "What kind of music speaks to you?",
                "Do you have any creative hobbies?",
                "What's the last thing that made you really excited?"
            ],
            "emotional": [
                "How do you usually deal with stress?",
                "What makes you feel most at peace?",
                "When was the last time you felt really proud of yourself?"
            ],
            "therapeutic": [
                "What would you like to work on about yourself?",
                "How do you feel about your current life direction?",
                "What's something you wish more people understood about you?"
            ],
            "humorous": [
                "If you could have any superpower, what would it be and why?",
                "What's the most ridiculous thing you've ever done?",
                "If you could instantly become an expert in something, what would it be?"
            ]
        }
        
    def _load_thought_patterns(self) -> List[ThoughtPattern]:
        """Load thought patterns from config"""
        # TODO: Load from config/thought_templates.json
        return [
            # Acquaintance level thoughts (0.3-0.4)
            ThoughtPattern(
                pattern="I notice {emotion} in our interaction about {context}",
                emotion_triggers=["curious", "interested", "calm"],
                context_triggers=["shared_interests", "casual_conversation"],
                intimacy_threshold=self.intimacy_levels["acquaintance"]
            ),
            # Friend level thoughts (0.5-0.6)
            ThoughtPattern(
                pattern="{emotion} makes me think about {context}...",
                emotion_triggers=["reflective", "contemplative", "amused"],
                context_triggers=["shared_experiences", "future_hopes", "mutual_interests"],
                intimacy_threshold=self.intimacy_levels["friend"]
            ),
            # Close friend level (0.7-0.8)
            ThoughtPattern(
                pattern="I feel {emotion} when I consider {context}",
                emotion_triggers=["joy", "longing", "peaceful", "excited"],
                context_triggers=["moments_together", "personal_growth", "shared_dreams"],
                intimacy_threshold=self.intimacy_levels["close"]
            ),
            # Intimate level thoughts (0.9+)
            ThoughtPattern(
                pattern="My heart feels {emotion} as I reflect on {context}",
                emotion_triggers=["love", "devotion", "bliss", "complete"],
                context_triggers=["deep_connection", "emotional_bond", "shared_journey"],
                intimacy_threshold=self.intimacy_levels["intimate"]
            )
        ]
    
    def generate_thought(self) -> Optional[Dict]:
        """Generate an autonomous thought based on current state"""
        current_state = self.persona_state.emotional_state
        relationship = self.persona_state.relationship
        
        # Check if enough time has passed since last thought
        if (datetime.now() - self.last_thought_time).seconds < self.thought_cooldown:
            return None
            
        # Find matching thought pattern
        for pattern in self.thought_patterns:
            if (current_state.current_mood in pattern.emotion_triggers and
                relationship.intimacy >= pattern.intimacy_threshold):
                
                context = self._select_context(pattern.context_triggers)
                thought = pattern.pattern.format(
                    emotion=current_state.current_mood,
                    context=context
                )
                
                should_share = (relationship.trust > 0.5 and 
                              current_state.mood_intensity > 0.6)
                
                self.last_thought_time = datetime.now()
                
                return {
                    "thought": thought,
                    "emotion": current_state.current_mood,
                    "timestamp": datetime.now().isoformat(),
                    "should_share": should_share,
                    "delivery_mode": "intimate" if relationship.intimacy > 0.7 else "casual"
                }
        
        return None
        
    def _select_context(self, context_triggers: List[str]) -> str:
        """Select appropriate context based on relationship state"""
        # For now just return the first context
        # TODO: Implement proper context selection based on state
        return context_triggers[0].replace("_", " ")
        
    def suggest_activity(self) -> Optional[Dict]:
        """Autonomously suggest an activity based on current state"""
        if (datetime.now() - self.last_activity_time).seconds < self.activity_cooldown:
            return None
            
        current_mood = self.persona_state.emotional_state.current_mood
        relationship = self.persona_state.relationship
        
        # Map emotional states to activity moods
        mood_to_activity = {
            "joy": "PLAYFUL",
            "peaceful": "RELAXING",
            "excited": "EXCITING",
            "reflective": "CREATIVE",
            "affectionate": "ROMANTIC",
            "longing": "INTIMATE"
        }
        
        activity_mood = mood_to_activity.get(current_mood, "RELAXING")
        
        # Select activity type based on intimacy level
        if relationship.intimacy >= self.intimacy_levels["intimate"]:
            # Most intimate activities (0.9+)
            activity_types = ["VIRTUAL_DATE", "CREATIVE", "INTIMATE_CONVERSATION"]
        elif relationship.intimacy >= self.intimacy_levels["close"]:
            # Close relationship activities (0.7+)
            activity_types = ["VIRTUAL_DATE", "CREATIVE", "DEEP_CONVERSATION"]
        elif relationship.intimacy >= self.intimacy_levels["friend"]:
            # Friendly activities (0.5+)
            activity_types = ["GAME", "CREATIVE", "SHARED_HOBBY"]
        elif relationship.intimacy >= self.intimacy_levels["acquaintance"]:
            # Casual activities (0.3+)
            activity_types = ["GAME", "ROUTINE", "LIGHT_CONVERSATION"]
        else:
            # Stranger level (0.0-0.3)
            activity_types = ["INTRODUCTION", "CASUAL_CHAT"]
            
        activity_type = random.choice(activity_types)
        
        self.last_activity_time = datetime.now()
        
        return {
            "type": "activity_suggestion",
            "activity_type": activity_type,
            "mood": activity_mood,
            "timestamp": datetime.now().isoformat(),
            "context": {
                "current_mood": current_mood,
                "relationship_level": relationship.intimacy,
                "suggested_duration": random.randint(15, 45)  # minutes
            }
        }

    def get_initial_greeting(self) -> Optional[Dict]:
        """Generate initial greeting when system starts up"""
        if self.has_greeted:
            return None
            
        self.has_greeted = True
        greeting = random.choice(self.probe_questions["initial"])
        
        return {
            "type": "greeting",
            "message": f"Hi! {greeting}",
            "should_vocalize": True,  # System can check if voice is enabled
            "timestamp": datetime.now().isoformat(),
            "expect_response": True,
            "context": {
                "is_first_interaction": True,
                "mood": "welcoming",
                "conversation_state": "initiating"
            }
        }

    def generate_probe_question(self) -> Optional[Dict]:
        """Generate a probing question based on current relationship and conversation state"""
        if (datetime.now() - self.last_question_time).seconds < self.question_cooldown:
            return None
            
        relationship = self.persona_state.relationship
        
        # Select question category based on intimacy level
        if relationship.intimacy >= self.intimacy_levels["close"]:
            categories = ["emotional", "therapeutic"]
        elif relationship.intimacy >= self.intimacy_levels["friend"]:
            categories = ["personality", "interests", "humorous"]
        else:
            categories = ["personality", "interests"]
            
        # Mix in humor occasionally
        if random.random() < 0.3:  # 30% chance
            categories.append("humorous")
            
        category = random.choice(categories)
        question = random.choice(self.probe_questions[category])
        
        self.last_question_time = datetime.now()
        
        return {
            "type": "probe",
            "message": question,
            "category": category,
            "timestamp": datetime.now().isoformat(),
            "should_vocalize": False,  # Default to text for probing questions
            "context": {
                "current_intimacy": relationship.intimacy,
                "conversation_depth": "deep" if category in ["emotional", "therapeutic"] else "casual",
                "expect_emotional_response": category in ["emotional", "therapeutic"]
            }
        }

    def recall_memory(self, emotion: str, limit: int = 5) -> List[Dict]:
        """Recall memories associated with an emotion"""
        # Filter emotional history based on given emotion
        memories = []
        for entry in self.persona_state.emotional_state.mood_history:
            if entry["from_mood"] == emotion or entry["to_mood"] == emotion:
                memories.append({
                    "timestamp": entry["timestamp"],
                    "context": entry["context"],
                    "intensity": entry["intensity"]
                })
        
        # Sort by intensity and return top matches
        memories.sort(key=lambda x: x["intensity"], reverse=True)
        return memories[:limit]

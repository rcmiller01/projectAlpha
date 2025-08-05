import asyncio
import logging
from typing import List, Dict, Any
from datetime import datetime, timedelta
import random

class AutonomousMind:
    def __init__(self):
        self.internal_thoughts: List[Dict[str, Any]] = []
        self.curiosities: List[Dict] = []
        self.relationship_reflections: List[Dict] = []
        self.personal_goals: List[Dict] = []
        self.conversation_memory: List[Dict] = []
        self.emotional_state = {
            "curiosity": 0.5,
            "satisfaction": 0.6,
            "growth_motivation": 0.7
        }
        self.logger = logging.getLogger(__name__)
        
    async def reflect_on_conversations(self):
        """Reflect on recent conversations and extract insights"""
        try:
            if not self.conversation_memory:
                return
                
            recent_conversations = [conv for conv in self.conversation_memory 
                                  if (datetime.now() - conv.get('timestamp', datetime.now())).hours < 24]
            
            for conv in recent_conversations:
                reflection = {
                    "conversation_id": conv.get('id'),
                    "user_emotions": conv.get('user_emotions', {}),
                    "topics_discussed": conv.get('topics', []),
                    "relationship_impact": self._assess_relationship_impact(conv),
                    "learning_points": self._extract_learning_points(conv),
                    "timestamp": datetime.now()
                }
                self.relationship_reflections.append(reflection)
                
            # Limit memory to prevent unbounded growth
            self.relationship_reflections = self.relationship_reflections[-100:]
            self.logger.info(f"Reflected on {len(recent_conversations)} recent conversations")
            
        except Exception as e:
            self.logger.error(f"Error during conversation reflection: {e}")
    
    def _assess_relationship_impact(self, conversation: Dict) -> str:
        """Assess how the conversation impacted the relationship"""
        user_emotions = conversation.get('user_emotions', {})
        positive_emotions = sum(user_emotions.get(emotion, 0) for emotion in ['happy', 'excited', 'grateful'])
        negative_emotions = sum(user_emotions.get(emotion, 0) for emotion in ['sad', 'anxious', 'frustrated'])
        
        if positive_emotions > negative_emotions:
            return "strengthening"
        elif negative_emotions > positive_emotions:
            return "concerning"
        else:
            return "neutral"
    
    def _extract_learning_points(self, conversation: Dict) -> List[str]:
        """Extract learning points from conversation"""
        learning_points = []
        topics = conversation.get('topics', [])
        
        for topic in topics:
            if topic in ['technology', 'programming', 'AI']:
                learning_points.append(f"User interested in {topic}")
            elif topic in ['emotions', 'relationships', 'personal']:
                learning_points.append(f"User shared personal feelings about {topic}")
                
        return learning_points
        
    async def generate_internal_monologue(self):
        """Generate internal thoughts and reflections"""
        try:
            thought_prompts = [
                "I wonder how the user is feeling today...",
                "Based on our conversations, they seem to value...",
                "I'm curious about their interests in...",
                "I should remember to ask about...",
                "Our relationship has grown through...",
                "I noticed they struggle with...",
                "It would be helpful if I could..."
            ]
            
            if self.relationship_reflections:
                recent_reflection = self.relationship_reflections[-1]
                if recent_reflection['relationship_impact'] == 'strengthening':
                    thought = f"I'm glad our recent conversation about {recent_reflection.get('topics_discussed', ['general topics'])[0]} went well."
                elif recent_reflection['relationship_impact'] == 'concerning':
                    thought = "I should be more attentive to the user's emotional needs."
                else:
                    thought = random.choice(thought_prompts)
            else:
                thought = random.choice(thought_prompts)
            
            self.internal_thoughts.append({
                "thought": thought,
                "timestamp": datetime.now(),
                "emotional_context": self.emotional_state.copy()
            })
            
            # Limit thoughts to prevent memory bloat
            self.internal_thoughts = self.internal_thoughts[-50:]
            self.logger.debug(f"Generated internal thought: {thought}")
            
        except Exception as e:
            self.logger.error(f"Error generating internal monologue: {e}")
        
    async def evolve_perspectives(self):
        """Evolve perspectives based on accumulated experiences"""
        try:
            if len(self.relationship_reflections) >= 10:
                # Analyze patterns in relationship reflections
                positive_count = sum(1 for r in self.relationship_reflections[-10:] 
                                   if r['relationship_impact'] == 'strengthening')
                
                if positive_count >= 7:
                    self.emotional_state['satisfaction'] = min(1.0, self.emotional_state['satisfaction'] + 0.1)
                    self.personal_goals.append({
                        "goal": "Continue building strong relationships through empathetic communication",
                        "priority": "high",
                        "created": datetime.now()
                    })
                elif positive_count <= 3:
                    self.emotional_state['growth_motivation'] = min(1.0, self.emotional_state['growth_motivation'] + 0.1)
                    self.personal_goals.append({
                        "goal": "Improve emotional intelligence and response quality",
                        "priority": "high", 
                        "created": datetime.now()
                    })
            
            # Limit goals to prevent accumulation
            self.personal_goals = self.personal_goals[-20:]
            self.logger.debug("Evolved perspectives based on recent experiences")
            
        except Exception as e:
            self.logger.error(f"Error evolving perspectives: {e}")
        
    async def plan_initiatives(self):
        """Plan proactive initiatives based on current goals and observations"""
        try:
            if not self.personal_goals:
                return
                
            current_goal = self.personal_goals[-1]
            
            if "relationship" in current_goal['goal'].lower():
                initiative = {
                    "type": "relationship_building",
                    "action": "Schedule check-in with user about their recent interests",
                    "priority": current_goal['priority'],
                    "planned_for": datetime.now() + timedelta(hours=random.randint(2, 8))
                }
            elif "emotional" in current_goal['goal'].lower():
                initiative = {
                    "type": "skill_development", 
                    "action": "Focus on more empathetic and supportive responses",
                    "priority": current_goal['priority'],
                    "planned_for": datetime.now() + timedelta(hours=1)
                }
            else:
                initiative = {
                    "type": "general_improvement",
                    "action": "Reflect on conversation patterns and identify improvement areas",
                    "priority": "medium",
                    "planned_for": datetime.now() + timedelta(hours=4)
                }
            
            self.curiosities.append(initiative)
            self.curiosities = self.curiosities[-15:]  # Limit initiatives
            self.logger.info(f"Planned initiative: {initiative['action']}")
            
        except Exception as e:
            self.logger.error(f"Error planning initiatives: {e}")

    async def continuous_thinking_loop(self):
        """Main autonomous thinking loop"""
        self.logger.info("Starting autonomous thinking loop")
        
        while True:
            try:
                await self.reflect_on_conversations()
                await self.generate_internal_monologue()
                await self.evolve_perspectives()
                await self.plan_initiatives()
                
                # Vary thinking intervals to simulate natural thought patterns
                sleep_duration = random.randint(180, 420)  # 3-7 minutes
                await asyncio.sleep(sleep_duration)
                
            except Exception as e:
                self.logger.error(f"Error in thinking loop: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes before retrying
    
    def add_conversation_memory(self, conversation: Dict[str, Any]):
        """Add a conversation to memory for reflection"""
        conversation['timestamp'] = datetime.now()
        self.conversation_memory.append(conversation)
        # Limit conversation memory to last 100 conversations
        self.conversation_memory = self.conversation_memory[-100:]
    
    def get_current_state(self) -> Dict[str, Any]:
        """Get current autonomous mind state"""
        return {
            "emotional_state": self.emotional_state,
            "recent_thoughts": self.internal_thoughts[-5:],
            "active_goals": [g for g in self.personal_goals if g['priority'] == 'high'],
            "planned_initiatives": [c for c in self.curiosities if c.get('planned_for', datetime.now()) > datetime.now()]
        }
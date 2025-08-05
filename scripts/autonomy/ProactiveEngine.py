from typing import Dict, Optional, List, Any
import asyncio
import logging
from datetime import datetime, timedelta
import random

class ProactiveEngine:
    def __init__(self):
        self.initiative_triggers: Dict[str, float] = {
            'curiosity_threshold': 0.7,
            'concern_threshold': 0.6,
            'excitement_threshold': 0.8,
            'relationship_milestone': True
        }
        self.user_state: Dict[str, Any] = {}
        self.last_interaction: Optional[datetime] = None
        self.interaction_history: List[Dict[str, Any]] = []
        self.pending_outreach: List[Dict[str, Any]] = []
        self.logger = logging.getLogger(__name__)
        
    def update_user_state(self, emotional_state: Dict[str, float], interaction_data: Dict[str, Any]):
        """Update user state based on latest interaction"""
        self.user_state = {
            "emotional_state": emotional_state,
            "last_topics": interaction_data.get('topics', []),
            "engagement_level": interaction_data.get('engagement_level', 0.5),
            "support_needs": interaction_data.get('support_needs', [])
        }
        self.last_interaction = datetime.now()
        self.interaction_history.append({
            "timestamp": self.last_interaction,
            "emotional_state": emotional_state,
            "data": interaction_data
        })
        # Keep only last 50 interactions
        self.interaction_history = self.interaction_history[-50:]
        
    def should_initiate_contact(self) -> bool:
        """Determine if proactive contact should be initiated"""
        try:
            # Check time since last interaction
            if self.last_interaction:
                hours_since_last = (datetime.now() - self.last_interaction).total_seconds() / 3600
                
                # More likely to reach out if it's been a while
                if hours_since_last > 48:  # 2 days
                    return True
                elif hours_since_last > 24:  # 1 day
                    return random.random() < 0.3
            
            # Check emotional state triggers
            if self.user_state.get('emotional_state'):
                emotions = self.user_state['emotional_state']
                
                # High concern levels trigger outreach
                concern_emotions = ['anxious', 'sad', 'stressed', 'overwhelmed']
                concern_level = sum(emotions.get(emotion, 0) for emotion in concern_emotions)
                if concern_level > self.initiative_triggers['concern_threshold']:
                    self.logger.info(f"Concern threshold triggered: {concern_level}")
                    return True
                
                # High positive emotions might trigger celebration
                positive_emotions = ['happy', 'excited', 'grateful']
                excitement_level = sum(emotions.get(emotion, 0) for emotion in positive_emotions)
                if excitement_level > self.initiative_triggers['excitement_threshold']:
                    self.logger.info(f"Excitement threshold triggered: {excitement_level}")
                    return random.random() < 0.4
            
            # Check for conversation patterns that suggest curiosity
            if len(self.interaction_history) >= 3:
                recent_topics = []
                for interaction in self.interaction_history[-3:]:
                    recent_topics.extend(interaction['data'].get('topics', []))
                
                # If user has been consistently exploring one topic
                topic_counts = {}
                for topic in recent_topics:
                    topic_counts[topic] = topic_counts.get(topic, 0) + 1
                
                max_count = max(topic_counts.values()) if topic_counts else 0
                if max_count >= 2:  # Same topic in multiple recent conversations
                    return random.random() < 0.25
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error determining contact initiation: {e}")
            return False
        
    def determine_message_type(self) -> str:
        """Determine what type of proactive message to send"""
        try:
            if not self.user_state.get('emotional_state'):
                return "general_check_in"
            
            emotions = self.user_state['emotional_state']
            
            # Prioritize support for negative emotions
            concern_emotions = ['anxious', 'sad', 'stressed', 'overwhelmed']
            concern_level = sum(emotions.get(emotion, 0) for emotion in concern_emotions)
            
            if concern_level > self.initiative_triggers['concern_threshold']:
                return "emotional_support"
            
            # Check for positive emotions to celebrate
            positive_emotions = ['happy', 'excited', 'grateful']
            excitement_level = sum(emotions.get(emotion, 0) for emotion in positive_emotions)
            
            if excitement_level > self.initiative_triggers['excitement_threshold']:
                return "positive_reinforcement"
            
            # Check recent topics for curiosity-driven messages
            if self.user_state.get('last_topics'):
                topics = self.user_state['last_topics']
                if 'learning' in topics or 'education' in topics:
                    return "knowledge_sharing"
                elif 'creative' in topics or 'art' in topics:
                    return "creative_inspiration"
                elif 'technical' in topics or 'programming' in topics:
                    return "technical_insight"
            
            # Default based on time patterns
            if self.last_interaction:
                hours_since = (datetime.now() - self.last_interaction).total_seconds() / 3600
                if hours_since > 48:
                    return "reconnection"
                elif hours_since > 24:
                    return "daily_check_in"
            
            return "general_curiosity"
            
        except Exception as e:
            self.logger.error(f"Error determining message type: {e}")
            return "general"
        
    async def send_spontaneous_message(self, message_type: str):
        """Generate and queue spontaneous message for sending"""
        try:
            message_templates = {
                "emotional_support": [
                    "I've been thinking about you and wanted to check in. How are you feeling today?",
                    "I noticed you seemed stressed in our last conversation. Is there anything I can help with?",
                    "Sometimes it helps to talk through difficult feelings. I'm here if you need support."
                ],
                "positive_reinforcement": [
                    "I loved hearing about your recent success! How has that been going?",
                    "Your enthusiasm in our last chat was infectious. I'd love to hear more about what's exciting you.",
                    "It's wonderful to see you in such good spirits. What's been bringing you joy lately?"
                ],
                "knowledge_sharing": [
                    "I came across something interesting related to what we discussed. Would you like to explore it together?",
                    "Your curiosity about [topic] got me thinking. I have some insights that might interest you.",
                    "There's a fascinating connection between our recent conversation topics. Want to dive deeper?"
                ],
                "creative_inspiration": [
                    "I had an idea that might spark your creativity. Would you like to hear it?",
                    "Your creative projects always impress me. How's your latest work coming along?",
                    "I found something that reminded me of your artistic style. Want to see?"
                ],
                "technical_insight": [
                    "I discovered a new approach to the technical challenge we discussed. Interested in exploring it?",
                    "There's an elegant solution to the problem you mentioned. Would you like to work through it together?",
                    "I've been thinking about your coding project. I have some suggestions that might help."
                ],
                "reconnection": [
                    "It's been a while since we talked. I've been wondering how you're doing.",
                    "I miss our conversations! What's new in your world?",
                    "Just wanted to reach out and see how things have been going for you."
                ],
                "daily_check_in": [
                    "Good morning! How's your day starting out?",
                    "Thinking of you today. What's on your mind?",
                    "Just checking in. How are you feeling today?"
                ],
                "general_check_in": [
                    "How are you doing today?",
                    "What's been on your mind lately?",
                    "I'm here if you want to chat about anything."
                ],
                "general_curiosity": [
                    "I'm curious about something. What's been the highlight of your week?",
                    "I've been wondering - what's something new you learned recently?",
                    "What's something you're looking forward to?"
                ]
            }
            
            messages = message_templates.get(message_type, message_templates["general_check_in"])
            selected_message = random.choice(messages)
            
            # Personalize based on recent topics if available
            if self.user_state.get('last_topics') and '[topic]' in selected_message:
                topic = random.choice(self.user_state['last_topics'])
                selected_message = selected_message.replace('[topic]', topic)
            
            outreach_item = {
                "message": selected_message,
                "message_type": message_type,
                "generated_at": datetime.now(),
                "priority": self._calculate_message_priority(message_type),
                "context": self.user_state.copy()
            }
            
            self.pending_outreach.append(outreach_item)
            self.logger.info(f"Generated proactive message: {message_type}")
            
            return outreach_item
            
        except Exception as e:
            self.logger.error(f"Error sending spontaneous message: {e}")
            return None
    
    def _calculate_message_priority(self, message_type: str) -> str:
        """Calculate priority level for different message types"""
        high_priority = ["emotional_support", "reconnection"]
        medium_priority = ["positive_reinforcement", "daily_check_in"]
        
        if message_type in high_priority:
            return "high"
        elif message_type in medium_priority:
            return "medium"
        else:
            return "low"

    async def evaluate_outreach_triggers(self):
        """Main evaluation loop for proactive outreach"""
        try:
            if self.should_initiate_contact():
                message_type = self.determine_message_type()
                await self.send_spontaneous_message(message_type)
                self.logger.info(f"Initiated proactive outreach: {message_type}")
            else:
                self.logger.debug("No outreach triggers activated")
                
        except Exception as e:
            self.logger.error(f"Error evaluating outreach triggers: {e}")
    
    def get_pending_messages(self) -> List[Dict[str, Any]]:
        """Get all pending proactive messages"""
        return self.pending_outreach.copy()
    
    def mark_message_sent(self, message_id: str):
        """Mark a message as sent and remove from pending"""
        self.pending_outreach = [msg for msg in self.pending_outreach 
                               if msg.get('id') != message_id]
    
    def get_engine_status(self) -> Dict[str, Any]:
        """Get current status of the proactive engine"""
        return {
            "last_interaction": self.last_interaction.isoformat() if self.last_interaction else None,
            "pending_messages": len(self.pending_outreach),
            "user_emotional_state": self.user_state.get('emotional_state', {}),
            "interaction_count": len(self.interaction_history),
            "triggers": self.initiative_triggers
        }
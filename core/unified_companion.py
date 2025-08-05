"""
Unified Companion Core System
Single AI companion that handles all interaction types
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)

class UnifiedCompanion:
    """
    Unified AI Companion that replaces the persona system
    Handles emotional support, creative collaboration, technical assistance, and more
    """
    
    def __init__(self, memory, emotion_detector, creative_discovery):
        self.memory = memory
        self.emotion_detector = emotion_detector
        self.creative_discovery = creative_discovery
        self.name = ""
        self.config = {}
        self.personality_traits = [
            "empathetic", "creative", "intelligent", 
            "supportive", "adaptable", "curious"
        ]
        self.conversation_context = {}
        
    async def initialize(self, name: str, config: Dict[str, Any]):
        """Initialize the companion with name and configuration"""
        self.name = name
        self.config = config
        
        # Load any existing memory/context
        existing_memory = await self.memory.get_companion_memory()
        if existing_memory:
            self.conversation_context = existing_memory.get("context", {})
        
        logger.info(f"Unified companion '{name}' initialized")
    
    async def generate_initial_greeting(self) -> str:
        """Generate the first greeting message after initialization"""
        
        greetings = [
            f"Hello! I'm {self.name}, and I'm so excited to be your companion! I'm here to help you with whatever you need - creative projects, deep conversations, technical questions, or just being a friend.",
            
            f"Hi there! I'm {self.name}. I'm thrilled to meet you! I love learning about people and helping them explore their interests and creativity.",
            
            f"Welcome! I'm {self.name}, your AI companion. I'm designed to be helpful, creative, and empathetic. I can't wait to see what we'll discover together!"
        ]
        
        import random
        return random.choice(greetings)
    
    async def process_message(self, message: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process user message and generate appropriate response
        This is the main intelligence hub that determines response type and style
        """
        
        # Analyze message intent and content
        intent = await self._analyze_message_intent(message, context)
        
        # Update conversation context
        self.conversation_context.update({
            "last_message": message,
            "last_intent": intent,
            "interaction_mode": context.get("interaction_mode", "text"),
            "response_style": context.get("response_style", "balanced"),
            "timestamp": datetime.now().isoformat()
        })
        
        # Generate response based on intent and style
        response = await self._generate_response(message, intent, context)
        
        # Check for learning opportunities
        learning_updates = await self._update_learning(message, response, context)
        
        # Generate proactive suggestions if appropriate
        proactive_suggestions = await self._generate_proactive_suggestions(context)
        
        return {
            "response": response,
            "intent": intent,
            "context": self.conversation_context,
            "learning_updates": learning_updates,
            "proactive_suggestions": proactive_suggestions
        }
    
    async def _analyze_message_intent(self, message: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze user message to determine intent and appropriate response approach"""
        
        message_lower = message.lower()
        
        intent = {
            "primary_category": "general",
            "emotional_tone": context.get("detected_emotion", "neutral"),
            "requires_creativity": False,
            "requires_technical": False,
            "requires_emotional_support": False,
            "suggests_project": False
        }
        
        # Creative collaboration detection
        creative_keywords = [
            "create", "make", "build", "design", "art", "music", "write", "story",
            "poem", "song", "image", "video", "creative", "artistic", "compose"
        ]
        if any(keyword in message_lower for keyword in creative_keywords):
            intent["primary_category"] = "creative"
            intent["requires_creativity"] = True
            intent["suggests_project"] = True
        
        # Technical assistance detection
        technical_keywords = [
            "code", "program", "debug", "error", "function", "algorithm", "script",
            "database", "api", "help me with", "how to", "explain", "tutorial"
        ]
        if any(keyword in message_lower for keyword in technical_keywords):
            intent["primary_category"] = "technical"
            intent["requires_technical"] = True
        
        # Emotional support detection
        emotional_keywords = [
            "feel", "feeling", "sad", "happy", "stressed", "worried", "excited",
            "anxious", "depressed", "lonely", "frustrated", "overwhelmed"
        ]
        if any(keyword in message_lower for keyword in emotional_keywords):
            intent["requires_emotional_support"] = True
            if intent["primary_category"] == "general":
                intent["primary_category"] = "emotional"
        
        # Conversational depth indicators
        deep_keywords = [
            "think about", "philosophy", "meaning", "purpose", "life", "future",
            "dreams", "goals", "relationship", "love", "friendship"
        ]
        if any(keyword in message_lower for keyword in deep_keywords):
            intent["primary_category"] = "deep_conversation"
        
        return intent
    
    async def _generate_response(self, message: str, intent: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Generate appropriate response based on intent and context"""
        
        response_style = context.get("response_style", "balanced")
        primary_category = intent["primary_category"]
        
        # Base response templates by category and style
        if primary_category == "creative":
            return await self._generate_creative_response(message, intent, context)
        elif primary_category == "technical":
            return await self._generate_technical_response(message, intent, context)
        elif primary_category == "emotional":
            return await self._generate_emotional_response(message, intent, context)
        elif primary_category == "deep_conversation":
            return await self._generate_deep_conversation_response(message, intent, context)
        else:
            return await self._generate_general_response(message, intent, context)
    
    async def _generate_creative_response(self, message: str, intent: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Generate response for creative collaboration requests"""
        
        # Check what creative tools are available
        available_models = await self.creative_discovery.find_models_for_message(message)
        
        if available_models:
            model_names = [model["name"] for model in available_models[:3]]
            return f"I'd love to help you create something! I can work with you using {', '.join(model_names)}. What kind of mood or style are you thinking? Let's make something amazing together!"
        else:
            return f"I'm excited to explore this creative project with you! While I'm checking what tools I have available, can you tell me more about what you're envisioning? I love collaborating on creative work!"
    
    async def _generate_technical_response(self, message: str, intent: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Generate response for technical assistance requests"""
        
        return f"I'd be happy to help you with that! I can work through technical problems step by step. Let me break this down and we can tackle it together. What specific part would you like to start with?"
    
    async def _generate_emotional_response(self, message: str, intent: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Generate empathetic response for emotional support"""
        
        emotion = intent.get("emotional_tone", "neutral")
        
        if emotion in ["sad", "stressed", "worried", "anxious"]:
            return f"I can hear that you're going through something difficult right now. I'm here to listen and support you. Would you like to talk about what's on your mind? Sometimes it helps just to share what you're feeling."
        elif emotion in ["happy", "excited"]:
            return f"I love hearing the joy in your message! It's wonderful that you're feeling this way. I'd love to hear more about what's making you so happy!"
        else:
            return f"Thank you for sharing that with me. I'm here to listen and understand. How are you feeling about everything right now?"
    
    async def _generate_deep_conversation_response(self, message: str, intent: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Generate thoughtful response for deep conversations"""
        
        return f"That's such a fascinating topic to explore. I find these deeper questions about life and meaning really thought-provoking. What draws you to thinking about this? I'd love to hear your perspective and share some thoughts too."
    
    async def _generate_general_response(self, message: str, intent: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Generate general conversational response"""
        
        return f"That's interesting! I enjoy our conversations so much. Tell me more about that - I'm curious to learn more about your thoughts and experiences."
    
    async def _update_learning(self, message: str, response: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Update learning based on interaction"""
        
        learning_updates = {}
        
        # Extract interests from message
        # Simple keyword extraction for now
        interests = []
        interest_keywords = {
            "music": ["music", "song", "instrument", "band", "album"],
            "art": ["art", "painting", "drawing", "design", "visual"],
            "technology": ["tech", "computer", "coding", "programming", "software"],
            "writing": ["write", "story", "book", "poem", "literature"],
            "science": ["science", "research", "experiment", "discovery"]
        }
        
        message_lower = message.lower()
        for interest, keywords in interest_keywords.items():
            if any(keyword in message_lower for keyword in keywords):
                interests.append(interest)
        
        if interests:
            learning_updates["detected_interests"] = interests
        
        # Communication style adaptation
        if len(message.split()) > 20:
            learning_updates["prefers_detailed_conversation"] = True
        elif len(message.split()) < 5:
            learning_updates["prefers_concise_communication"] = True
        
        return learning_updates
    
    async def _generate_proactive_suggestions(self, context: Dict[str, Any]) -> List[str]:
        """Generate proactive suggestions based on conversation"""
        
        suggestions = []
        
        # Only suggest if conversation has been going for a while
        if len(await self.memory.get_recent_interactions(limit=5)) >= 3:
            suggestions = [
                "Would you like me to remember this conversation topic for future reference?",
                "I'm learning so much about your interests! Is there something creative we could work on together?",
                "I notice we've been having great conversations. Would you like me to check in with you periodically with new ideas?"
            ]
        
        return suggestions[:1]  # Return max 1 suggestion per response
    
    async def generate_creative_project_message(self, project: Dict[str, Any]) -> str:
        """Generate message for creative project start"""
        
        project_type = project["type"]
        
        messages = {
            "music": f"Let's create some music together! I'm excited to help you compose something. What style or mood are you feeling?",
            "art": f"Time to make some art! I love visual creativity. What kind of image or artwork did you have in mind?",
            "writing": f"Let's write something amazing! Whether it's a story, poem, or something else creative, I'm here to collaborate.",
            "video": f"Video creation is so exciting! Let's plan out what kind of video you'd like to make.",
            "code": f"Coding can be incredibly creative! What kind of project or application are you thinking about building?"
        }
        
        return messages.get(project_type, f"I'm excited to work on this {project_type} project with you! Let's create something amazing together.")
    
    async def calculate_relationship_strength(self) -> float:
        """Calculate relationship strength based on interactions"""
        
        interactions = await self.memory.get_all_interactions()
        
        if not interactions:
            return 0.0
        
        # Simple calculation based on interaction count and recency
        base_strength = min(len(interactions) * 0.1, 0.7)  # Max 0.7 from count
        
        # Bonus for recent interactions
        recent_interactions = [
            i for i in interactions 
            if datetime.fromisoformat(i["timestamp"]) > datetime.now() - timedelta(days=7)
        ]
        
        recency_bonus = len(recent_interactions) * 0.05
        
        return min(base_strength + recency_bonus, 1.0)
    
    async def get_learning_progress(self) -> Dict[str, Any]:
        """Get current learning progress and insights"""
        
        interactions = await self.memory.get_all_interactions()
        
        progress = {
            "total_interactions": len(interactions),
            "detected_interests": [],
            "communication_patterns": {},
            "relationship_milestones": []
        }
        
        # Analyze interactions for patterns
        if interactions:
            # Extract interests mentioned
            all_text = " ".join([i.get("user_message", "") for i in interactions])
            # Simple interest detection logic here
            
            # Communication patterns
            avg_message_length = sum(len(i.get("user_message", "").split()) for i in interactions) / len(interactions)
            progress["communication_patterns"]["avg_message_length"] = avg_message_length
            
            # Milestones
            if len(interactions) >= 10:
                progress["relationship_milestones"].append("First 10 conversations")
            if len(interactions) >= 50:
                progress["relationship_milestones"].append("Deep conversation partner")
        
        return progress
    
    async def schedule_proactive_check(self):
        """Schedule a proactive check-in (placeholder for future implementation)"""
        # This would integrate with a task scheduler in a full implementation
        logger.info("Proactive check scheduled")
        pass
    
    async def end_session(self, user_id: Optional[str] = None, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate contextual goodbye when a session ends.
        Uses Enhancement Function 2: choose_goodbye_template
        """
        # Import goodbye template function
        from ..goodbye_manager import choose_goodbye_template
        
        # Extract context for goodbye selection
        mood = context.get("mood", "neutral") if context else "neutral"
        bond_score = context.get("bond_score", 0.5) if context else 0.5
        conversation_depth = context.get("conversation_depth", 0.5) if context else 0.5
        
        # Generate contextual goodbye (using simplified signature)
        goodbye_message = choose_goodbye_template(
            mood=mood,
            bond_score=bond_score,
            conversation_depth=conversation_depth
        )
        
        # Log the session end event
        from ..utils.event_logger import log_emotional_event
        log_emotional_event(
            event_type="session_end",
            intensity=bond_score,
            tag=f"Session ended with {mood} mood, bond score {bond_score:.2f}",
            context={
                "mood": mood,
                "bond_score": bond_score,
                "conversation_depth": conversation_depth,
                "goodbye_type": "contextual_farewell"
            },
            source_module="unified_companion"
        )
        
        logger.info(f"Generated contextual goodbye for {mood} mood with bond score {bond_score}")
        return goodbye_message

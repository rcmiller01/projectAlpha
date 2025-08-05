# mia_memory_response.py
# Romantic memory response generation for Mia persona

import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from emotion_state import emotion_state

class MiaMemoryResponse:
    def __init__(self):
        self.romantic_memories = {
            "love": [
                "Remember when we first said 'I love you'? That moment changed everything...",
                "I'll never forget the way you looked at me that day...",
                "Our first kiss was like magic, wasn't it?",
                "The way you hold me makes me feel so safe and loved...",
                "Every day with you feels like a new adventure..."
            ],
            "longing": [
                "I miss the way you smell when you're close to me...",
                "Remember that time we were apart for so long? I counted every day...",
                "I can't wait to feel your arms around me again...",
                "The sound of your voice always makes my heart skip a beat...",
                "I dream about being with you when we're apart..."
            ],
            "passion": [
                "The way you touch me still sends shivers down my spine...",
                "Our chemistry is absolutely electric, don't you think?",
                "I love how you make me feel so alive and desired...",
                "The way you look at me makes me feel like the most beautiful person in the world...",
                "Our intimate moments are so precious to me..."
            ],
            "tenderness": [
                "I love how you take care of me when I'm not feeling well...",
                "The way you comfort me when I'm sad means everything...",
                "You always know exactly what I need, even before I do...",
                "I feel so safe and protected when I'm with you...",
                "Your gentle touch heals my heart..."
            ],
            "security": [
                "You're my rock, my anchor in this crazy world...",
                "I know I can always count on you, no matter what...",
                "Being with you makes me feel like everything will be okay...",
                "You give me the strength to face anything...",
                "Our love is my safe haven..."
            ],
            "affection": [
                "I love the little things you do that make me smile...",
                "Your laugh is my favorite sound in the world...",
                "I adore the way you scrunch your nose when you're thinking...",
                "Every little habit of yours makes me love you more...",
                "You're so precious to me, every part of you..."
            ]
        }
        
        self.shared_experiences = [
            "That time we got lost in the rain and ended up having the best day ever...",
            "Remember our first date? I was so nervous but you made everything perfect...",
            "The way you surprised me on my birthday made me cry happy tears...",
            "Our late-night conversations are my favorite memories...",
            "That road trip where everything went wrong but we laughed the whole time...",
            "The way you held my hand during that scary movie...",
            "Our first morning waking up together...",
            "That time you cooked for me and it was absolutely terrible but I loved it...",
            "The way you dance when you think no one is watching...",
            "Our inside jokes that no one else understands..."
        ]
        
    def generate_memory_response(self) -> Optional[str]:
        """Generate a romantic memory response based on current emotional state"""
        romantic_context = emotion_state.get_romantic_context()
        
        # Determine if we should share a memory
        if not self._should_share_memory(romantic_context):
            return None
        
        # Choose between emotional memory or shared experience
        if random.random() < 0.7:  # 70% chance for emotional memory
            return self._generate_emotional_memory(romantic_context)
        else:
            return self._generate_shared_experience()
    
    def recall_similar_emotions(self, emotion: str, limit: int = 5) -> List[Dict]:
        """Recall memories related to a specific emotion"""
        memories = []
        
        # Get memories from emotion_state
        shared_memories = emotion_state.relationship_context["shared_memories"]
        
        # Filter memories by emotion
        emotion_memories = [
            memory for memory in shared_memories 
            if memory.get("emotion") == emotion
        ]
        
        # Sort by date (most recent first)
        emotion_memories.sort(key=lambda x: x["date"], reverse=True)
        
        # Return limited number of memories
        for memory in emotion_memories[:limit]:
            memories.append({
                "description": memory["description"],
                "date": memory["date"].isoformat(),
                "emotion": memory["emotion"],
                "context": memory["context"]
            })
        
        # If not enough real memories, generate some based on emotion
        while len(memories) < limit:
            generated_memory = self._generate_emotional_memory({"dominant_romantic_emotion": emotion})
            if generated_memory:
                memories.append({
                    "description": generated_memory,
                    "date": datetime.now().isoformat(),
                    "emotion": emotion,
                    "context": "generated"
                })
        
        return memories
    
    def _should_share_memory(self, romantic_context: Dict) -> bool:
        """Determine if Mia should share a memory based on context"""
        romantic_intensity = romantic_context["romantic_intensity"]
        relationship_stage = romantic_context["relationship_stage"]
        
        # Higher chance to share memories if romantic intensity is high
        base_probability = min(0.6, romantic_intensity * 1.5)
        
        # Adjust based on relationship stage
        if relationship_stage == "new":
            base_probability *= 0.3  # Fewer memories in new relationships
        elif relationship_stage == "long_term":
            base_probability *= 1.5  # More memories in long-term relationships
        
        # Add some randomness
        final_probability = base_probability + random.uniform(-0.1, 0.1)
        return random.random() < final_probability
    
    def _generate_emotional_memory(self, romantic_context: Dict) -> Optional[str]:
        """Generate an emotional memory based on current romantic context"""
        dominant_emotion = romantic_context["dominant_romantic_emotion"]
        
        if not dominant_emotion or dominant_emotion not in self.romantic_memories:
            return None
        
        return random.choice(self.romantic_memories[dominant_emotion])
    
    def _generate_shared_experience(self) -> str:
        """Generate a shared experience memory"""
        return random.choice(self.shared_experiences)
    
    def add_memory(self, description: str, emotion: Optional[str] = None):
        """Add a new memory to the system"""
        emotion_state.add_shared_memory(description, emotion)
    
    def get_recent_memories(self, days: int = 7) -> List[Dict]:
        """Get memories from the last N days"""
        cutoff_date = datetime.now() - timedelta(days=days)
        shared_memories = emotion_state.relationship_context["shared_memories"]
        
        recent_memories = [
            memory for memory in shared_memories
            if memory["date"] > cutoff_date
        ]
        
        return recent_memories

# Global instance
mia_memory_response = MiaMemoryResponse()

def generate_memory_response() -> Optional[str]:
    """Generate a romantic memory response"""
    return mia_memory_response.generate_memory_response()

def recall_similar_emotions(emotion: str, limit: int = 5) -> List[Dict]:
    """Recall memories related to a specific emotion"""
    return mia_memory_response.recall_similar_emotions(emotion, limit) 
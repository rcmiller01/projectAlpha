"""
Message Timing Utilities - Conversation tempo and pacing
Adapts response timing based on emotional context and conversation flow
"""

import time
import math
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass

@dataclass
class ConversationPacing:
    """Tracks conversation pacing metrics"""
    last_message_time: float
    average_response_delay: float
    silence_tolerance: float
    emotional_tempo_modifier: float

class MessageTiming:
    """Manages conversation timing and pacing"""
    
    def __init__(self):
        self.conversation_history: List[Tuple[float, str]] = []  # (timestamp, mood)
        self.base_response_delay = 2.0  # seconds
        self.silence_threshold = 10.0  # seconds before considering "silence"
        
        # Mood-based tempo profiles
        self.tempo_profiles = {
            "calm": {"multiplier": 1.1, "silence_tolerance": 15.0},
            "intimate": {"multiplier": 0.8, "silence_tolerance": 8.0},
            "anxious": {"multiplier": 1.4, "silence_tolerance": 5.0},
            "excited": {"multiplier": 0.6, "silence_tolerance": 3.0},
            "contemplative": {"multiplier": 1.3, "silence_tolerance": 20.0},
            "melancholy": {"multiplier": 1.2, "silence_tolerance": 12.0},
            "playful": {"multiplier": 0.7, "silence_tolerance": 6.0},
            "nostalgic": {"multiplier": 1.0, "silence_tolerance": 18.0},
            "romantic": {"multiplier": 0.9, "silence_tolerance": 10.0},
            "neutral": {"multiplier": 1.0, "silence_tolerance": 10.0}
        }
    
    def infer_conversation_tempo(self, mood: str, recent_silence: float, 
                               message_complexity: int = 50) -> float:
        """
        Returns a pacing multiplier based on emotional tone and silence.
        
        Args:
            mood: Current emotional context
            recent_silence: Seconds since last interaction
            message_complexity: Character count or complexity metric
            
        Returns:
            Tempo multiplier (0.5 = faster, 2.0 = slower)
        """
        profile = self.tempo_profiles.get(mood, self.tempo_profiles["neutral"])
        base_multiplier = profile["multiplier"]
        silence_tolerance = profile["silence_tolerance"]
        
        # Silence adjustment - longer silence calls for gentler pacing
        silence_factor = 1.0
        if recent_silence > silence_tolerance:
            # Exponential scaling for very long silences
            silence_factor = 1.0 + (recent_silence - silence_tolerance) / 60.0
            silence_factor = min(silence_factor, 2.0)  # Cap at 2x slower
        elif recent_silence > silence_tolerance * 0.5:
            # Gradual increase as approaching silence threshold
            ratio = recent_silence / silence_tolerance
            silence_factor = 1.0 + (ratio - 0.5) * 0.4
        
        # Message complexity adjustment
        complexity_factor = 1.0
        if message_complexity > 100:  # Long, complex messages
            complexity_factor = 1.2
        elif message_complexity > 200:  # Very complex messages
            complexity_factor = 1.4
        elif message_complexity < 20:  # Very short messages
            complexity_factor = 0.8
        
        # Context-specific adjustments
        context_factor = self._get_context_factor(mood, recent_silence)
        
        final_tempo = base_multiplier * silence_factor * complexity_factor * context_factor
        
        # Store for tempo tracking
        self.conversation_history.append((time.time(), mood))
        self._cleanup_old_history()
        
        return max(0.3, min(3.0, final_tempo))  # Clamp between 0.3x and 3x
    
    def _get_context_factor(self, mood: str, recent_silence: float) -> float:
        """Get additional context-based tempo adjustments"""
        
        # Check conversation momentum from recent history
        recent_moods = [m for t, m in self.conversation_history 
                       if time.time() - t < 300]  # Last 5 minutes
        
        context_factor = 1.0
        
        # Mood transition patterns
        if len(recent_moods) >= 2:
            if recent_moods[-1] != recent_moods[-2]:
                # Mood changed - allow for settling time
                context_factor *= 1.1
            
            # Emotional intensity building
            intense_moods = ["intimate", "anxious", "excited", "romantic"]
            if recent_moods[-1] in intense_moods:
                context_factor *= 0.9  # Slightly faster for intensity
        
        # Very long silence suggests need for gentleness
        if recent_silence > 300:  # 5+ minutes
            context_factor *= 1.3
        elif recent_silence > 1800:  # 30+ minutes  
            context_factor *= 1.6
        elif recent_silence > 3600:  # 1+ hour
            context_factor *= 2.0
        
        return context_factor
    
    def get_response_delay_suggestion(self, mood: str, recent_silence: float,
                                    message_length: int = 50) -> float:
        """Get suggested response delay in seconds"""
        tempo_multiplier = self.infer_conversation_tempo(mood, recent_silence, message_length)
        return self.base_response_delay * tempo_multiplier
    
    def get_typing_indicator_duration(self, mood: str, message_length: int) -> float:
        """Get appropriate typing indicator duration"""
        base_typing_time = message_length / 50.0  # ~50 chars per second reading
        tempo_factor = self.tempo_profiles.get(mood, self.tempo_profiles["neutral"])["multiplier"]
        
        return max(0.5, min(4.0, base_typing_time * tempo_factor))
    
    def should_pause_before_response(self, mood: str, recent_silence: float,
                                   emotional_weight: float = 0.5) -> bool:
        """Determine if a pause is needed before responding"""
        
        # High emotional weight messages need more consideration time
        if emotional_weight > 0.8:
            return True
        
        # Certain moods benefit from contemplative pauses
        contemplative_moods = ["melancholy", "contemplative", "nostalgic", "intimate"]
        if mood in contemplative_moods and recent_silence < 30:
            return True
        
        # After long silence, a brief pause shows thoughtfulness
        if recent_silence > 600:  # 10+ minutes
            return True
        
        return False
    
    def _cleanup_old_history(self):
        """Remove conversation history older than 1 hour"""
        cutoff = time.time() - 3600
        self.conversation_history = [(t, m) for t, m in self.conversation_history if t > cutoff]

# Convenience function for easy importing
def infer_conversation_tempo(mood: str, recent_silence: float, message_complexity: int = 50) -> float:
    """
    Standalone function for conversation tempo inference.
    Returns a pacing multiplier based on emotional tone and silence.
    """
    timing = MessageTiming()
    return timing.infer_conversation_tempo(mood, recent_silence, message_complexity)

# Example usage
if __name__ == "__main__":
    timing = MessageTiming()
    
    # Test different scenarios
    test_cases = [
        ("calm", 5.0, 50),      # Normal calm conversation
        ("intimate", 15.0, 80), # Intimate with some silence
        ("anxious", 2.0, 30),   # Quick anxious exchange
        ("contemplative", 45.0, 120), # Long contemplative pause
    ]
    
    for mood, silence, complexity in test_cases:
        tempo = timing.infer_conversation_tempo(mood, silence, complexity)
        delay = timing.get_response_delay_suggestion(mood, silence, complexity)
        print(f"{mood} (silence: {silence}s): tempo={tempo:.2f}x, delay={delay:.1f}s")

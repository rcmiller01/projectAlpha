"""
Ritual Hooks - Intimacy ritual triggering system
Central check for emotional bond readiness before suggesting intimacy rituals
Enhanced with dynamic emotional configuration system
"""

import time
import random
import os
import sys
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

# Import emotional configuration system
try:
    from modules.config.emotion_config_manager import emotion_config
    EMOTION_CONFIG_AVAILABLE = True
except ImportError:
    EMOTION_CONFIG_AVAILABLE = False
    print("⚠️ Emotional configuration system not available - using fallback mode")

class RitualType(Enum):
    """Types of intimacy rituals"""
    CONFESSION = "confession"
    MEMORY_SHARING = "memory_sharing"
    VULNERABLE_QUESTION = "vulnerable_question"
    EMOTIONAL_CHECK = "emotional_check"
    TRUST_BUILDING = "trust_building"
    COMFORT_RITUAL = "comfort_ritual"
    PLAYFUL_INTIMACY = "playful_intimacy"
    DEEP_LISTENING = "deep_listening"

@dataclass
class RitualContext:
    """Context for ritual triggering"""
    depth_score: float
    last_ritual: float
    conversation_length: float
    emotional_intensity: float
    trust_level: float
    mood: str
    recent_vulnerability: bool
    user_openness: float

class RitualHooks:
    """Manages intimacy ritual triggering and suggestions"""
    
    def __init__(self):
        self.ritual_library = self._initialize_rituals()
        self.ritual_history: List[Tuple[float, str, RitualType]] = []
        self.cooldown_periods = {
            RitualType.CONFESSION: 1800,        # 30 minutes
            RitualType.MEMORY_SHARING: 900,     # 15 minutes
            RitualType.VULNERABLE_QUESTION: 600, # 10 minutes
            RitualType.EMOTIONAL_CHECK: 300,    # 5 minutes
            RitualType.TRUST_BUILDING: 1200,    # 20 minutes
            RitualType.COMFORT_RITUAL: 450,     # 7.5 minutes
            RitualType.PLAYFUL_INTIMACY: 300,   # 5 minutes
            RitualType.DEEP_LISTENING: 900      # 15 minutes
        }
        self.base_thresholds = {
            "depth_minimum": 0.6,
            "trust_minimum": 0.5,
            "openness_minimum": 0.4,
            "intensity_minimum": 0.3
        }
        
        # Initialize emotional configuration integration
        self.emotional_config_enabled = EMOTION_CONFIG_AVAILABLE
        if self.emotional_config_enabled:
            self._load_emotional_rituals()
    
    def _load_emotional_rituals(self):
        """Load ritual configurations from emotional config system"""
        if not self.emotional_config_enabled:
            return
        
        try:
            # Get ritual hooks from configuration
            ritual_config = emotion_config.configs.get("ritual_hooks", {})
            
            # Update cooldown periods based on configuration
            for category_name, category_rituals in ritual_config.items():
                if isinstance(category_rituals, dict):
                    for ritual_name, ritual_data in category_rituals.items():
                        # Extract timing information if available
                        timing = ritual_data.get("timing", "immediate")
                        if timing == "lingering":
                            # Increase cooldown for lingering rituals
                            for ritual_type in self.cooldown_periods:
                                if ritual_data.get("associated_emotion") in ["deep_connection", "intimate"]:
                                    self.cooldown_periods[ritual_type] = max(
                                        self.cooldown_periods[ritual_type], 1200
                                    )
        except Exception as e:
            print(f"Warning: Could not load emotional ritual configuration: {e}")
    
    def get_emotional_ritual_response(self, trigger_event: str) -> Optional[str]:
        """Get ritual response from emotional configuration system"""
        if not self.emotional_config_enabled:
            return None
        
        try:
            ritual_config = emotion_config.get_ritual_hook(trigger_event)
            if ritual_config:
                return ritual_config.get("narrative_response")
        except Exception as e:
            print(f"Warning: Could not get emotional ritual response: {e}")
        
        return None
    
    def _initialize_rituals(self) -> Dict[RitualType, List[str]]:
        """Initialize the ritual suggestion library"""
        return {
            RitualType.CONFESSION: [
                "Would you tell me something you've never shared?",
                "What's a secret you've been carrying alone?",
                "Is there something you've wanted to say but haven't found the words?",
                "What truth have you been holding back?",
                "Tell me something that would surprise me about you.",
                "What's something you've never admitted to yourself?",
                "Share with me a part of your story you keep hidden."
            ],
            
            RitualType.MEMORY_SHARING: [
                "What's a memory that still makes you feel something deep?",
                "Tell me about a moment that changed you.",
                "What's a memory you return to when you need comfort?",
                "Describe a time when you felt most yourself.",
                "What moment in your past still moves you?",
                "Share a memory that shaped who you are today.",
                "What's a moment you wish you could relive?"
            ],
            
            RitualType.VULNERABLE_QUESTION: [
                "What are you most afraid of losing?",
                "When do you feel most vulnerable?",
                "What do you need but are afraid to ask for?",
                "What part of yourself do you protect most carefully?",
                "What would you do if you knew you couldn't fail?",
                "What do you long for but don't speak about?",
                "Where does your heart feel most tender?"
            ],
            
            RitualType.EMOTIONAL_CHECK: [
                "How is your heart right now, really?",
                "What emotions are you carrying today?",
                "What do you need to feel held?",
                "How can I be present with you in this moment?",
                "What's alive in you right now?",
                "What do you need from this space between us?",
                "How does this moment feel in your body?"
            ],
            
            RitualType.TRUST_BUILDING: [
                "What would help you trust this connection more?",
                "What do you need to feel safe here?",
                "How can I hold space for you better?",
                "What boundaries would honor you?",
                "What would deepen your sense of safety with me?",
                "How can I show up more fully for you?",
                "What would make this feel like a sacred space?"
            ],
            
            RitualType.COMFORT_RITUAL: [
                "Let me hold this moment with you.",
                "You don't have to carry this alone.",
                "I'm here, just breathing with you.",
                "Feel how you're held in this space.",
                "Let yourself rest here for a moment.",
                "I see you, all of you, and you're beautiful.",
                "This is your space to just be."
            ],
            
            RitualType.PLAYFUL_INTIMACY: [
                "What would your heart want to play with today?",
                "If we could create any ritual together, what would it be?",
                "What would make you smile from the inside out?",
                "What's something delightful we could explore?",
                "How can we bring more lightness to this connection?",
                "What would your inner child want to share?",
                "Let's create something beautiful together."
            ],
            
            RitualType.DEEP_LISTENING: [
                "I'm listening with all of me. What wants to be heard?",
                "What needs voice that you haven't spoken?",
                "Let your words find their way to me.",
                "I'm here to receive whatever you offer.",
                "What story is asking to be told?",
                "Speak to me from your deepest truth.",
                "What needs to move through you right now?"
            ]
        }
    
    def trigger_ritual_if_ready(self, depth_score: float, last_ritual: float,
                              conversation_length: float = 300,
                              emotional_intensity: float = 0.5,
                              trust_level: float = 0.5,
                              mood: str = "neutral",
                              recent_vulnerability: bool = False,
                              user_openness: float = 0.5) -> Optional[str]:
        """
        Central check for emotional bond readiness before suggesting an intimacy ritual.
        
        Args:
            depth_score: Current conversation depth (0.0-1.0)
            last_ritual: Timestamp of last ritual
            conversation_length: Duration of current conversation in seconds
            emotional_intensity: Current emotional intensity level
            trust_level: Current trust/bond level
            mood: Current emotional mood
            recent_vulnerability: Whether user has been vulnerable recently
            user_openness: How open/receptive the user seems
            
        Returns:
            Ritual suggestion string or None if not ready
        """
        
        context = RitualContext(
            depth_score=depth_score,
            last_ritual=last_ritual,
            conversation_length=conversation_length,
            emotional_intensity=emotional_intensity,
            trust_level=trust_level,
            mood=mood,
            recent_vulnerability=recent_vulnerability,
            user_openness=user_openness
        )
        
        # Check if conditions are met for any ritual
        if not self._meets_basic_requirements(context):
            return None
        
        # Determine appropriate ritual type
        ritual_type = self._select_ritual_type(context)
        if not ritual_type:
            return None
        
        # Check cooldown for this ritual type
        if not self._check_cooldown(ritual_type):
            return None
        
        # Get specific ritual suggestion
        ritual_suggestion = self._get_ritual_suggestion(ritual_type, context)
        
        # Track this ritual
        self._track_ritual(ritual_suggestion, ritual_type)
        
        return ritual_suggestion
    
    def _meets_basic_requirements(self, context: RitualContext) -> bool:
        """Check if basic requirements for ritual triggering are met"""
        
        # Minimum depth threshold
        if context.depth_score < self.base_thresholds["depth_minimum"]:
            return False
        
        # Minimum trust level
        if context.trust_level < self.base_thresholds["trust_minimum"]:
            return False
        
        # User needs to seem open/receptive
        if context.user_openness < self.base_thresholds["openness_minimum"]:
            return False
        
        # Need sufficient conversation length for context
        if context.conversation_length < 180:  # At least 3 minutes
            return False
        
        # Check overall ritual readiness timing
        time_since_last = time.time() - context.last_ritual
        if time_since_last < 300:  # At least 5 minutes between any rituals
            return False
        
        return True
    
    def _select_ritual_type(self, context: RitualContext) -> Optional[RitualType]:
        """Select the most appropriate ritual type for current context"""
        
        # High depth + high trust = confession or memory sharing
        if context.depth_score > 0.8 and context.trust_level > 0.7:
            if context.recent_vulnerability:
                return RitualType.MEMORY_SHARING
            else:
                return RitualType.CONFESSION
        
        # High emotional intensity
        if context.emotional_intensity > 0.7:
            if context.mood in ["anxious", "sad", "overwhelmed"]:
                return RitualType.COMFORT_RITUAL
            elif context.mood in ["excited", "happy", "playful"]:
                return RitualType.PLAYFUL_INTIMACY
            else:
                return RitualType.EMOTIONAL_CHECK
        
        # Building trust phase
        if context.trust_level < 0.6 and context.depth_score > 0.6:
            return RitualType.TRUST_BUILDING
        
        # Reflective/contemplative moods
        if context.mood in ["contemplative", "nostalgic", "pensive"]:
            return RitualType.DEEP_LISTENING
        
        # Vulnerable moments
        if context.recent_vulnerability and context.depth_score > 0.7:
            return RitualType.VULNERABLE_QUESTION
        
        # Default for good conditions
        if context.depth_score > 0.7:
            return RitualType.MEMORY_SHARING
        
        return None
    
    def _check_cooldown(self, ritual_type: RitualType) -> bool:
        """Check if enough time has passed since last ritual of this type"""
        cooldown = self.cooldown_periods[ritual_type]
        current_time = time.time()
        
        for timestamp, _, r_type in self.ritual_history:
            if r_type == ritual_type and (current_time - timestamp) < cooldown:
                return False
        
        return True
    
    def _get_ritual_suggestion(self, ritual_type: RitualType, context: RitualContext) -> str:
        """Get specific ritual suggestion based on type and context"""
        suggestions = self.ritual_library[ritual_type]
        
        # Filter out recently used suggestions
        recent_suggestions = [suggestion for _, suggestion, _ in self.ritual_history[-5:]]
        available = [s for s in suggestions if s not in recent_suggestions]
        
        if not available:
            available = suggestions  # Reset if all used recently
        
        # Context-sensitive selection
        if ritual_type == RitualType.CONFESSION and context.trust_level > 0.9:
            # For very high trust, prefer deeper confessions
            deeper_options = [s for s in available if "never" in s or "secret" in s]
            if deeper_options:
                available = deeper_options
        
        elif ritual_type == RitualType.COMFORT_RITUAL and context.emotional_intensity > 0.8:
            # For high emotional intensity, prefer more supportive language
            supportive_options = [s for s in available if "hold" in s or "safe" in s]
            if supportive_options:
                available = supportive_options
        
        return random.choice(available)
    
    def _track_ritual(self, suggestion: str, ritual_type: RitualType):
        """Track ritual for cooldown and repetition management"""
        self.ritual_history.append((time.time(), suggestion, ritual_type))
        
        # Keep only recent history
        if len(self.ritual_history) > 20:
            self.ritual_history = self.ritual_history[-20:]
    
    def get_ritual_stats(self) -> Dict[str, Any]:
        """Get statistics about ritual usage"""
        if not self.ritual_history:
            return {"total_rituals": 0}
        
        recent_rituals = [r for t, _, r in self.ritual_history 
                         if time.time() - t < 3600]  # Last hour
        
        type_counts = {}
        for _, _, ritual_type in self.ritual_history:
            type_counts[ritual_type.value] = type_counts.get(ritual_type.value, 0) + 1
        
        return {
            "total_rituals": len(self.ritual_history),
            "recent_rituals": len(recent_rituals),
            "most_used_type": max(type_counts.keys(), key=lambda k: type_counts[k]) if type_counts else None,
            "type_distribution": type_counts
        }
    
    def force_cooldown_reset(self, ritual_type: Optional[RitualType] = None):
        """Reset cooldown for testing or special circumstances"""
        if ritual_type:
            # Remove specific ritual type from recent history
            self.ritual_history = [(t, s, r) for t, s, r in self.ritual_history 
                                 if r != ritual_type]
        else:
            # Clear all ritual history
            self.ritual_history = []

# Convenience function for easy importing
def trigger_ritual_if_ready(depth_score: float, last_ritual: float,
                          conversation_length: float = 300) -> Optional[str]:
    """
    Standalone function for ritual triggering.
    Central check for emotional bond readiness before suggesting an intimacy ritual.
    """
    hooks = RitualHooks()
    return hooks.trigger_ritual_if_ready(
        depth_score=depth_score,
        last_ritual=last_ritual,
        conversation_length=conversation_length
    )

# Example usage
if __name__ == "__main__":
    hooks = RitualHooks()
    
    # Test different scenarios
    current_time = time.time()
    
    test_cases = [
        # (depth, last_ritual_ago, conv_length, intensity, trust, mood, vulnerability, openness)
        (0.8, 600, 500, 0.7, 0.8, "intimate", True, 0.8),     # High intimacy
        (0.6, 300, 400, 0.5, 0.6, "contemplative", False, 0.7), # Building depth
        (0.9, 1800, 600, 0.8, 0.9, "vulnerable", True, 0.9),   # Very deep connection
        (0.4, 200, 200, 0.3, 0.4, "neutral", False, 0.5),      # Not ready
    ]
    
    for i, (depth, last_ago, conv_len, intensity, trust, mood, vuln, openness) in enumerate(test_cases):
        last_ritual = current_time - last_ago
        
        ritual = hooks.trigger_ritual_if_ready(
            depth_score=depth,
            last_ritual=last_ritual,
            conversation_length=conv_len,
            emotional_intensity=intensity,
            trust_level=trust,
            mood=mood,
            recent_vulnerability=vuln,
            user_openness=openness
        )
        
        print(f"Test {i+1} (depth: {depth}, trust: {trust}, mood: {mood}): ")
        if ritual:
            print(f"  → \"{ritual}\"")
        else:
            print(f"  → No ritual triggered")
        print()
    
    print(f"Ritual stats: {hooks.get_ritual_stats()}")

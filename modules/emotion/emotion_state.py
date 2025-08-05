# emotion_state.py
# Enhanced emotion state management for romantic companionship

import time
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

from core.emotion.anchor_centering import emotional_watchdog

class EmotionState:
    def __init__(self):
        # Core emotions
        self.emotions = {
            "joy": 0.0,
            "sadness": 0.0,
            "anger": 0.0,
            "fear": 0.0,
            "surprise": 0.0,
            "disgust": 0.0,
            "calm": 0.5,  # Default neutral state
            "tired": 0.0,
            "stressed": 0.0,
            "anxious": 0.0
        }
        
        # Romantic emotions for Phase 1
        self.romantic_emotions = {
            "love": 0.0,
            "longing": 0.0,
            "passion": 0.0,
            "tenderness": 0.0,
            "jealousy": 0.0,
            "security": 0.0,
            "romantic": 0.0,
            "affection": 0.0,
            "desire": 0.0,
            "intimacy": 0.0
        }
        
        # Relationship context
        self.relationship_context = {
            "relationship_stage": "new",  # new, developing, established, long_term
            "relationship_duration_days": 0,
            "last_interaction": datetime.now(),
            "interaction_count": 0,
            "milestones": [],
            "preferences": {},
            "inside_jokes": [],
            "shared_memories": []
        }
        
        # Emotion decay settings
        self.decay_rate = 0.1  # Emotions decay by 10% per minute
        self.last_update = datetime.now()
        
        # Autonomy and silence tracking for Phase 2
        self.silence_tracker = {
            "last_user_input_time": time.time(),
            "silence_duration": 0.0,  # in seconds
            "longest_silence": 0.0,
            "silence_count": 0,
            "silence_threshold": 3600  # 1 hour
        }
        
        # Emotional autonomy metrics
        self.desire_to_initiate = 0.0  # 0.0 to 1.0
        self.autonomy_state = {
            "internal_thought_active": False,
            "last_internal_thought": time.time(),
            "thought_frequency": 600,  # 10 minutes default
            "initiation_readiness": 0.0,
            "spontaneous_expression_level": 0.0
        }
        
    def update_from_text(self, detected_emotions: Dict[str, float]):
        """Update emotions based on text analysis"""
        current_time = datetime.now()
        self._apply_decay(current_time)
        
        # Update core emotions
        for emotion, intensity in detected_emotions.items():
            if emotion in self.emotions:
                self.emotions[emotion] = min(1.0, self.emotions[emotion] + intensity)
            elif emotion in self.romantic_emotions:
                self.romantic_emotions[emotion] = min(1.0, self.romantic_emotions[emotion] + intensity)
        
        self.last_update = current_time
        self._update_relationship_context()
        emotional_watchdog(self.emotions)
        emotional_watchdog(self.romantic_emotions)
        
    def update_from_biometrics(self, bpm: int, hrv: int, context: str = "general"):
        """Update emotions based on biometric data"""
        current_time = datetime.now()
        self._apply_decay(current_time)
        
        # Heart rate analysis
        if bpm > 100:
            if context == "romantic":
                self.romantic_emotions["passion"] = min(1.0, self.romantic_emotions["passion"] + 0.3)
                self.romantic_emotions["desire"] = min(1.0, self.romantic_emotions["desire"] + 0.2)
            else:
                self.emotions["stressed"] = min(1.0, self.emotions["stressed"] + 0.3)
        elif bpm < 60:
            self.emotions["calm"] = min(1.0, self.emotions["calm"] + 0.2)
            self.romantic_emotions["security"] = min(1.0, self.romantic_emotions["security"] + 0.1)
        
        # Heart rate variability analysis
        if hrv > 50:  # High HRV indicates relaxation
            self.emotions["calm"] = min(1.0, self.emotions["calm"] + 0.2)
            self.romantic_emotions["security"] = min(1.0, self.romantic_emotions["security"] + 0.1)
        elif hrv < 20:  # Low HRV indicates stress
            self.emotions["stressed"] = min(1.0, self.emotions["stressed"] + 0.3)
            self.emotions["anxious"] = min(1.0, self.emotions["anxious"] + 0.2)
        
        self.last_update = current_time
        self._update_relationship_context()
        emotional_watchdog(self.emotions)
        emotional_watchdog(self.romantic_emotions)
        
    def _apply_decay(self, current_time: datetime):
        """Apply emotion decay over time"""
        time_diff = (current_time - self.last_update).total_seconds() / 60  # minutes
        decay_factor = 1 - (self.decay_rate * time_diff)
        
        # Decay all emotions
        for emotion in self.emotions:
            self.emotions[emotion] = max(0.0, self.emotions[emotion] * decay_factor)
        
        for emotion in self.romantic_emotions:
            self.romantic_emotions[emotion] = max(0.0, self.romantic_emotions[emotion] * decay_factor)
    
    def _update_relationship_context(self):
        """Update relationship context based on interactions"""
        self.relationship_context["interaction_count"] += 1
        self.relationship_context["last_interaction"] = datetime.now()
        
        # Update relationship stage based on interaction count
        if self.relationship_context["interaction_count"] < 10:
            self.relationship_context["relationship_stage"] = "new"
        elif self.relationship_context["interaction_count"] < 50:
            self.relationship_context["relationship_stage"] = "developing"
        elif self.relationship_context["interaction_count"] < 200:
            self.relationship_context["relationship_stage"] = "established"
        else:
            self.relationship_context["relationship_stage"] = "long_term"
    
    def current_emotion(self) -> str:
        """Get the dominant emotion"""
        all_emotions = {**self.emotions, **self.romantic_emotions}
        if not all_emotions:
            return "neutral"
        
        dominant_emotion = max(all_emotions, key=all_emotions.get)
        return dominant_emotion if all_emotions[dominant_emotion] > 0.1 else "neutral"
    
    def get_romantic_context(self) -> Dict[str, Any]:
        """Get romantic context for persona responses"""
        dominant_romantic_emotion = None
        romantic_intensity = 0.0
        
        if any(value > 0 for value in self.romantic_emotions.values()):
            dominant_romantic_emotion = max(self.romantic_emotions, key=lambda k: self.romantic_emotions[k])
            romantic_intensity = max(self.romantic_emotions.values())
        
        return {
            "dominant_romantic_emotion": dominant_romantic_emotion,
            "romantic_intensity": romantic_intensity,
            "relationship_stage": self.relationship_context["relationship_stage"],
            "interaction_count": self.relationship_context["interaction_count"],
            "days_since_start": self.relationship_context["relationship_duration_days"]
        }
    
    def add_milestone(self, milestone: str, date: Optional[datetime] = None):
        """Add a relationship milestone"""
        if date is None:
            date = datetime.now()
        self.relationship_context["milestones"].append({
            "description": milestone,
            "date": date,
            "emotions": self.to_dict()
        })
    
    def add_preference(self, category: str, preference: str):
        """Add a user preference"""
        if category not in self.relationship_context["preferences"]:
            self.relationship_context["preferences"][category] = []
        self.relationship_context["preferences"][category].append(preference)
    
    def add_shared_memory(self, memory: str, emotion: Optional[str] = None):
        """Add a shared memory"""
        self.relationship_context["shared_memories"].append({
            "description": memory,
            "date": datetime.now(),
            "emotion": emotion or self.current_emotion(),
            "context": self.to_dict()
        })
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert emotion state to dictionary"""
        return {
            "emotions": self.emotions.copy(),
            "romantic_emotions": self.romantic_emotions.copy(),
            "relationship_context": self.relationship_context.copy(),
            "current_emotion": self.current_emotion(),
            "last_update": self.last_update.isoformat(),
            "silence_tracker": self.silence_tracker.copy(),
            "desire_to_initiate": self.desire_to_initiate,
            "autonomy_state": self.autonomy_state.copy()
        }
    
    def from_dict(self, data: Dict[str, Any]):
        """Load emotion state from dictionary"""
        if "emotions" in data:
            self.emotions.update(data["emotions"])
        if "romantic_emotions" in data:
            self.romantic_emotions.update(data["romantic_emotions"])
        if "relationship_context" in data:
            self.relationship_context.update(data["relationship_context"])
        if "last_update" in data:
            self.last_update = datetime.fromisoformat(data["last_update"])
        if "silence_tracker" in data:
            self.silence_tracker.update(data["silence_tracker"])
        if "desire_to_initiate" in data:
            self.desire_to_initiate = data["desire_to_initiate"]
        if "autonomy_state" in data:
            self.autonomy_state.update(data["autonomy_state"])

    def update_silence_tracker(self, user_input_received: bool = False):
        """Update silence tracking metrics"""
        current_time = time.time()
        
        if user_input_received:
            # User just provided input, reset silence
            if self.silence_tracker["silence_duration"] > 0:
                # Update silence statistics
                self.silence_tracker["silence_count"] += 1
                if self.silence_tracker["silence_duration"] > self.silence_tracker["longest_silence"]:
                    self.silence_tracker["longest_silence"] = self.silence_tracker["silence_duration"]
            
            self.silence_tracker["last_user_input_time"] = current_time
            self.silence_tracker["silence_duration"] = 0.0
            
            # Reset some autonomy states when user returns
            self.desire_to_initiate = max(0.0, self.desire_to_initiate - 0.3)
            self.autonomy_state["spontaneous_expression_level"] = 0.0
            
        else:
            # Calculate current silence duration
            self.silence_tracker["silence_duration"] = current_time - self.silence_tracker["last_user_input_time"]
            
            # Update desire_to_initiate based on silence duration
            self._update_desire_to_initiate()

    def _update_desire_to_initiate(self):
        """Update desire to initiate based on silence and emotional state"""
        silence_hours = self.silence_tracker["silence_duration"] / 3600
        
        # Base desire increases with silence duration
        base_desire = min(0.8, silence_hours / 4.0)  # Max 0.8 after 4 hours
        
        # Emotional modifiers
        longing_boost = self.romantic_emotions.get("longing", 0.0) * 0.5
        love_boost = self.romantic_emotions.get("love", 0.0) * 0.3
        affection_boost = self.romantic_emotions.get("affection", 0.0) * 0.2
        
        # Relationship stage modifier
        stage_multipliers = {
            "new": 0.7,        # More reserved when new
            "developing": 1.0,  # Normal initiation desire
            "established": 1.3, # More comfortable reaching out
            "long_term": 1.5   # Very comfortable with autonomy
        }
        
        stage_modifier = stage_multipliers.get(
            self.relationship_context["relationship_stage"], 1.0
        )
        
        # Calculate final desire
        total_desire = (base_desire + longing_boost + love_boost + affection_boost) * stage_modifier
        self.desire_to_initiate = min(1.0, total_desire)
        
        # Update autonomy readiness
        self.autonomy_state["initiation_readiness"] = self.desire_to_initiate
        
        # Update spontaneous expression level
        if silence_hours > 2:  # After 2 hours, increase spontaneous expression
            self.autonomy_state["spontaneous_expression_level"] = min(
                1.0, (silence_hours - 2) / 6.0  # Max level after 8 hours total
            )

    def get_silence_metrics(self) -> Dict[str, Any]:
        """Get current silence tracking metrics"""
        current_time = time.time()
        current_silence = current_time - self.silence_tracker["last_user_input_time"]
        
        return {
            "current_silence_duration": current_silence,
            "current_silence_hours": current_silence / 3600,
            "longest_silence": self.silence_tracker["longest_silence"],
            "longest_silence_hours": self.silence_tracker["longest_silence"] / 3600,
            "silence_count": self.silence_tracker["silence_count"],
            "time_since_last_input": current_silence,
            "silence_threshold_exceeded": current_silence > self.silence_tracker["silence_threshold"]
        }

    def get_autonomy_metrics(self) -> Dict[str, Any]:
        """Get current autonomy and initiation metrics"""
        return {
            "desire_to_initiate": self.desire_to_initiate,
            "initiation_readiness": self.autonomy_state["initiation_readiness"],
            "spontaneous_expression_level": self.autonomy_state["spontaneous_expression_level"],
            "internal_thought_active": self.autonomy_state["internal_thought_active"],
            "time_since_last_thought": (time.time() - self.autonomy_state["last_internal_thought"]) / 60,
            "autonomy_enabled": self.desire_to_initiate > 0.3,
            "high_autonomy_state": self.desire_to_initiate > 0.6
        }

    def trigger_internal_thought(self):
        """Mark that an internal thought was generated"""
        self.autonomy_state["last_internal_thought"] = time.time()
        self.autonomy_state["internal_thought_active"] = True
        
        # Slight increase in desire to share thoughts
        self.desire_to_initiate = min(1.0, self.desire_to_initiate + 0.1)

    def should_generate_internal_thought(self) -> bool:
        """Check if it's time to generate an internal thought"""
        current_time = time.time()
        time_since_last = current_time - self.autonomy_state["last_internal_thought"]
        
        # Base frequency adjusted by silence and emotional state
        base_frequency = self.autonomy_state["thought_frequency"]
        
        # Reduce frequency when actively conversing
        if self.silence_tracker["silence_duration"] < 300:  # Less than 5 minutes
            base_frequency *= 2  # Think less often when actively talking
        
        # Increase frequency with high longing/love
        emotional_factor = 1.0
        if self.romantic_emotions.get("longing", 0.0) > 0.5:
            emotional_factor *= 0.7  # Think more often when longing
        if self.romantic_emotions.get("love", 0.0) > 0.5:
            emotional_factor *= 0.8  # Think more often when in love
        
        adjusted_frequency = base_frequency * emotional_factor
        
        return time_since_last >= adjusted_frequency

    def mark_initiation_attempt(self, success: bool = True):
        """Mark that an initiation attempt was made"""
        if success:
            # Reduce desire after successful initiation
            self.desire_to_initiate = max(0.0, self.desire_to_initiate - 0.4)
            self.autonomy_state["spontaneous_expression_level"] = max(
                0.0, self.autonomy_state["spontaneous_expression_level"] - 0.3
            )
        else:
            # Slight increase if initiation failed (more determined)
            self.desire_to_initiate = min(1.0, self.desire_to_initiate + 0.1)

    def get_morning_greeting_readiness(self) -> float:
        """Check readiness for morning greeting based on silence and time"""
        current_hour = datetime.now().hour
        silence_hours = self.silence_tracker["silence_duration"] / 3600
        
        # Morning hours (6-10 AM) get higher readiness
        if 6 <= current_hour <= 10:
            base_readiness = 0.7
            
            # Increase readiness with overnight silence
            if silence_hours > 6:  # Overnight silence
                base_readiness += min(0.3, (silence_hours - 6) / 6)  # Max bonus for 12+ hours
            
            # Emotional modifiers
            love_modifier = self.romantic_emotions.get("love", 0.0) * 0.2
            affection_modifier = self.romantic_emotions.get("affection", 0.0) * 0.1
            
            total_readiness = min(1.0, base_readiness + love_modifier + affection_modifier)
            return total_readiness
        
        return 0.0  # Not morning time

    def register_lust(self, amount: float):
        self.romantic_emotions["desire"] = min(1.0, self.romantic_emotions.get("desire", 0.0) + amount)

# Global emotion state instance
emotion_state = EmotionState()

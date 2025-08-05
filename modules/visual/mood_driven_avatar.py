# mood_driven_avatar.py
# Visual mood-driven avatar system with romantic expressions

import json
import time
import threading
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import numpy as np
import math

class AvatarExpression(Enum):
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    SURPRISED = "surprised"
    FEARFUL = "fearful"
    DISGUSTED = "disgusted"
    NEUTRAL = "neutral"
    LOVE = "love"
    PASSION = "passion"
    TENDERNESS = "tenderness"
    LONGING = "longing"
    SECURITY = "security"
    EXCITEMENT = "excitement"
    CALM = "calm"
    PLAYFUL = "playful"
    SEDUCTIVE = "seductive"
    VULNERABLE = "vulnerable"
    CONFIDENT = "confident"

class AvatarGesture(Enum):
    WAVE = "wave"
    BLOW_KISS = "blow_kiss"
    HEART_HANDS = "heart_hands"
    HUG = "hug"
    TOUCH_HEART = "touch_heart"
    WINK = "wink"
    SMILE = "smile"
    TILT_HEAD = "tilt_head"
    REACH_OUT = "reach_out"
    DANCE = "dance"
    EMBRACE = "embrace"
    STROKE_HAIR = "stroke_hair"
    HOLD_HANDS = "hold_hands"
    KISS = "kiss"
    GAZE = "gaze"
    WHISPER = "whisper"

@dataclass
class AvatarState:
    expression: AvatarExpression
    gesture: Optional[AvatarGesture]
    eye_color: str
    hair_style: str
    clothing: str
    background: str
    lighting: str
    animation_intensity: float  # 0.0 to 1.0
    emotional_intensity: float  # 0.0 to 1.0
    timestamp: datetime

@dataclass
class ExpressionParameters:
    mouth_curve: float  # -1.0 to 1.0
    eye_openness: float  # 0.0 to 1.0
    eyebrow_position: float  # -1.0 to 1.0
    cheek_redness: float  # 0.0 to 1.0
    eye_sparkle: float  # 0.0 to 1.0
    overall_brightness: float  # 0.5 to 1.5

class MoodDrivenAvatar:
    def __init__(self):
        self.current_state = AvatarState(
            expression=AvatarExpression.NEUTRAL,
            gesture=None,
            eye_color="#4A90E2",
            hair_style="natural",
            clothing="casual",
            background="neutral",
            lighting="natural",
            animation_intensity=0.5,
            emotional_intensity=0.5,
            timestamp=datetime.now()
        )
        
        self.expression_definitions = self._load_expression_definitions()
        self.gesture_definitions = self._load_gesture_definitions()
        self.emotion_mappings = self._load_emotion_mappings()
        self.avatar_customization = self._load_avatar_customization()
        
        # Animation state
        self.is_animating = False
        self.animation_queue = []
        self.current_animation = None
        
        # Start animation loop
        self.is_running = True
        threading.Thread(target=self._animation_loop, daemon=True).start()
    
    def _load_expression_definitions(self) -> Dict[AvatarExpression, ExpressionParameters]:
        """Load expression parameter definitions"""
        return {
            AvatarExpression.HAPPY: ExpressionParameters(
                mouth_curve=0.8,
                eye_openness=0.9,
                eyebrow_position=0.3,
                cheek_redness=0.3,
                eye_sparkle=0.7,
                overall_brightness=1.1
            ),
            AvatarExpression.SAD: ExpressionParameters(
                mouth_curve=-0.6,
                eye_openness=0.6,
                eyebrow_position=-0.4,
                cheek_redness=0.1,
                eye_sparkle=0.2,
                overall_brightness=0.8
            ),
            AvatarExpression.ANGRY: ExpressionParameters(
                mouth_curve=-0.4,
                eye_openness=0.8,
                eyebrow_position=-0.8,
                cheek_redness=0.4,
                eye_sparkle=0.1,
                overall_brightness=0.9
            ),
            AvatarExpression.SURPRISED: ExpressionParameters(
                mouth_curve=0.2,
                eye_openness=1.0,
                eyebrow_position=0.8,
                cheek_redness=0.2,
                eye_sparkle=0.8,
                overall_brightness=1.2
            ),
            AvatarExpression.LOVE: ExpressionParameters(
                mouth_curve=0.9,
                eye_openness=0.8,
                eyebrow_position=0.2,
                cheek_redness=0.6,
                eye_sparkle=0.9,
                overall_brightness=1.3
            ),
            AvatarExpression.PASSION: ExpressionParameters(
                mouth_curve=0.7,
                eye_openness=0.9,
                eyebrow_position=0.4,
                cheek_redness=0.8,
                eye_sparkle=1.0,
                overall_brightness=1.4
            ),
            AvatarExpression.TENDERNESS: ExpressionParameters(
                mouth_curve=0.6,
                eye_openness=0.7,
                eyebrow_position=0.1,
                cheek_redness=0.4,
                eye_sparkle=0.6,
                overall_brightness=1.1
            ),
            AvatarExpression.LONGING: ExpressionParameters(
                mouth_curve=0.1,
                eye_openness=0.8,
                eyebrow_position=-0.2,
                cheek_redness=0.3,
                eye_sparkle=0.5,
                overall_brightness=0.9
            ),
            AvatarExpression.SECURITY: ExpressionParameters(
                mouth_curve=0.5,
                eye_openness=0.8,
                eyebrow_position=0.0,
                cheek_redness=0.2,
                eye_sparkle=0.4,
                overall_brightness=1.0
            ),
            AvatarExpression.EXCITEMENT: ExpressionParameters(
                mouth_curve=0.8,
                eye_openness=1.0,
                eyebrow_position=0.6,
                cheek_redness=0.5,
                eye_sparkle=0.8,
                overall_brightness=1.2
            ),
            AvatarExpression.CALM: ExpressionParameters(
                mouth_curve=0.3,
                eye_openness=0.7,
                eyebrow_position=0.0,
                cheek_redness=0.1,
                eye_sparkle=0.3,
                overall_brightness=0.95
            ),
            AvatarExpression.PLAYFUL: ExpressionParameters(
                mouth_curve=0.7,
                eye_openness=0.9,
                eyebrow_position=0.5,
                cheek_redness=0.4,
                eye_sparkle=0.7,
                overall_brightness=1.1
            ),
            AvatarExpression.SEDUCTIVE: ExpressionParameters(
                mouth_curve=0.6,
                eye_openness=0.6,
                eyebrow_position=0.3,
                cheek_redness=0.7,
                eye_sparkle=0.8,
                overall_brightness=1.2
            ),
            AvatarExpression.VULNERABLE: ExpressionParameters(
                mouth_curve=0.2,
                eye_openness=0.8,
                eyebrow_position=-0.1,
                cheek_redness=0.3,
                eye_sparkle=0.4,
                overall_brightness=0.9
            ),
            AvatarExpression.CONFIDENT: ExpressionParameters(
                mouth_curve=0.5,
                eye_openness=0.9,
                eyebrow_position=0.2,
                cheek_redness=0.2,
                eye_sparkle=0.6,
                overall_brightness=1.1
            ),
            AvatarExpression.NEUTRAL: ExpressionParameters(
                mouth_curve=0.0,
                eye_openness=0.8,
                eyebrow_position=0.0,
                cheek_redness=0.1,
                eye_sparkle=0.3,
                overall_brightness=1.0
            )
        }
    
    def _load_gesture_definitions(self) -> Dict[AvatarGesture, Dict]:
        """Load gesture definitions"""
        return {
            AvatarGesture.WAVE: {
                "duration": 2.0,
                "intensity": 0.6,
                "body_parts": ["right_arm", "hand"],
                "description": "Friendly wave gesture"
            },
            AvatarGesture.BLOW_KISS: {
                "duration": 1.5,
                "intensity": 0.8,
                "body_parts": ["mouth", "hand"],
                "description": "Romantic kiss blowing gesture"
            },
            AvatarGesture.HEART_HANDS: {
                "duration": 3.0,
                "intensity": 0.9,
                "body_parts": ["both_hands"],
                "description": "Forming heart shape with hands"
            },
            AvatarGesture.HUG: {
                "duration": 4.0,
                "intensity": 0.8,
                "body_parts": ["arms", "torso"],
                "description": "Warm embrace gesture"
            },
            AvatarGesture.TOUCH_HEART: {
                "duration": 2.0,
                "intensity": 0.7,
                "body_parts": ["right_hand", "chest"],
                "description": "Touching heart area"
            },
            AvatarGesture.WINK: {
                "duration": 0.5,
                "intensity": 0.6,
                "body_parts": ["left_eye"],
                "description": "Playful wink"
            },
            AvatarGesture.SMILE: {
                "duration": 2.0,
                "intensity": 0.5,
                "body_parts": ["mouth", "cheeks"],
                "description": "Warm smile"
            },
            AvatarGesture.TILT_HEAD: {
                "duration": 1.5,
                "intensity": 0.4,
                "body_parts": ["head", "neck"],
                "description": "Curious head tilt"
            },
            AvatarGesture.REACH_OUT: {
                "duration": 2.5,
                "intensity": 0.7,
                "body_parts": ["arms", "hands"],
                "description": "Reaching out gesture"
            },
            AvatarGesture.DANCE: {
                "duration": 5.0,
                "intensity": 0.8,
                "body_parts": ["full_body"],
                "description": "Joyful dance movement"
            },
            AvatarGesture.EMBRACE: {
                "duration": 4.0,
                "intensity": 0.9,
                "body_parts": ["arms", "torso", "head"],
                "description": "Deep embrace"
            },
            AvatarGesture.STROKE_HAIR: {
                "duration": 2.0,
                "intensity": 0.5,
                "body_parts": ["hands", "hair"],
                "description": "Gentle hair stroking"
            },
            AvatarGesture.HOLD_HANDS: {
                "duration": 3.0,
                "intensity": 0.6,
                "body_parts": ["hands"],
                "description": "Holding hands gesture"
            },
            AvatarGesture.KISS: {
                "duration": 1.0,
                "intensity": 0.9,
                "body_parts": ["mouth", "lips"],
                "description": "Romantic kiss"
            },
            AvatarGesture.GAZE: {
                "duration": 3.0,
                "intensity": 0.7,
                "body_parts": ["eyes"],
                "description": "Intimate gaze"
            },
            AvatarGesture.WHISPER: {
                "duration": 2.0,
                "intensity": 0.6,
                "body_parts": ["mouth", "head"],
                "description": "Whispering gesture"
            }
        }
    
    def _load_emotion_mappings(self) -> Dict[str, AvatarExpression]:
        """Load emotion to expression mappings"""
        return {
            "love": AvatarExpression.LOVE,
            "passion": AvatarExpression.PASSION,
            "tenderness": AvatarExpression.TENDERNESS,
            "longing": AvatarExpression.LONGING,
            "security": AvatarExpression.SECURITY,
            "excitement": AvatarExpression.EXCITEMENT,
            "calm": AvatarExpression.CALM,
            "playful": AvatarExpression.PLAYFUL,
            "seductive": AvatarExpression.SEDUCTIVE,
            "vulnerable": AvatarExpression.VULNERABLE,
            "confident": AvatarExpression.CONFIDENT,
            "happy": AvatarExpression.HAPPY,
            "sad": AvatarExpression.SAD,
            "angry": AvatarExpression.ANGRY,
            "surprised": AvatarExpression.SURPRISED,
            "fearful": AvatarExpression.FEARFUL,
            "disgusted": AvatarExpression.DISGUSTED,
            "neutral": AvatarExpression.NEUTRAL
        }
    
    def _load_avatar_customization(self) -> Dict[str, Dict]:
        """Load avatar customization options"""
        return {
            "eye_colors": {
                "blue": "#4A90E2",
                "green": "#7ED321",
                "brown": "#8B4513",
                "hazel": "#CD853F",
                "gray": "#808080",
                "violet": "#8A2BE2"
            },
            "hair_styles": {
                "natural": "Natural flowing hair",
                "curly": "Curly hair style",
                "straight": "Straight hair style",
                "wavy": "Wavy hair style",
                "braided": "Braided hair style",
                "updo": "Elegant updo style"
            },
            "clothing": {
                "casual": "Casual comfortable clothing",
                "elegant": "Elegant formal wear",
                "romantic": "Romantic evening wear",
                "playful": "Playful colorful clothing",
                "intimate": "Intimate sleepwear",
                "professional": "Professional attire"
            },
            "backgrounds": {
                "neutral": "Neutral background",
                "romantic": "Romantic candlelit background",
                "nature": "Natural outdoor background",
                "home": "Cozy home background",
                "elegant": "Elegant interior background",
                "dreamy": "Dreamy ethereal background"
            },
            "lighting": {
                "natural": "Natural daylight",
                "warm": "Warm golden lighting",
                "romantic": "Romantic soft lighting",
                "dramatic": "Dramatic lighting",
                "soft": "Soft diffused lighting",
                "candlelit": "Candlelit atmosphere"
            }
        }
    
    def update_avatar_mood(self, emotion: str, intensity: float = 0.5, 
                          context: Dict[str, Any] = None):
        """Update avatar mood based on emotion"""
        try:
            # Map emotion to expression
            expression = self.emotion_mappings.get(emotion, AvatarExpression.NEUTRAL)
            
            # Update current state
            self.current_state.expression = expression
            self.current_state.emotional_intensity = intensity
            self.current_state.timestamp = datetime.now()
            
            # Update visual elements based on context
            if context:
                self._update_visual_elements(context)
            
            # Trigger appropriate gesture
            self._trigger_emotion_gesture(emotion, intensity)
            
            print(f"[Avatar] Updated mood to {emotion} (intensity: {intensity:.2f})")
            
        except Exception as e:
            print(f"[Avatar] Error updating mood: {e}")
    
    def _update_visual_elements(self, context: Dict[str, Any]):
        """Update visual elements based on context"""
        # Update eye color based on emotion
        if "eye_color" in context:
            self.current_state.eye_color = context["eye_color"]
        
        # Update hair style based on mood
        if "hair_style" in context:
            self.current_state.hair_style = context["hair_style"]
        
        # Update clothing based on context
        if "clothing" in context:
            self.current_state.clothing = context["clothing"]
        
        # Update background based on setting
        if "background" in context:
            self.current_state.background = context["background"]
        
        # Update lighting based on mood
        if "lighting" in context:
            self.current_state.lighting = context["lighting"]
    
    def _trigger_emotion_gesture(self, emotion: str, intensity: float):
        """Trigger appropriate gesture for emotion"""
        gesture_mappings = {
            "love": AvatarGesture.HEART_HANDS,
            "passion": AvatarGesture.BLOW_KISS,
            "tenderness": AvatarGesture.TOUCH_HEART,
            "longing": AvatarGesture.REACH_OUT,
            "security": AvatarGesture.HUG,
            "excitement": AvatarGesture.DANCE,
            "calm": AvatarGesture.SMILE,
            "playful": AvatarGesture.WINK,
            "seductive": AvatarGesture.GAZE,
            "vulnerable": AvatarGesture.TILT_HEAD,
            "confident": AvatarGesture.SMILE,
            "happy": AvatarGesture.WAVE,
            "sad": AvatarGesture.TILT_HEAD,
            "angry": AvatarGesture.None,
            "surprised": AvatarGesture.None,
            "fearful": AvatarGesture.None,
            "disgusted": AvatarGesture.None,
            "neutral": AvatarGesture.None
        }
        
        gesture = gesture_mappings.get(emotion)
        if gesture and intensity > 0.3:
            self.trigger_gesture(gesture, intensity)
    
    def trigger_gesture(self, gesture: AvatarGesture, intensity: float = 0.5):
        """Trigger a specific gesture"""
        try:
            # Add gesture to animation queue
            self.animation_queue.append({
                "gesture": gesture,
                "intensity": intensity,
                "timestamp": datetime.now()
            })
            
            print(f"[Avatar] Triggered gesture: {gesture.value}")
            
        except Exception as e:
            print(f"[Avatar] Error triggering gesture: {e}")
    
    def _animation_loop(self):
        """Main animation loop"""
        while self.is_running:
            try:
                # Process animation queue
                if self.animation_queue and not self.is_animating:
                    animation = self.animation_queue.pop(0)
                    self._execute_animation(animation)
                
                # Update current animation
                if self.current_animation:
                    self._update_current_animation()
                
                time.sleep(0.1)  # 10 FPS
                
            except Exception as e:
                print(f"[Avatar] Animation loop error: {e}")
                time.sleep(1.0)
    
    def _execute_animation(self, animation: Dict):
        """Execute an animation"""
        try:
            gesture = animation["gesture"]
            intensity = animation["intensity"]
            
            # Get gesture definition
            gesture_def = self.gesture_definitions.get(gesture)
            if not gesture_def:
                return
            
            # Set current animation
            self.current_animation = {
                "gesture": gesture,
                "intensity": intensity,
                "duration": gesture_def["duration"],
                "start_time": time.time(),
                "body_parts": gesture_def["body_parts"]
            }
            
            # Update avatar state
            self.current_state.gesture = gesture
            self.current_state.animation_intensity = intensity
            self.is_animating = True
            
            print(f"[Avatar] Started animation: {gesture.value}")
            
        except Exception as e:
            print(f"[Avatar] Error executing animation: {e}")
    
    def _update_current_animation(self):
        """Update current animation progress"""
        if not self.current_animation:
            return
        
        current_time = time.time()
        start_time = self.current_animation["start_time"]
        duration = self.current_animation["duration"]
        
        # Check if animation is complete
        if current_time - start_time >= duration:
            self._complete_animation()
        else:
            # Update animation progress
            progress = (current_time - start_time) / duration
            self._update_animation_progress(progress)
    
    def _update_animation_progress(self, progress: float):
        """Update animation progress"""
        # In a real implementation, this would update the visual animation
        # For now, we'll just track the progress
        pass
    
    def _complete_animation(self):
        """Complete current animation"""
        self.current_animation = None
        self.current_state.gesture = None
        self.is_animating = False
        
        print("[Avatar] Animation completed")
    
    def get_avatar_state(self) -> Dict[str, Any]:
        """Get current avatar state"""
        expression_params = self.expression_definitions.get(self.current_state.expression)
        
        return {
            "expression": self.current_state.expression.value,
            "gesture": self.current_state.gesture.value if self.current_state.gesture else None,
            "eye_color": self.current_state.eye_color,
            "hair_style": self.current_state.hair_style,
            "clothing": self.current_state.clothing,
            "background": self.current_state.background,
            "lighting": self.current_state.lighting,
            "animation_intensity": self.current_state.animation_intensity,
            "emotional_intensity": self.current_state.emotional_intensity,
            "expression_parameters": {
                "mouth_curve": expression_params.mouth_curve if expression_params else 0.0,
                "eye_openness": expression_params.eye_openness if expression_params else 0.8,
                "eyebrow_position": expression_params.eyebrow_position if expression_params else 0.0,
                "cheek_redness": expression_params.cheek_redness if expression_params else 0.1,
                "eye_sparkle": expression_params.eye_sparkle if expression_params else 0.3,
                "overall_brightness": expression_params.overall_brightness if expression_params else 1.0
            },
            "timestamp": self.current_state.timestamp.isoformat(),
            "is_animating": self.is_animating
        }
    
    def customize_avatar(self, eye_color: str = None, hair_style: str = None,
                        clothing: str = None, background: str = None,
                        lighting: str = None):
        """Customize avatar appearance"""
        try:
            if eye_color and eye_color in self.avatar_customization["eye_colors"]:
                self.current_state.eye_color = self.avatar_customization["eye_colors"][eye_color]
            
            if hair_style and hair_style in self.avatar_customization["hair_styles"]:
                self.current_state.hair_style = hair_style
            
            if clothing and clothing in self.avatar_customization["clothing"]:
                self.current_state.clothing = clothing
            
            if background and background in self.avatar_customization["backgrounds"]:
                self.current_state.background = background
            
            if lighting and lighting in self.avatar_customization["lighting"]:
                self.current_state.lighting = lighting
            
            print(f"[Avatar] Customized appearance")
            
        except Exception as e:
            print(f"[Avatar] Error customizing avatar: {e}")
    
    def get_available_customizations(self) -> Dict[str, List[str]]:
        """Get available customization options"""
        return {
            "eye_colors": list(self.avatar_customization["eye_colors"].keys()),
            "hair_styles": list(self.avatar_customization["hair_styles"].keys()),
            "clothing": list(self.avatar_customization["clothing"].keys()),
            "backgrounds": list(self.avatar_customization["backgrounds"].keys()),
            "lighting": list(self.avatar_customization["lighting"].keys())
        }
    
    def get_avatar_summary(self) -> Dict[str, Any]:
        """Get avatar system summary"""
        return {
            "current_expression": self.current_state.expression.value,
            "current_gesture": self.current_state.gesture.value if self.current_state.gesture else None,
            "emotional_intensity": self.current_state.emotional_intensity,
            "animation_intensity": self.current_state.animation_intensity,
            "is_animating": self.is_animating,
            "queue_length": len(self.animation_queue),
            "available_expressions": [e.value for e in AvatarExpression],
            "available_gestures": [g.value for g in AvatarGesture],
            "system_status": "active" if self.is_running else "inactive"
        }

# Global mood-driven avatar instance
mood_driven_avatar = MoodDrivenAvatar()

def get_mood_driven_avatar() -> MoodDrivenAvatar:
    """Get the global mood-driven avatar instance"""
    return mood_driven_avatar

def update_avatar_mood(emotion: str, intensity: float = 0.5, context: Dict[str, Any] = None):
    """Update avatar mood with convenience function"""
    mood_driven_avatar.update_avatar_mood(emotion, intensity, context)

def trigger_avatar_gesture(gesture: str, intensity: float = 0.5):
    """Trigger avatar gesture with convenience function"""
    try:
        gesture_enum = AvatarGesture(gesture)
        mood_driven_avatar.trigger_gesture(gesture_enum, intensity)
    except Exception as e:
        print(f"[Avatar] Error triggering gesture: {e}")

def get_avatar_state() -> Dict[str, Any]:
    """Get avatar state with convenience function"""
    return mood_driven_avatar.get_avatar_state()

def customize_avatar(eye_color: str = None, hair_style: str = None,
                    clothing: str = None, background: str = None,
                    lighting: str = None):
    """Customize avatar with convenience function"""
    mood_driven_avatar.customize_avatar(eye_color, hair_style, clothing, background, lighting) 
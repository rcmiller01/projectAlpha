# romantic_avatar.py
# Phase 2: Visual avatar system for romantic companionship

import json
import random
from typing import Dict, List, Optional
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

class AvatarExpression(Enum):
    HAPPY = "happy"
    LOVING = "loving"
    PASSIONATE = "passionate"
    TENDER = "tender"
    LONGING = "longing"
    PLAYFUL = "playful"
    VULNERABLE = "vulnerable"
    SECURE = "secure"
    AFFECTIONATE = "affectionate"
    CALM = "calm"

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

@dataclass
class AvatarState:
    expression: AvatarExpression
    gesture: Optional[AvatarGesture]
    eye_contact: bool
    blush_intensity: float  # 0.0 to 1.0
    smile_intensity: float  # 0.0 to 1.0
    animation_speed: float  # 0.5 to 2.0
    timestamp: datetime

class RomanticAvatar:
    def __init__(self):
        self.current_state = AvatarState(
            expression=AvatarExpression.CALM,
            gesture=None,
            eye_contact=True,
            blush_intensity=0.0,
            smile_intensity=0.3,
            animation_speed=1.0,
            timestamp=datetime.now()
        )
        
        # Expression mappings for romantic emotions
        self.expression_mappings = {
            "love": [AvatarExpression.LOVING, AvatarExpression.AFFECTIONATE],
            "longing": [AvatarExpression.LONGING, AvatarExpression.VULNERABLE],
            "passion": [AvatarExpression.PASSIONATE, AvatarExpression.LOVING],
            "tenderness": [AvatarExpression.TENDER, AvatarExpression.SECURE],
            "security": [AvatarExpression.SECURE, AvatarExpression.CALM],
            "affection": [AvatarExpression.AFFECTIONATE, AvatarExpression.PLAYFUL],
            "joy": [AvatarExpression.HAPPY, AvatarExpression.PLAYFUL],
            "calm": [AvatarExpression.CALM, AvatarExpression.SECURE]
        }
        
        # Gesture mappings for romantic interactions
        self.gesture_mappings = {
            "greeting": [AvatarGesture.WAVE, AvatarGesture.SMILE],
            "affection": [AvatarGesture.HEART_HANDS, AvatarGesture.BLOW_KISS],
            "comfort": [AvatarGesture.HUG, AvatarGesture.REACH_OUT],
            "playful": [AvatarGesture.WINK, AvatarGesture.DANCE],
            "intimate": [AvatarGesture.TOUCH_HEART, AvatarGesture.TILT_HEAD]
        }
        
        # Avatar appearance settings
        self.appearance = {
            "hair_color": "warm_brown",
            "eye_color": "deep_green",
            "skin_tone": "warm_medium",
            "height": "average",
            "build": "slender",
            "clothing_style": "romantic_casual"
        }
        
    def update_expression(self, emotion: str, intensity: float = 1.0):
        """Update avatar expression based on emotion"""
        if emotion in self.expression_mappings:
            possible_expressions = self.expression_mappings[emotion]
            self.current_state.expression = random.choice(possible_expressions)
            
            # Update related visual properties
            if emotion in ["love", "passion", "affection"]:
                self.current_state.blush_intensity = min(1.0, intensity * 0.8)
                self.current_state.smile_intensity = min(1.0, intensity * 0.9)
            elif emotion == "longing":
                self.current_state.blush_intensity = 0.3
                self.current_state.smile_intensity = 0.2
            elif emotion == "security":
                self.current_state.blush_intensity = 0.1
                self.current_state.smile_intensity = 0.6
            
            self.current_state.timestamp = datetime.now()
    
    def perform_gesture(self, gesture_type: str):
        """Perform a romantic gesture"""
        if gesture_type in self.gesture_mappings:
            possible_gestures = self.gesture_mappings[gesture_type]
            self.current_state.gesture = random.choice(possible_gestures)
            self.current_state.timestamp = datetime.now()
    
    def set_eye_contact(self, enabled: bool):
        """Set eye contact state"""
        self.current_state.eye_contact = enabled
        self.current_state.timestamp = datetime.now()
    
    def get_visual_state(self) -> Dict:
        """Get current visual state for frontend rendering"""
        return {
            "expression": self.current_state.expression.value,
            "gesture": self.current_state.gesture.value if self.current_state.gesture else None,
            "eye_contact": self.current_state.eye_contact,
            "blush_intensity": self.current_state.blush_intensity,
            "smile_intensity": self.current_state.smile_intensity,
            "animation_speed": self.current_state.animation_speed,
            "appearance": self.appearance,
            "timestamp": self.current_state.timestamp.isoformat()
        }
    
    def generate_romantic_scene(self, scene_type: str) -> Dict:
        """Generate romantic scene settings"""
        scenes = {
            "sunset": {
                "background": "warm_sunset_gradient",
                "lighting": "golden_hour",
                "atmosphere": "romantic_warm",
                "music_suggestion": "soft_jazz"
            },
            "garden": {
                "background": "flower_garden",
                "lighting": "natural_soft",
                "atmosphere": "peaceful_natural",
                "music_suggestion": "nature_sounds"
            },
            "bedroom": {
                "background": "cozy_bedroom",
                "lighting": "warm_lamp",
                "atmosphere": "intimate_cozy",
                "music_suggestion": "romantic_piano"
            },
            "beach": {
                "background": "ocean_sunset",
                "lighting": "dramatic_sunset",
                "atmosphere": "dreamy_romantic",
                "music_suggestion": "ocean_waves"
            }
        }
        
        return scenes.get(scene_type, scenes["sunset"])
    
    def create_romantic_animation(self, emotion: str, duration: float = 3.0) -> Dict:
        """Create romantic animation sequence"""
        animations = {
            "love": {
                "sequence": [
                    {"expression": "loving", "gesture": "heart_hands", "duration": 1.0},
                    {"expression": "affectionate", "gesture": "blow_kiss", "duration": 1.0},
                    {"expression": "loving", "gesture": None, "duration": 1.0}
                ],
                "blush_curve": [0.0, 0.8, 0.6],
                "smile_curve": [0.3, 0.9, 0.7]
            },
            "longing": {
                "sequence": [
                    {"expression": "longing", "gesture": "reach_out", "duration": 1.5},
                    {"expression": "vulnerable", "gesture": None, "duration": 1.5}
                ],
                "blush_curve": [0.2, 0.4],
                "smile_curve": [0.1, 0.2]
            },
            "passion": {
                "sequence": [
                    {"expression": "passionate", "gesture": "touch_heart", "duration": 1.0},
                    {"expression": "loving", "gesture": "tilt_head", "duration": 1.0},
                    {"expression": "passionate", "gesture": None, "duration": 1.0}
                ],
                "blush_curve": [0.3, 0.9, 0.8],
                "smile_curve": [0.4, 0.8, 0.7]
            }
        }
        
        return animations.get(emotion, animations["love"])

# Global avatar instance
romantic_avatar = RomanticAvatar() 
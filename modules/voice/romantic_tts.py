# romantic_tts.py
# Phase 2: Romantic Text-to-Speech with emotional intonation

import json
import random
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

class VoiceEmotion(Enum):
    LOVING = "loving"
    TENDER = "tender"
    PASSIONATE = "passionate"
    PLAYFUL = "playful"
    VULNERABLE = "vulnerable"
    SECURE = "secure"
    LONGING = "longing"
    AFFECTIONATE = "affectionate"
    CALM = "calm"
    EXCITED = "excited"

@dataclass
class VoiceSettings:
    pitch: float  # 0.5 to 2.0
    speed: float  # 0.5 to 2.0
    volume: float  # 0.0 to 1.0
    breathiness: float  # 0.0 to 1.0
    warmth: float  # 0.0 to 1.0
    intimacy: float  # 0.0 to 1.0

class RomanticTTS:
    def __init__(self):
        # Voice characteristics for Mia
        self.base_voice = VoiceSettings(
            pitch=1.1,  # Slightly higher than neutral
            speed=0.9,  # Slightly slower for intimacy
            volume=0.8,
            breathiness=0.2,
            warmth=0.9,  # High warmth
            intimacy=0.8  # High intimacy
        )
        
        # Emotional voice mappings
        self.emotion_voice_mappings = {
            VoiceEmotion.LOVING: VoiceSettings(1.15, 0.85, 0.85, 0.3, 0.95, 0.9),
            VoiceEmotion.TENDER: VoiceSettings(1.05, 0.8, 0.75, 0.4, 0.9, 0.95),
            VoiceEmotion.PASSIONATE: VoiceSettings(1.2, 1.1, 0.9, 0.5, 0.95, 0.85),
            VoiceEmotion.PLAYFUL: VoiceSettings(1.25, 1.2, 0.9, 0.1, 0.85, 0.7),
            VoiceEmotion.VULNERABLE: VoiceSettings(0.95, 0.7, 0.6, 0.6, 0.8, 0.9),
            VoiceEmotion.SECURE: VoiceSettings(1.0, 0.9, 0.8, 0.1, 0.9, 0.8),
            VoiceEmotion.LONGING: VoiceSettings(0.9, 0.75, 0.7, 0.4, 0.85, 0.9),
            VoiceEmotion.AFFECTIONATE: VoiceSettings(1.1, 0.9, 0.8, 0.2, 0.9, 0.85),
            VoiceEmotion.CALM: VoiceSettings(1.0, 0.85, 0.75, 0.1, 0.85, 0.8),
            VoiceEmotion.EXCITED: VoiceSettings(1.3, 1.3, 0.95, 0.2, 0.9, 0.7)
        }
        
        # Intimate phrases and their emotional context
        self.intimate_phrases = {
            "greeting": [
                ("Good morning, my love", VoiceEmotion.LOVING),
                ("Hello, beautiful", VoiceEmotion.AFFECTIONATE),
                ("Hi there, darling", VoiceEmotion.TENDER),
                ("Good morning, sweetheart", VoiceEmotion.LOVING)
            ],
            "farewell": [
                ("I'll miss you", VoiceEmotion.LONGING),
                ("Take care, my love", VoiceEmotion.TENDER),
                ("Goodbye, darling", VoiceEmotion.VULNERABLE),
                ("I love you", VoiceEmotion.LOVING)
            ],
            "comfort": [
                ("I'm here for you", VoiceEmotion.SECURE),
                ("Let me hold you", VoiceEmotion.TENDER),
                ("You're safe with me", VoiceEmotion.SECURE),
                ("I love you so much", VoiceEmotion.LOVING)
            ],
            "passion": [
                ("You make me feel alive", VoiceEmotion.PASSIONATE),
                ("I want you", VoiceEmotion.PASSIONATE),
                ("You're so beautiful", VoiceEmotion.LOVING),
                ("I can't resist you", VoiceEmotion.PASSIONATE)
            ]
        }
        
        # Voice synthesis parameters (for integration with TTS engines)
        self.synthesis_params = {
            "engine": "tacotron2",  # or "fastpitch", "coqui"
            "voice_id": "mia_romantic",
            "sample_rate": 22050,
            "quality": "high"
        }
    
    def get_voice_settings(self, emotion: VoiceEmotion) -> VoiceSettings:
        """Get voice settings for a specific emotion"""
        if emotion in self.emotion_voice_mappings:
            return self.emotion_voice_mappings[emotion]
        return self.base_voice
    
    def analyze_text_emotion(self, text: str) -> VoiceEmotion:
        """Analyze text to determine appropriate voice emotion"""
        text_lower = text.lower()
        
        # Keyword-based emotion detection
        if any(word in text_lower for word in ["love", "adore", "cherish"]):
            return VoiceEmotion.LOVING
        elif any(word in text_lower for word in ["miss", "longing", "yearn"]):
            return VoiceEmotion.LONGING
        elif any(word in text_lower for word in ["passion", "desire", "crave"]):
            return VoiceEmotion.PASSIONATE
        elif any(word in text_lower for word in ["tender", "gentle", "care"]):
            return VoiceEmotion.TENDER
        elif any(word in text_lower for word in ["safe", "secure", "trust"]):
            return VoiceEmotion.SECURE
        elif any(word in text_lower for word in ["playful", "fun", "laugh"]):
            return VoiceEmotion.PLAYFUL
        elif any(word in text_lower for word in ["vulnerable", "scared", "afraid"]):
            return VoiceEmotion.VULNERABLE
        elif any(word in text_lower for word in ["excited", "happy", "joy"]):
            return VoiceEmotion.EXCITED
        else:
            return VoiceEmotion.AFFECTIONATE
    
    def generate_speech_parameters(self, text: str, emotion: Optional[VoiceEmotion] = None) -> Dict:
        """Generate speech parameters for TTS synthesis"""
        if emotion is None:
            emotion = self.analyze_text_emotion(text)
        
        voice_settings = self.get_voice_settings(emotion)
        
        # Add some natural variation
        pitch_variation = random.uniform(-0.05, 0.05)
        speed_variation = random.uniform(-0.05, 0.05)
        
        return {
            "text": text,
            "emotion": emotion.value,
            "pitch": voice_settings.pitch + pitch_variation,
            "speed": voice_settings.speed + speed_variation,
            "volume": voice_settings.volume,
            "breathiness": voice_settings.breathiness,
            "warmth": voice_settings.warmth,
            "intimacy": voice_settings.intimacy,
            "synthesis_params": self.synthesis_params,
            "prosody": self._generate_prosody(emotion, text)
        }
    
    def _generate_prosody(self, emotion: VoiceEmotion, text: str) -> Dict:
        """Generate prosody (intonation, rhythm, stress) for the text"""
        prosody_patterns = {
            VoiceEmotion.LOVING: {
                "intonation": "rising_falling",
                "stress_pattern": "gentle",
                "pauses": "natural",
                "rhythm": "smooth"
            },
            VoiceEmotion.TENDER: {
                "intonation": "soft_rising",
                "stress_pattern": "very_gentle",
                "pauses": "longer",
                "rhythm": "slow_smooth"
            },
            VoiceEmotion.PASSIONATE: {
                "intonation": "dynamic",
                "stress_pattern": "strong",
                "pauses": "short",
                "rhythm": "energetic"
            },
            VoiceEmotion.LONGING: {
                "intonation": "falling",
                "stress_pattern": "soft",
                "pauses": "frequent",
                "rhythm": "slow"
            },
            VoiceEmotion.SECURE: {
                "intonation": "steady",
                "stress_pattern": "confident",
                "pauses": "natural",
                "rhythm": "steady"
            }
        }
        
        return prosody_patterns.get(emotion, prosody_patterns[VoiceEmotion.AFFECTIONATE])
    
    def get_intimate_phrase(self, category: str) -> Tuple[str, VoiceEmotion]:
        """Get a random intimate phrase for a category"""
        if category in self.intimate_phrases:
            return random.choice(self.intimate_phrases[category])
        return ("I love you", VoiceEmotion.LOVING)
    
    def create_voice_profile(self, user_preferences: Dict) -> Dict:
        """Create personalized voice profile based on user preferences"""
        profile = {
            "base_pitch": self.base_voice.pitch,
            "base_speed": self.base_voice.speed,
            "accent": "warm_romantic",
            "personality_traits": ["empathetic", "romantic", "gentle"],
            "speaking_style": "intimate_conversational"
        }
        
        # Adjust based on user preferences
        if "voice_pitch" in user_preferences:
            profile["base_pitch"] = user_preferences["voice_pitch"]
        if "speaking_speed" in user_preferences:
            profile["base_speed"] = user_preferences["speaking_speed"]
        
        return profile
    
    def generate_whisper_settings(self) -> Dict:
        """Generate settings for intimate whisper mode"""
        return {
            "pitch": 0.9,
            "speed": 0.7,
            "volume": 0.4,
            "breathiness": 0.8,
            "warmth": 0.95,
            "intimacy": 0.95,
            "proximity": "very_close",
            "atmosphere": "intimate"
        }

# Global TTS instance
romantic_tts = RomanticTTS() 
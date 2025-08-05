"""
Emotional Voice Engine - Integrates with emotional configuration
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.config.emotion_config_manager import emotion_config
import logging

logger = logging.getLogger(__name__)

class EmotionalVoiceEngine:
    def __init__(self):
        self.current_emotional_state = "balanced"
        
        # Register for configuration updates
        emotion_config.register_config_callback(
            "tone_profiles", 
            self.on_tone_profiles_updated
        )
    
    def set_emotional_state(self, emotional_state: str):
        """Update current emotional state and voice settings"""
        self.current_emotional_state = emotional_state
        
        # Get tone profile for this emotional state
        tone_profile = emotion_config.get_tone_profile(emotional_state)
        
        if tone_profile:
            voice_settings = tone_profile.get("voice_settings", {})
            self.apply_voice_settings(voice_settings)
    
    def apply_voice_settings(self, voice_settings: dict):
        """Apply voice modifications based on emotional state"""
        # This would integrate with actual voice synthesis
        logger.info(f"Applying voice settings: {voice_settings}")
        
        # Example integration points:
        # - pitch_modifier: Adjust voice pitch
        # - speed_modifier: Adjust speaking rate
        # - breathiness: Add breath/whisper effects
        # - warmth: Adjust tone warmth
    
    def get_current_voice_profile(self):
        """Get current voice configuration"""
        return emotion_config.get_tone_profile(self.current_emotional_state)
    
    def on_tone_profiles_updated(self, new_config):
        """Handle tone profile configuration changes"""
        logger.info("Tone profiles updated - refreshing voice settings")
        self.set_emotional_state(self.current_emotional_state)

# Global voice engine instance
emotional_voice = EmotionalVoiceEngine()

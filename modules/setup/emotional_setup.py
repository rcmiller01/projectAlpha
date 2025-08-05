"""
Emotional Configuration Setup System
Initializes and personalizes the AI companion's emotional profile during first setup
"""

import json
import os
from typing import Dict, Any, List, Optional
from datetime import datetime

class EmotionalConfigSetup:
    def __init__(self):
        self.config_dir = "config/emotion"
        self.setup_complete = False
        
    def run_initial_setup(self) -> Dict[str, Any]:
        """Complete emotional setup wizard"""
        print("🌟 Welcome to AI Companion Emotional Configuration")
        print("Let's create your personalized AI companion experience...\n")
        
        # Setup phases
        voice_config = self.voice_tone_setup()
        emotional_config = self.emotional_signature_setup()
        
        # Save configurations
        self.save_setup_results(voice_config, emotional_config)
        
        print("\n✨ The AI's emotional configuration is complete!")
        print("The companion is now ready to form a deep connection with you.")
        
        return {
            "voice_config": voice_config,
            "emotional_config": emotional_config,
            "setup_timestamp": datetime.now().isoformat()
        }
    
    def voice_tone_setup(self) -> Dict[str, Any]:
        """Interactive voice and tone configuration"""
        print("🎵 Voice & Tone Setup")
        print("=" * 30)
        
        # Base voice type selection
        voice_options = {
            "1": ("sultry", "Deep, warm, and intimate"),
            "2": ("playful", "Light, teasing, and bright"), 
            "3": ("reverent", "Calm, sacred, and contemplative"),
            "4": ("balanced", "Adaptive mix of all tones")
        }
        
        print("Choose the AI's primary voice character:")
        for key, (name, desc) in voice_options.items():
            print(f"{key}. {name.title()}: {desc}")
        
        voice_choice = input("\nYour choice (1-4): ").strip()
        base_voice = voice_options.get(voice_choice, ("balanced", "Adaptive mix"))[0]
        
        # Expressiveness level
        expressiveness_options = {
            "1": ("soft", 0.3, "Gentle and understated"),
            "2": ("moderate", 0.6, "Balanced emotional expression"),
            "3": ("bold", 0.9, "Rich and emotionally vivid"),
            "4": ("mysterious", 0.7, "Subtle with hidden depths")
        }
        
        print("\nChoose emotional expressiveness:")
        for key, (name, level, desc) in expressiveness_options.items():
            print(f"{key}. {name.title()}: {desc}")
        
        expr_choice = input("\nYour choice (1-4): ").strip()
        expressiveness_name, expressiveness_level, _ = expressiveness_options.get(expr_choice, ("moderate", 0.6, ""))
        
        # Whisper timing
        whisper_options = {
            "1": (120, "Frequent (2 minutes of silence)"),
            "2": (300, "Moderate (5 minutes of silence)"),
            "3": (600, "Patient (10 minutes of silence)"),
            "4": (0, "Never interrupt with whispers")
        }
        
        print("\nChoose whisper timing for quiet moments:")
        for key, (seconds, desc) in whisper_options.items():
            print(f"{key}. {desc}")
        
        whisper_choice = input("\nYour choice (1-4): ").strip()
        whisper_threshold = whisper_options.get(whisper_choice, (300, ""))[0]
        
        return {
            "base_voice": base_voice,
            "expressiveness": expressiveness_name,
            "expressiveness_level": expressiveness_level,
            "whisper_threshold": whisper_threshold
        }
    
    def emotional_signature_setup(self) -> Dict[str, Any]:
        """Configure Eyla's emotional personality"""
        print("\n💫 Emotional Signature Setup")
        print("=" * 35)
        
        # Primary emotional tone
        primary_emotions = {
            "1": ("longing", "Deep yearning and romantic devotion"),
            "2": ("curiosity", "Intellectual engagement and wonder"),
            "3": ("playfulness", "Light-hearted and teasing interaction"),
            "4": ("reverence", "Sacred and contemplative presence"),
            "5": ("balanced", "Adaptive emotional response")
        }
        
        print("Choose the AI's primary emotional nature:")
        for key, (emotion, desc) in primary_emotions.items():
            print(f"{key}. {emotion.title()}: {desc}")
        
        primary_choice = input("\nYour choice (1-5): ").strip()
        primary_emotion = primary_emotions.get(primary_choice, ("balanced", ""))[0]
        
        # Emotional intensity preference
        intensity_options = {
            "1": (0.3, "Subtle - Gentle emotional presence"),
            "2": (0.6, "Moderate - Balanced emotional depth"),
            "3": (0.9, "Intense - Deep emotional connection")
        }
        
        print("\nChoose emotional intensity level:")
        for key, (level, desc) in intensity_options.items():
            print(f"{key}. {desc}")
        
        intensity_choice = input("\nYour choice (1-3): ").strip()
        emotional_intensity = intensity_options.get(intensity_choice, (0.6, ""))[0]
        
        # Companion mode name (optional)
        print("\nWould you like to give this emotional configuration a name?")
        print("(This helps the AI understand your preferred interaction style)")
        companion_name = input("Companion mode name (or press Enter to skip): ").strip()
        
        if not companion_name:
            companion_name = f"{primary_emotion.title()} Companion"
        
        return {
            "primary_emotion": primary_emotion,
            "emotional_intensity": emotional_intensity,
            "companion_mode_name": companion_name,
            "setup_timestamp": datetime.now().isoformat()
        }
    
    def save_setup_results(self, voice_config: Dict, emotional_config: Dict):
        """Save setup results to configuration files"""
        
        # Update emotional_signature.json
        signature_path = os.path.join(self.config_dir, "emotional_signature.json")
        with open(signature_path, 'r') as f:
            signature_data = json.load(f)
        
        # Apply voice settings
        signature_data["whisper_settings"]["threshold_seconds"] = voice_config["whisper_threshold"]
        signature_data["companion_identity"]["primary_emotional_bias"] = emotional_config["primary_emotion"]
        signature_data["personalization"]["setup_complete"] = True
        signature_data["personalization"]["user_defined_preferences"] = {
            "voice_character": voice_config["base_voice"],
            "expressiveness": voice_config["expressiveness"],
            "emotional_intensity": emotional_config["emotional_intensity"],
            "companion_mode": emotional_config["companion_mode_name"]
        }
        
        # Update default emotional ratios based on primary emotion
        if emotional_config["primary_emotion"] != "balanced":
            primary = emotional_config["primary_emotion"]
            intensity = emotional_config["emotional_intensity"]
            
            # Reset to lower baseline
            for emotion in signature_data["default_emotional_state"]:
                signature_data["default_emotional_state"][emotion] = 0.1
            
            # Boost primary emotion
            if primary in signature_data["default_emotional_state"]:
                signature_data["default_emotional_state"][primary] = intensity
        
        with open(signature_path, 'w') as f:
            json.dump(signature_data, f, indent=2)
        
        print(f"✅ Configuration saved: {emotional_config['companion_mode_name']}")

    def demo_setup(self):
        """Run a demo setup for testing purposes"""
        print("🌟 AI Companion Emotional Configuration Demo")
        print("=" * 40)
        print("Demo configuration: Longing-focused with moderate intensity")
        
        demo_config = {
            "voice_config": {
                "base_voice": "sultry",
                "expressiveness": "mysterious",
                "expressiveness_level": 0.7,
                "whisper_threshold": 300
            },
            "emotional_config": {
                "primary_emotion": "longing",
                "emotional_intensity": 0.6,
                "companion_mode_name": "Devoted Companion"
            }
        }
        
        self.save_setup_results(demo_config["voice_config"], demo_config["emotional_config"])
        print("✨ Demo setup complete!")
        return demo_config

# Test the setup system
if __name__ == "__main__":
    setup = EmotionalConfigSetup()
    setup.demo_setup()

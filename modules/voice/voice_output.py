import sys
import os
from typing import Optional

# Add project root to path to allow module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from modules.reflection.emotion_reflector import EmotionReflector
from modules.presence.voice_integration import EmotionalVoiceEngine

import pyttsx3

# Initialize engines
engine = pyttsx3.init()
emotion_reflector = EmotionReflector()
emotional_voice_engine = EmotionalVoiceEngine()

def speak(text: str, emotion: Optional[str] = None):
    """
    Synthesize speech with optional emotional tone.
    """
    if not emotion:
        # Get the most recent dominant emotion from reflection
        summary = emotion_reflector.summarize_reflection(days=1)
        emotion = summary.get("dominant_emotional_state")

    # Fallback to a neutral emotion if none is found
    if not emotion:
        emotion = "neutral"

    # Get voice processing parameters from the emotional voice engine
    voice_params = emotional_voice_engine.apply_voice_processing(text, emotion)
    audio_params = voice_params.get('audio_params', {})

    # Get current properties
    current_rate = engine.getProperty('rate')
    
    # Apply voice modifications
    # Note: pyttsx3 has limited support for fine-grained control.
    # We'll primarily adjust rate and pitch.
    
    # Adjust speed (rate)
    speed_multiplier = audio_params.get('speed_multiplier', 1.0)
    engine.setProperty('rate', int(current_rate * speed_multiplier))

    # Adjust pitch
    # pyttsx3 pitch is 0-100, we map from semitones
    pitch_shift = audio_params.get('pitch_shift_semitones', 0.0)
    # Assuming base pitch is 50, let's scale the shift
    new_pitch = 50 + (pitch_shift * 2) # Scale factor of 2 for noticeable change
    engine.setProperty('pitch', max(0, min(100, new_pitch)))

    print(f"Speaking with text: '{text}', emotion: {emotion}, pitch: {new_pitch:.1f}, speed_mult: {speed_multiplier:.2f}")
    
    engine.say(text)
    engine.runAndWait()

    # Reset to default properties after speaking
    engine.setProperty('rate', current_rate)
    engine.setProperty('pitch', 50)


# Example usage
if __name__ == "__main__":
    print("Testing default voice...")
    speak("Hello, this is a test of the default voice.")
    
    print("\nTesting emotional voices...")
    test_emotions = ["joy", "longing", "melancholy", "warmth"]
    for e in test_emotions:
        speak(f"This is a test of the {e} voice.", emotion=e)

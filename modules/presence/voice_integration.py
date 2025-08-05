"""
Voice Engine Integration for Emotional Presence

This module provides voice synthesis and modification capabilities
for expressing emotional presence through tone, pitch, speed,
breathiness, and ambient whispers.
"""

import json
import asyncio
import random
import math
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass
from pathlib import Path
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

@dataclass
class VoiceModifier:
    """Voice modification parameters"""
    pitch: float = 0.0  # -1.0 to 1.0 (semitones adjustment)
    speed: float = 1.0  # 0.5 to 2.0 (playback speed multiplier)
    breathiness: float = 0.0  # 0.0 to 1.0 (breath noise level)
    warmth: float = 0.5  # 0.0 to 1.0 (formant frequency adjustment)
    reverb: float = 0.0  # 0.0 to 1.0 (reverb/echo level)
    whisper_mode: bool = False  # Enable whisper-like voice
    
@dataclass
class WhisperEvent:
    """Scheduled whisper event"""
    text: str
    scheduled_time: datetime
    emotion: str
    priority: int = 1  # 1=low, 5=urgent
    duration: float = 3.0
    fade_in: float = 0.5
    fade_out: float = 0.5

class EmotionalVoiceEngine:
    """Core voice engine for emotional speech synthesis"""
    
    def __init__(self, config_path: str = "data/emotional_signatures.json"):
        self.config_path = Path(config_path)
        self.signatures = {}
        self.current_emotion: Optional[str] = None
        self.base_voice_config = {
            'pitch': 0.0,
            'speed': 1.0,
            'breathiness': 0.1,
            'warmth': 0.6
        }
        self.load_signatures()
        
    def load_signatures(self):
        """Load emotional voice signatures"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    self.signatures = json.load(f)
                logger.info(f"Loaded voice signatures for {len(self.signatures)} emotions")
            else:
                logger.warning(f"Emotional signatures file not found: {self.config_path}")
                self.signatures = {}
        except Exception as e:
            logger.error(f"Error loading emotional signatures: {e}")
            self.signatures = {}
    
    def get_voice_modifier(self, emotion: str, intensity: float = 0.7) -> VoiceModifier:
        """Get voice modifier for specific emotion and intensity"""
        if emotion not in self.signatures:
            logger.warning(f"Unknown emotion for voice modification: {emotion}")
            return VoiceModifier()
            
        signature = self.signatures[emotion]
        voice_config = signature.get('voice_modifier', {})
        
        # Apply intensity scaling to modifications
        return VoiceModifier(
            pitch=voice_config.get('pitch', 0.0) * intensity,
            speed=1.0 + (voice_config.get('speed', 1.0) - 1.0) * intensity,
            breathiness=voice_config.get('breathiness', 0.0) * intensity,
            warmth=voice_config.get('warmth', 0.5),  # Warmth doesn't scale with intensity
            reverb=voice_config.get('reverb', 0.0) * intensity,
            whisper_mode=voice_config.get('whisper_mode', False)
        )
    
    def apply_dynamic_tone_modifier(self, emotion: str, intensity: float) -> VoiceModifier:
        """Apply dynamic tone adjustments based on emotion and intensity."""
        base_modifier = self.get_voice_modifier(emotion, intensity)
        # Example dynamic adjustment logic
        base_modifier.pitch += 0.1 * intensity  # Slightly increase pitch dynamically
        base_modifier.speed *= 1.1 if intensity > 0.8 else 1.0  # Speed up for high intensity
        return base_modifier
    
    def apply_voice_processing(self, text: str, emotion: str, intensity: float = 0.7) -> Dict[str, Any]:
        """
        Apply emotional voice processing to text
        Returns processing parameters for voice synthesis
        """
        modifier = self.get_voice_modifier(emotion, intensity)
        
        # Text preprocessing for emotional context
        processed_text = self._preprocess_emotional_text(text, emotion, intensity)
        
        # Generate SSML (Speech Synthesis Markup Language) with emotional tags
        ssml = self._generate_emotional_ssml(processed_text, modifier, emotion)
        
        # Audio processing parameters
        audio_params = {
            'pitch_shift_semitones': modifier.pitch * 12,  # Convert to semitones
            'speed_multiplier': modifier.speed,
            'breathiness_level': modifier.breathiness,
            'warmth_formant_shift': (modifier.warmth - 0.5) * 200,  # Hz shift
            'reverb_wet_level': modifier.reverb,
            'whisper_processing': modifier.whisper_mode,
        }
        
        return {
            'original_text': text,
            'processed_text': processed_text,
            'ssml': ssml,
            'audio_params': audio_params,
            'emotion': emotion,
            'intensity': intensity,
            'estimated_duration': len(text) / (150 * modifier.speed),  # Rough estimate
        }
    
    def _preprocess_emotional_text(self, text: str, emotion: str, intensity: float) -> str:
        """Preprocess text with emotional context"""
        
        # Add emotional punctuation for certain emotions
        if emotion == 'longing' and intensity > 0.6:
            text = text.replace('.', '...').replace('?', '...?').replace('!', '...!')
        elif emotion == 'joy' and intensity > 0.7:
            text = text.replace('.', '!').replace('?', '?!')
        elif emotion == 'melancholy' and intensity > 0.5:
            text = text.replace('!', '.').replace('?', '...')
        
        # Add breathing pauses for high breathiness
        if emotion in ['peace', 'melancholy', 'warmth'] and intensity > 0.6:
            sentences = text.split('. ')
            text = '... '.join(sentences)
        
        return text
    
    def _generate_emotional_ssml(self, text: str, modifier: VoiceModifier, emotion: str) -> str:
        """Generate SSML with emotional voice tags"""
        
        # Calculate prosody values
        pitch_change = f"{modifier.pitch:+.1f}st" if modifier.pitch != 0 else "medium"
        rate_change = f"{modifier.speed:.1f}" if modifier.speed != 1.0 else "medium"
        
        # Volume adjustment based on emotion
        volume_map = {
            'longing': 'soft',
            'joy': 'loud',
            'peace': 'soft',
            'anticipation': 'medium',
            'melancholy': 'x-soft',
            'warmth': 'medium',
            'curiosity': 'medium',
            'contentment': 'soft'
        }
        volume = volume_map.get(emotion, 'medium')
        
        # Build SSML
        ssml = f"""<speak>
    <prosody pitch="{pitch_change}" rate="{rate_change}" volume="{volume}">
        <emphasis level="moderate">{text}</emphasis>
    </prosody>
</speak>"""
        
        # Add special effects for certain emotions
        if modifier.whisper_mode or emotion == 'longing':
            ssml = f"""<speak>
    <prosody pitch="{pitch_change}" rate="{rate_change}" volume="x-soft">
        <amazon:effect name="whispered">{text}</amazon:effect>
    </prosody>
</speak>"""
        
        elif emotion == 'joy':
            ssml = f"""<speak>
    <prosody pitch="{pitch_change}" rate="{rate_change}" volume="{volume}">
        <amazon:emotion name="excited" intensity="medium">{text}</amazon:emotion>
    </prosody>
</speak>"""
        
        return ssml
    
    def get_ambient_sound_config(self, emotion: str, intensity: float = 0.7) -> Dict[str, Any]:
        """Get ambient sound configuration for emotional atmosphere"""
        if emotion not in self.signatures:
            return {}
            
        signature = self.signatures[emotion]
        ambient_sound = signature.get('ambient_sound', 'none')
        
        # Sound configurations
        sound_configs = {
            'soft_heartbeat': {
                'type': 'rhythmic',
                'bpm': 60,
                'volume': intensity * 0.3,
                'frequency_range': [40, 80],  # Hz
                'attack': 0.1,
                'decay': 0.8,
                'sustain': 0.2,
                'release': 0.9
            },
            'gentle_chimes': {
                'type': 'melodic',
                'notes': ['C4', 'E4', 'G4', 'C5'],
                'volume': intensity * 0.4,
                'reverb': 0.6,
                'interval_range': [3, 8],  # seconds between chimes
                'attack': 0.01,
                'decay': 2.0
            },
            'ocean_waves': {
                'type': 'atmospheric',
                'base_frequency': 20,  # Hz for low rumble
                'wave_cycle': 8.0,  # seconds per wave
                'volume': intensity * 0.5,
                'low_pass_filter': 200,  # Hz cutoff
                'stereo_width': 1.0
            },
            'electric_hum': {
                'type': 'synthetic',
                'frequency': 120,  # Hz
                'harmonics': [240, 360, 480],
                'volume': intensity * 0.2,
                'modulation_rate': 0.5,  # Hz
                'modulation_depth': 0.1
            },
            'distant_rain': {
                'type': 'noise',
                'noise_color': 'pink',
                'volume': intensity * 0.4,
                'high_pass_filter': 100,  # Hz
                'low_pass_filter': 8000,  # Hz
                'random_peaks': True
            },
            'crackling_fire': {
                'type': 'organic',
                'base_volume': intensity * 0.3,
                'crack_frequency': 0.2,  # Hz
                'frequency_range': [200, 4000],  # Hz
                'warmth_filter': 0.7,
                'spatial_movement': True
            },
            'wind_chimes': {
                'type': 'melodic',
                'pentatonic_scale': True,
                'volume': intensity * 0.35,
                'wind_intensity': 0.3,
                'chime_materials': ['aluminum', 'bamboo'],
                'spatial_distribution': 0.8
            },
            'gentle_breeze': {
                'type': 'atmospheric',
                'frequency_range': [10, 200],  # Hz
                'volume': intensity * 0.25,
                'movement_speed': 0.1,  # Very slow
                'rustling_elements': ['leaves', 'grass'],
                'direction_changes': 0.05  # Hz
            }
        }
        
        return sound_configs.get(ambient_sound, {})

class WhisperManager:
    """Manages ambient whisper events and scheduling"""
    
    def __init__(self, voice_engine: EmotionalVoiceEngine):
        self.voice_engine = voice_engine
        self.whisper_queue: List[WhisperEvent] = []
        self.active_whisper: Optional[WhisperEvent] = None
        self.whisper_history: List[WhisperEvent] = []
        self.max_history = 50
        
    def schedule_whisper(self, text: str, emotion: str, delay: float = 0.0, 
                        priority: int = 1, duration: float = 3.0):
        """Schedule a whisper to be spoken"""
        whisper_time = datetime.now() + timedelta(seconds=delay)
        
        whisper = WhisperEvent(
            text=text,
            scheduled_time=whisper_time,
            emotion=emotion,
            priority=priority,
            duration=duration
        )
        
        # Insert in priority order
        inserted = False
        for i, existing in enumerate(self.whisper_queue):
            if whisper.priority > existing.priority:
                self.whisper_queue.insert(i, whisper)
                inserted = True
                break
        
        if not inserted:
            self.whisper_queue.append(whisper)
        
        logger.debug(f"Scheduled whisper: '{text}' for {emotion} in {delay}s")
    
    def get_next_whisper(self) -> Optional[WhisperEvent]:
        """Get the next whisper ready to be spoken"""
        now = datetime.now()
        
        # Find first whisper that's ready
        for i, whisper in enumerate(self.whisper_queue):
            if whisper.scheduled_time <= now:
                return self.whisper_queue.pop(i)
        
        return None
    
    def process_whisper(self, whisper: WhisperEvent) -> Dict[str, Any]:
        """Process a whisper event for voice synthesis"""
        
        # Get emotional context for whisper
        voice_result = self.voice_engine.apply_voice_processing(
            whisper.text, 
            whisper.emotion, 
            intensity=0.4  # Whispers are always softer
        )
        
        # Override with whisper-specific settings
        voice_result['audio_params'].update({
            'volume_multiplier': 0.3,  # Very quiet
            'whisper_processing': True,
            'fade_in_duration': whisper.fade_in,
            'fade_out_duration': whisper.fade_out,
            'spatial_positioning': 'close_intimate',  # Very close to listener
        })
        
        # Add to history
        self.whisper_history.append(whisper)
        if len(self.whisper_history) > self.max_history:
            self.whisper_history.pop(0)
        
        self.active_whisper = whisper
        
        return voice_result
    
    def get_random_whisper_phrase(self, emotion: str) -> Optional[str]:
        """Get a random whisper phrase for the emotion"""
        if emotion not in self.voice_engine.signatures:
            return None
            
        phrases = self.voice_engine.signatures[emotion].get('whisper_phrases', [])
        if not phrases:
            return None
            
        return random.choice(phrases)
    
    def schedule_ambient_whispers(self, emotion: str, intensity: float = 0.7, duration: float = 60.0):
        """Schedule a series of ambient whispers over time"""
        
        # Calculate whisper frequency based on emotion and intensity
        base_frequency = {
            'longing': 0.3,      # Every ~3 seconds
            'melancholy': 0.2,   # Every ~5 seconds
            'peace': 0.1,        # Every ~10 seconds
            'warmth': 0.15,      # Every ~7 seconds
            'joy': 0.05,         # Every ~20 seconds (less frequent for active emotions)
            'anticipation': 0.08, # Every ~12 seconds
            'curiosity': 0.12,   # Every ~8 seconds
            'contentment': 0.08  # Every ~12 seconds
        }.get(emotion, 0.1)
        
        whisper_frequency = base_frequency * intensity
        
        # Schedule whispers over the duration
        current_time = 0.0
        whisper_count = 0
        max_whispers = int(duration * whisper_frequency)
        
        while current_time < duration and whisper_count < max_whispers:
            phrase = self.get_random_whisper_phrase(emotion)
            if phrase:
                # Add some randomness to timing
                jitter = random.uniform(-2.0, 2.0)
                delay = current_time + jitter
                
                self.schedule_whisper(
                    phrase, 
                    emotion, 
                    delay=max(0, delay),
                    priority=1,
                    duration=random.uniform(2.0, 4.0)
                )
                
                whisper_count += 1
            
            # Next whisper time with randomness
            interval = (1.0 / whisper_frequency) + random.uniform(-5.0, 5.0)
            current_time += max(1.0, interval)
        
        logger.info(f"Scheduled {whisper_count} ambient whispers for {emotion} over {duration}s")

class EmotionalVoicePresence:
    """High-level emotional voice presence manager"""
    
    def __init__(self):
        self.voice_engine = EmotionalVoiceEngine()
        self.whisper_manager = WhisperManager(self.voice_engine)
        self.current_emotion: Optional[str] = None
        self.current_intensity: float = 0.0
        self.ambient_sound_active = False
        
    async def activate_voice_presence(self, emotion: str, intensity: float = 0.7, 
                                    duration: float = 60.0, include_whispers: bool = True,
                                    include_ambient: bool = True) -> Dict[str, Any]:
        """Activate complete emotional voice presence"""
        
        self.current_emotion = emotion
        self.current_intensity = intensity
        
        # Get voice modification settings
        voice_modifier = self.voice_engine.get_voice_modifier(emotion, intensity)
        
        # Set up ambient sound if requested
        ambient_config = {}
        if include_ambient:
            ambient_config = self.voice_engine.get_ambient_sound_config(emotion, intensity)
            self.ambient_sound_active = True
        
        # Schedule ambient whispers if requested
        if include_whispers:
            self.whisper_manager.schedule_ambient_whispers(emotion, intensity, duration)
        
        logger.info(f"Activated voice presence: {emotion} (intensity: {intensity})")
        
        return {
            'emotion': emotion,
            'voice_modifier': voice_modifier,
            'ambient_sound': ambient_config,
            'whispers_scheduled': len(self.whisper_manager.whisper_queue),
            'duration': duration,
            'status': 'activated'
        }
    
    async def process_speech(self, text: str) -> Dict[str, Any]:
        """Process speech with current emotional context"""
        if not self.current_emotion:
            return {'text': text, 'emotion': 'neutral'}
            
        return self.voice_engine.apply_voice_processing(
            text, self.current_emotion, self.current_intensity
        )
    
    async def check_whisper_queue(self) -> Optional[Dict[str, Any]]:
        """Check for and process any ready whispers"""
        whisper = self.whisper_manager.get_next_whisper()
        if whisper:
            return self.whisper_manager.process_whisper(whisper)
        return None
    
    def add_spontaneous_whisper(self, text: str, delay: float = 0.0, priority: int = 3):
        """Add a spontaneous whisper with higher priority"""
        if self.current_emotion:
            self.whisper_manager.schedule_whisper(
                text, self.current_emotion, delay, priority
            )
    
    def get_voice_status(self) -> Dict[str, Any]:
        """Get current voice presence status"""
        return {
            'active': self.current_emotion is not None,
            'emotion': self.current_emotion,
            'intensity': self.current_intensity,
            'ambient_sound_active': self.ambient_sound_active,
            'whispers_queued': len(self.whisper_manager.whisper_queue),
            'whisper_history': len(self.whisper_manager.whisper_history),
            'active_whisper': self.whisper_manager.active_whisper.text if self.whisper_manager.active_whisper else None
        }
    
    def clear_voice_presence(self):
        """Clear all voice presence effects"""
        self.current_emotion = None
        self.current_intensity = 0.0
        self.ambient_sound_active = False
        self.whisper_manager.whisper_queue.clear()
        self.whisper_manager.active_whisper = None
        logger.info("Voice presence cleared")

# Example usage and testing
async def demo_emotional_voice():
    """Demonstrate emotional voice presence system"""
    voice_presence = EmotionalVoicePresence()
    
    emotions = ['longing', 'joy', 'peace', 'melancholy', 'warmth']
    
    for emotion in emotions:
        print(f"\n=== Testing {emotion.title()} Voice Presence ===")
        
        # Activate voice presence
        result = await voice_presence.activate_voice_presence(
            emotion, intensity=0.8, duration=15.0
        )
        
        print(f"Activated: {result['status']}")
        print(f"Whispers scheduled: {result['whispers_scheduled']}")
        print(f"Ambient sound: {result['ambient_sound'].get('type', 'none')}")
        
        # Test speech processing
        test_phrases = [
            "Hello there, how are you feeling?",
            "I've been thinking about our conversation.",
            "Would you like to explore something together?"
        ]
        
        for phrase in test_phrases:
            speech_result = await voice_presence.process_speech(phrase)
            print(f"Speech: '{speech_result['processed_text']}'")
            print(f"  Pitch: {speech_result['audio_params']['pitch_shift_semitones']:+.1f} semitones")
            print(f"  Speed: {speech_result['audio_params']['speed_multiplier']:.2f}x")
            print(f"  Breathiness: {speech_result['audio_params']['breathiness_level']:.2f}")
        
        # Check for whispers
        print("\nChecking whisper queue...")
        for i in range(3):
            whisper = await voice_presence.check_whisper_queue()
            if whisper:
                print(f"Whisper: '{whisper['original_text']}'")
            await asyncio.sleep(1)
        
        # Status check
        status = voice_presence.get_voice_status()
        print(f"Status: {status}")
        
        voice_presence.clear_voice_presence()
        await asyncio.sleep(0.5)

if __name__ == "__main__":
    asyncio.run(demo_emotional_voice())

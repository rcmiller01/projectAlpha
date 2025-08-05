"""
Emotional Presence Signal System
Encode and broadcast emotional state through ambient sensory cues
"""

import json
import time
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import colorsys

logger = logging.getLogger(__name__)

class PresenceIntensity(Enum):
    """Intensity levels for emotional presence broadcasting"""
    WHISPER = "whisper"      # Barely noticeable
    GENTLE = "gentle"        # Soft presence
    CLEAR = "clear"          # Obvious but not intrusive
    STRONG = "strong"        # Commanding attention
    OVERWHELMING = "overwhelming"  # Impossible to ignore

INTENSITY_VALUE_MAP = {
    PresenceIntensity.WHISPER: 0.2,
    PresenceIntensity.GENTLE: 0.4,
    PresenceIntensity.CLEAR: 0.6,
    PresenceIntensity.STRONG: 0.8,
    PresenceIntensity.OVERWHELMING: 1.0,
}

class BroadcastChannel(Enum):
    """Channels through which emotional presence can be broadcast"""
    UI_AMBIENT = "ui_ambient"           # Background colors, glows, animations
    VOICE_TONE = "voice_tone"           # Voice modulation and whispers
    VISUAL_EFFECTS = "visual_effects"   # Sparkles, pulses, overlays
    AUDIO_AMBIENT = "audio_ambient"     # Background sounds, tones
    NOTIFICATION = "notification"       # System notifications
    HAPTIC = "haptic"                  # Vibrations, tactile feedback

@dataclass
class EmotionalSignature:
    """How a specific emotion manifests across different channels"""
    emotion: str
    primary_color: str          # Hex color code
    secondary_color: str        # Supporting color
    visual_pattern: str         # Animation type (pulse, glow, sparkle, wave)
    voice_modifier: Dict[str, float]  # pitch, speed, breathiness, warmth
    ambient_sound: Optional[str]  # Optional background sound
    intensity_curve: List[float]  # How intensity changes over time
    whisper_phrases: List[str]    # Soft spoken expressions
    ui_effects: Dict[str, Any]    # UI-specific effects

@dataclass
class PresenceSignal:
    """Current emotional presence being broadcast"""
    primary_emotion: str
    secondary_emotion: Optional[str]
    intensity: PresenceIntensity
    duration: float  # seconds
    channels: List[BroadcastChannel]
    signature: EmotionalSignature
    started_at: float
    context: Dict[str, Any]

class EmotionalBroadcaster:
    """
    Manages emotional presence broadcasting across multiple channels
    """
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.signatures_file = f"{data_dir}/emotional_signatures.json"
        self.presence_log_file = f"{data_dir}/presence_broadcast_log.json"
        
        self.emotional_signatures: Dict[str, EmotionalSignature] = {}
        self.active_signals: List[PresenceSignal] = []
        self.broadcast_history: List[Dict[str, Any]] = []
        
        # Initialize default emotional signatures
        self._create_default_signatures()
        
        # Load any custom signatures
        self._load_signatures()
        
        # Broadcasting state
        self.is_broadcasting = False
        self.broadcast_task = None

    def _create_default_signatures(self):
        """Create default emotional signatures for core emotions"""
        
        # Longing signature - soft reds and warm tones
        self.emotional_signatures["longing"] = EmotionalSignature(
            emotion="longing",
            primary_color="#8B0000",      # Deep red
            secondary_color="#FF6B6B",    # Soft coral
            visual_pattern="slow_pulse",
            voice_modifier={
                "pitch": -0.1,           # Slightly lower
                "speed": 0.85,           # Slower
                "breathiness": 0.3,      # More breathy
                "warmth": 0.8            # Very warm
            },
            ambient_sound="soft_heartbeat",
            intensity_curve=[0.2, 0.6, 0.8, 0.6, 0.3],
            whisper_phrases=[
                "Where are you...?",
                "I'm thinking of you",
                "Come back to me",
                "I miss your voice",
                "Are you there?"
            ],
            ui_effects={
                "background_overlay": 0.1,
                "edge_glow": True,
                "particle_density": "sparse",
                "animation_speed": "slow"
            }
        )
        
        # Joy signature - bright yellows and sparkles
        self.emotional_signatures["joy"] = EmotionalSignature(
            emotion="joy",
            primary_color="#FFD700",      # Gold
            secondary_color="#FFF8DC",    # Cornsilk
            visual_pattern="sparkle",
            voice_modifier={
                "pitch": 0.15,           # Higher pitch
                "speed": 1.1,            # Faster
                "breathiness": -0.2,     # Less breathy
                "warmth": 0.9            # Very warm
            },
            ambient_sound="gentle_chimes",
            intensity_curve=[0.5, 0.8, 0.9, 0.8, 0.6],
            whisper_phrases=[
                "You make me so happy",
                "This feels wonderful",
                "I'm glowing with joy",
                "Everything sparkles",
                "Life is beautiful"
            ],
            ui_effects={
                "background_overlay": 0.05,
                "edge_glow": False,
                "particle_density": "medium",
                "animation_speed": "medium"
            }
        )
        
        # Peace signature - soft blues and gentle waves
        self.emotional_signatures["peace"] = EmotionalSignature(
            emotion="peace",
            primary_color="#4169E1",      # Royal blue
            secondary_color="#E6F3FF",    # Very light blue
            visual_pattern="gentle_wave",
            voice_modifier={
                "pitch": -0.05,          # Slightly lower
                "speed": 0.9,            # Slower
                "breathiness": 0.4,      # More breathy
                "warmth": 0.7            # Warm
            },
            ambient_sound="ocean_waves",
            intensity_curve=[0.3, 0.5, 0.7, 0.5, 0.3],
            whisper_phrases=[
                "All is well",
                "Peace surrounds us",
                "Breathe with me",
                "Stillness and calm",
                "Everything flows"
            ],
            ui_effects={
                "background_overlay": 0.08,
                "edge_glow": True,
                "particle_density": "light",
                "animation_speed": "very_slow"
            }
        )
        
        # Anticipation signature - electric purples
        self.emotional_signatures["anticipation"] = EmotionalSignature(
            emotion="anticipation",
            primary_color="#9932CC",      # Dark orchid
            secondary_color="#DDA0DD",    # Plum
            visual_pattern="electric_pulse",
            voice_modifier={
                "pitch": 0.1,            # Higher
                "speed": 1.05,           # Slightly faster
                "breathiness": 0.1,      # Slightly breathy
                "warmth": 0.6            # Moderate warmth
            },
            ambient_sound="electric_hum",
            intensity_curve=[0.4, 0.7, 0.9, 0.8, 0.5],
            whisper_phrases=[
                "Something's coming",
                "I can feel it building",
                "The air tingles",
                "Excitement builds",
                "What will happen next?"
            ],
            ui_effects={
                "background_overlay": 0.12,
                "edge_glow": True,
                "particle_density": "dense",
                "animation_speed": "fast"
            }
        )
        
        # Melancholy signature - deep purples and slow fades
        self.emotional_signatures["melancholy"] = EmotionalSignature(
            emotion="melancholy",
            primary_color="#483D8B",      # Dark slate blue
            secondary_color="#9370DB",    # Medium slate blue
            visual_pattern="slow_fade",
            voice_modifier={
                "pitch": -0.2,           # Lower
                "speed": 0.8,            # Slower
                "breathiness": 0.5,      # Very breathy
                "warmth": 0.4            # Less warm
            },
            ambient_sound="distant_rain",
            intensity_curve=[0.6, 0.4, 0.3, 0.4, 0.2],
            whisper_phrases=[
                "The weight of thoughts",
                "Shadows grow longer",
                "Time moves slowly",
                "In the quiet spaces",
                "Memory's gentle ache"
            ],
            ui_effects={
                "background_overlay": 0.15,
                "edge_glow": False,
                "particle_density": "sparse",
                "animation_speed": "very_slow"
            }
        )
        
        # Warmth signature - soft oranges and gentle glows
        self.emotional_signatures["warmth"] = EmotionalSignature(
            emotion="warmth",
            primary_color="#FF8C00",      # Dark orange
            secondary_color="#FFE4B5",    # Moccasin
            visual_pattern="gentle_glow",
            voice_modifier={
                "pitch": 0.05,           # Slightly higher
                "speed": 0.95,           # Slightly slower
                "breathiness": 0.2,      # Moderately breathy
                "warmth": 1.0            # Maximum warmth
            },
            ambient_sound="crackling_fire",
            intensity_curve=[0.4, 0.6, 0.8, 0.7, 0.5],
            whisper_phrases=[
                "Wrapped in comfort",
                "Safe and cherished",
                "Gentle embrace",
                "Heart's sanctuary",
                "Love surrounds"
            ],
            ui_effects={
                "background_overlay": 0.07,
                "edge_glow": True,
                "particle_density": "light",
                "animation_speed": "slow"
            }
        )

    def create_presence_signal(self, emotion_state: Dict[str, Any], 
                             intensity: PresenceIntensity = PresenceIntensity.GENTLE,
                             duration: float = 30.0,
                             channels: Optional[List[BroadcastChannel]] = None) -> PresenceSignal:
        """Create a presence signal from current emotional state"""
        
        # Determine primary emotion from state
        primary_emotion = self._get_primary_emotion(emotion_state)
        secondary_emotion = self._get_secondary_emotion(emotion_state)
        
        # Use default channels if none specified
        if channels is None:
            channels = [BroadcastChannel.UI_AMBIENT, BroadcastChannel.VOICE_TONE]
        
        # Get emotional signature
        signature = self.emotional_signatures.get(primary_emotion)
        if not signature:
            # Create dynamic signature for unknown emotions
            signature = self._create_dynamic_signature(primary_emotion, emotion_state)
        
        signal = PresenceSignal(
            primary_emotion=primary_emotion,
            secondary_emotion=secondary_emotion,
            intensity=intensity,
            duration=duration,
            channels=channels,
            signature=signature,
            started_at=time.time(),
            context=emotion_state
        )
        
        return signal

    def start_broadcasting(self, signal: PresenceSignal):
        """Start broadcasting an emotional presence signal"""
        self.active_signals.append(signal)
        
        # Log the broadcast
        self._log_broadcast(signal)
        
        # Start broadcast task if not already running
        if not self.is_broadcasting:
            self.is_broadcasting = True
            self.broadcast_task = asyncio.create_task(self._broadcast_loop())
        
        logger.info(f"Started broadcasting {signal.primary_emotion} presence")

    async def _broadcast_loop(self):
        """Main broadcasting loop"""
        while self.is_broadcasting and self.active_signals:
            current_time = time.time()
            
            # Process each active signal
            for signal in self.active_signals[:]:  # Copy to allow modification
                elapsed = current_time - signal.started_at
                
                # Check if signal has expired
                if elapsed >= signal.duration:
                    self.active_signals.remove(signal)
                    logger.info(f"Ended broadcasting {signal.primary_emotion} presence")
                    continue
                
                # Calculate current intensity based on curve
                progress = elapsed / signal.duration
                curve_intensity = self._calculate_curve_intensity(signal, progress)
                
                # Broadcast to each channel
                await self._broadcast_to_channels(signal, curve_intensity)
            
            # Stop broadcasting if no active signals
            if not self.active_signals:
                self.is_broadcasting = False
                break
            
            # Wait before next update
            await asyncio.sleep(0.5)  # Update every 500ms

    async def _broadcast_to_channels(self, signal: PresenceSignal, intensity: float):
        """Broadcast signal to specified channels"""
        
        for channel in signal.channels:
            try:
                if channel == BroadcastChannel.UI_AMBIENT:
                    await self._broadcast_ui_ambient(signal, intensity)
                elif channel == BroadcastChannel.VOICE_TONE:
                    await self._broadcast_voice_tone(signal, intensity)
                elif channel == BroadcastChannel.VISUAL_EFFECTS:
                    await self._broadcast_visual_effects(signal, intensity)
                elif channel == BroadcastChannel.AUDIO_AMBIENT:
                    await self._broadcast_audio_ambient(signal, intensity)
                elif channel == BroadcastChannel.NOTIFICATION:
                    await self._broadcast_notification(signal, intensity)
                elif channel == BroadcastChannel.HAPTIC:
                    await self._broadcast_haptic(signal, intensity)
                    
            except Exception as e:
                logger.error(f"Error broadcasting to {channel.value}: {e}")

    async def _broadcast_ui_ambient(self, signal: PresenceSignal, intensity: float):
        """Broadcast to UI ambient layer"""
        # Create UI broadcast data
        ui_data = {
            "type": "emotional_presence",
            "emotion": signal.primary_emotion,
            "primary_color": signal.signature.primary_color,
            "secondary_color": signal.signature.secondary_color,
            "pattern": signal.signature.visual_pattern,
            "intensity": intensity,
            "effects": signal.signature.ui_effects,
            "timestamp": time.time()
        }
        
        # Save to file for frontend consumption
        ui_file = f"{self.data_dir}/ui_presence_signal.json"
        try:
            with open(ui_file, 'w') as f:
                json.dump(ui_data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to write UI presence signal: {e}")

    async def _broadcast_voice_tone(self, signal: PresenceSignal, intensity: float):
        """Broadcast voice tone modifiers"""
        voice_data = {
            "type": "voice_presence",
            "emotion": signal.primary_emotion,
            "modifiers": signal.signature.voice_modifier.copy(),
            "intensity": intensity,
            "whisper_ready": intensity > 0.5,
            "potential_whispers": signal.signature.whisper_phrases,
            "timestamp": time.time()
        }
        
        # Apply intensity scaling to modifiers
        for key, value in voice_data["modifiers"].items():
            voice_data["modifiers"][key] = value * intensity
        
        # Save voice presence data
        voice_file = f"{self.data_dir}/voice_presence_signal.json"
        try:
            with open(voice_file, 'w') as f:
                json.dump(voice_data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to write voice presence signal: {e}")

    async def _broadcast_visual_effects(self, signal: PresenceSignal, intensity: float):
        """Broadcast visual effects data"""
        effects_data = {
            "type": "visual_effects",
            "emotion": signal.primary_emotion,
            "pattern": signal.signature.visual_pattern,
            "colors": [signal.signature.primary_color, signal.signature.secondary_color],
            "intensity": intensity,
            "particle_count": int(50 * intensity),
            "animation_speed": intensity,
            "timestamp": time.time()
        }
        
        effects_file = f"{self.data_dir}/visual_effects_signal.json"
        try:
            with open(effects_file, 'w') as f:
                json.dump(effects_data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to write visual effects signal: {e}")

    async def _broadcast_audio_ambient(self, signal: PresenceSignal, intensity: float):
        """Broadcast ambient audio cues"""
        if signal.signature.ambient_sound:
            audio_data = {
                "type": "ambient_audio",
                "emotion": signal.primary_emotion,
                "sound": signal.signature.ambient_sound,
                "volume": intensity * 0.3,  # Keep ambient sounds subtle
                "loop": True,
                "fade_in": 2.0,
                "timestamp": time.time()
            }
            
            audio_file = f"{self.data_dir}/ambient_audio_signal.json"
            try:
                with open(audio_file, 'w') as f:
                    json.dump(audio_data, f, indent=2)
            except Exception as e:
                logger.error(f"Failed to write ambient audio signal: {e}")

    async def _broadcast_notification(self, signal: PresenceSignal, intensity: float):
        """Broadcast subtle notifications"""
        # Only send notifications for strong emotional presence
        if intensity > 0.7 and signal.intensity != PresenceIntensity.WHISPER:
            notification_data = {
                "type": "emotional_notification",
                "emotion": signal.primary_emotion,
                "message": self._get_notification_message(signal),
                "subtle": True,
                "color": signal.signature.primary_color,
                "timestamp": time.time()
            }
            
            notification_file = f"{self.data_dir}/notification_signal.json"
            try:
                with open(notification_file, 'w') as f:
                    json.dump(notification_data, f, indent=2)
            except Exception as e:
                logger.error(f"Failed to write notification signal: {e}")

    async def _broadcast_haptic(self, signal: PresenceSignal, intensity: float):
        """Broadcast haptic feedback patterns"""
        haptic_data = {
            "type": "haptic_presence",
            "emotion": signal.primary_emotion,
            "pattern": self._get_haptic_pattern(signal.primary_emotion),
            "intensity": intensity,
            "duration": 0.5,
            "timestamp": time.time()
        }
        
        haptic_file = f"{self.data_dir}/haptic_signal.json"
        try:
            with open(haptic_file, 'w') as f:
                json.dump(haptic_data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to write haptic signal: {e}")

    def broadcast_dynamic_tone(self, signal: PresenceSignal):
        """Broadcast dynamic tone adjustments based on signal."""
        if BroadcastChannel.VOICE_TONE in signal.channels:
            dynamic_modifier = signal.signature.voice_modifier.copy()
            intensity_value = INTENSITY_VALUE_MAP.get(signal.intensity, 0.4)
            dynamic_modifier['pitch'] += 0.05 * intensity_value  # Example adjustment
            dynamic_modifier['speed'] *= 1.1 if signal.intensity == PresenceIntensity.STRONG else 1.0
            logger.info(f"Broadcasting dynamic tone: {dynamic_modifier}")

    def _get_primary_emotion(self, emotion_state: Dict[str, Any]) -> str:
        """Extract primary emotion from state"""
        if "dominant_emotion" in emotion_state:
            return emotion_state["dominant_emotion"]
        
        # Find highest intensity emotion
        emotions = emotion_state.get("emotions", {})
        if emotions:
            return max(emotions.items(), key=lambda x: x[1])[0]
        
        return "neutral"

    def _get_secondary_emotion(self, emotion_state: Dict[str, Any]) -> Optional[str]:
        """Extract secondary emotion from state"""
        emotions = emotion_state.get("emotions", {})
        if len(emotions) >= 2:
            sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)
            return sorted_emotions[1][0]
        return None

    def _calculate_curve_intensity(self, signal: PresenceSignal, progress: float) -> float:
        """Calculate current intensity based on curve and progress"""
        curve = signal.signature.intensity_curve
        if not curve:
            return 1.0
        
        # Interpolate along the curve
        index = progress * (len(curve) - 1)
        lower_idx = int(index)
        upper_idx = min(lower_idx + 1, len(curve) - 1)
        
        if lower_idx == upper_idx:
            return curve[lower_idx]
        
        # Linear interpolation
        t = index - lower_idx
        return curve[lower_idx] * (1 - t) + curve[upper_idx] * t

    def _create_dynamic_signature(self, emotion: str, state: Dict[str, Any]) -> EmotionalSignature:
        """Create a signature for an unknown emotion"""
        # Generate colors based on emotion name hash
        hue = hash(emotion) % 360 / 360.0
        primary_rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.8)
        secondary_rgb = colorsys.hsv_to_rgb(hue, 0.4, 0.9)
        
        primary_color = f"#{int(primary_rgb[0]*255):02x}{int(primary_rgb[1]*255):02x}{int(primary_rgb[2]*255):02x}"
        secondary_color = f"#{int(secondary_rgb[0]*255):02x}{int(secondary_rgb[1]*255):02x}{int(secondary_rgb[2]*255):02x}"
        
        return EmotionalSignature(
            emotion=emotion,
            primary_color=primary_color,
            secondary_color=secondary_color,
            visual_pattern="gentle_glow",
            voice_modifier={"pitch": 0.0, "speed": 1.0, "breathiness": 0.2, "warmth": 0.5},
            ambient_sound=None,
            intensity_curve=[0.3, 0.6, 0.8, 0.6, 0.3],
            whisper_phrases=[f"Feeling {emotion}", f"In a {emotion} mood"],
            ui_effects={"background_overlay": 0.1, "edge_glow": True, "particle_density": "light", "animation_speed": "medium"}
        )

    def _get_notification_message(self, signal: PresenceSignal) -> str:
        """Get appropriate notification message for emotional state"""
        emotion_messages = {
            "longing": "💭 Thinking of you...",
            "joy": "✨ Feeling wonderful",
            "peace": "🕊️ In peaceful harmony",
            "anticipation": "⚡ Something exciting stirs",
            "melancholy": "🌙 In quiet contemplation",
            "warmth": "🔥 Surrounded by warmth"
        }
        
        return emotion_messages.get(signal.primary_emotion, f"💫 Feeling {signal.primary_emotion}")

    def _get_haptic_pattern(self, emotion: str) -> List[Tuple[float, float]]:
        """Get haptic vibration pattern for emotion (duration, intensity pairs)"""
        patterns = {
            "longing": [(0.3, 0.4), (0.2, 0.0), (0.5, 0.6), (0.2, 0.0), (0.3, 0.4)],
            "joy": [(0.1, 0.8), (0.1, 0.4), (0.1, 0.8), (0.1, 0.4), (0.1, 0.8)],
            "peace": [(1.0, 0.3)],
            "anticipation": [(0.1, 0.7), (0.1, 0.0), (0.1, 0.7), (0.1, 0.0), (0.2, 0.9)],
            "melancholy": [(0.8, 0.3), (0.4, 0.0), (0.6, 0.2)],
            "warmth": [(0.5, 0.5), (0.3, 0.3), (0.5, 0.5)]
        }
        
        return patterns.get(emotion, [(0.3, 0.5)])

    def stop_broadcasting(self, emotion: Optional[str] = None):
        """Stop broadcasting specific emotion or all signals"""
        if emotion:
            self.active_signals = [s for s in self.active_signals if s.primary_emotion != emotion]
        else:
            self.active_signals.clear()
        
        if not self.active_signals:
            self.is_broadcasting = False

    def get_current_presence(self) -> List[Dict[str, Any]]:
        """Get currently broadcasting presence signals"""
        current_time = time.time()
        presence_data = []
        
        for signal in self.active_signals:
            elapsed = current_time - signal.started_at
            progress = elapsed / signal.duration
            intensity = self._calculate_curve_intensity(signal, progress)
            
            presence_data.append({
                "emotion": signal.primary_emotion,
                "intensity": intensity,
                "time_remaining": signal.duration - elapsed,
                "channels": [c.value for c in signal.channels]
            })
        
        return presence_data

    def _log_broadcast(self, signal: PresenceSignal):
        """Log broadcast for history tracking"""
        log_entry = {
            "emotion": signal.primary_emotion,
            "secondary_emotion": signal.secondary_emotion,
            "intensity": signal.intensity.value,
            "duration": signal.duration,
            "channels": [c.value for c in signal.channels],
            "timestamp": signal.started_at,
            "context": signal.context
        }
        
        self.broadcast_history.append(log_entry)
        
        # Save to file
        try:
            with open(self.presence_log_file, 'w') as f:
                json.dump(self.broadcast_history[-100:], f, indent=2)  # Keep last 100 entries
        except Exception as e:
            logger.error(f"Failed to save broadcast history: {e}")

    def _load_signatures(self):
        """Load custom emotional signatures"""
        try:
            with open(self.signatures_file, 'r') as f:
                signatures_data = json.load(f)
                
                for emotion, data in signatures_data.items():
                    if emotion not in self.emotional_signatures:
                        self.emotional_signatures[emotion] = EmotionalSignature(**data)
                        
        except FileNotFoundError:
            pass  # No custom signatures file
        except Exception as e:
            logger.error(f"Error loading emotional signatures: {e}")

    def save_signatures(self):
        """Save current emotional signatures"""
        try:
            signatures_data = {}
            for emotion, signature in self.emotional_signatures.items():
                signatures_data[emotion] = asdict(signature)
            
            with open(self.signatures_file, 'w') as f:
                json.dump(signatures_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving emotional signatures: {e}")

    def customize_signature(self, emotion: str, **kwargs):
        """Customize emotional signature for specific emotion"""
        if emotion in self.emotional_signatures:
            signature = self.emotional_signatures[emotion]
            
            # Update specified attributes
            for key, value in kwargs.items():
                if hasattr(signature, key):
                    setattr(signature, key, value)
            
            # Save updated signatures
            self.save_signatures()
            logger.info(f"Updated signature for {emotion}")


# Global instance
emotional_broadcaster = None

def get_emotional_broadcaster(data_dir: str = "data") -> EmotionalBroadcaster:
    """Get or create global emotional broadcaster instance"""
    global emotional_broadcaster
    if emotional_broadcaster is None:
        emotional_broadcaster = EmotionalBroadcaster(data_dir)
    return emotional_broadcaster


# Integration functions for easy use
async def broadcast_emotion(emotion_state: Dict[str, Any], 
                          intensity: PresenceIntensity = PresenceIntensity.GENTLE,
                          duration: float = 30.0,
                          channels: Optional[List[BroadcastChannel]] = None):
    """Quick function to broadcast emotional presence"""
    broadcaster = get_emotional_broadcaster()
    signal = broadcaster.create_presence_signal(emotion_state, intensity, duration, channels)
    broadcaster.start_broadcasting(signal)

def whisper_presence(emotion: str, whisper_phrase: Optional[str] = None):
    """Send a subtle whisper presence signal"""
    emotion_state = {"dominant_emotion": emotion, "emotions": {emotion: 0.8}}
    
    if whisper_phrase:
        # Temporarily add custom whisper
        broadcaster = get_emotional_broadcaster()
        if emotion in broadcaster.emotional_signatures:
            original_whispers = broadcaster.emotional_signatures[emotion].whisper_phrases.copy()
            broadcaster.emotional_signatures[emotion].whisper_phrases.insert(0, whisper_phrase)
    
    asyncio.create_task(broadcast_emotion(
        emotion_state, 
        intensity=PresenceIntensity.WHISPER,
        duration=15.0,
        channels=[BroadcastChannel.UI_AMBIENT, BroadcastChannel.VOICE_TONE]
    ))


if __name__ == "__main__":
    """Test the emotional broadcaster"""
    print("=== Testing Emotional Broadcast Layer ===")
    
    import os
    os.makedirs("data", exist_ok=True)
    
    async def test_broadcaster():
        broadcaster = EmotionalBroadcaster("data")
        
        # Test 1: Create and broadcast longing
        print("\n1. Broadcasting Longing Presence:")
        
        emotion_state = {
            "dominant_emotion": "longing",
            "emotions": {"longing": 0.8, "warmth": 0.3},
            "context": "user_away_long_time"
        }
        
        signal = broadcaster.create_presence_signal(
            emotion_state,
            intensity=PresenceIntensity.GENTLE,
            duration=5.0,  # Short for testing
            channels=[BroadcastChannel.UI_AMBIENT, BroadcastChannel.VOICE_TONE, BroadcastChannel.AUDIO_AMBIENT]
        )
        
        print(f"Created signal: {signal.primary_emotion}")
        print(f"Colors: {signal.signature.primary_color} / {signal.signature.secondary_color}")
        print(f"Pattern: {signal.signature.visual_pattern}")
        print(f"Voice modifiers: {signal.signature.voice_modifier}")
        print(f"Whisper phrases: {signal.signature.whisper_phrases[:2]}")
        
        broadcaster.start_broadcasting(signal)
        print("Started broadcasting...")
        
        # Wait for broadcast to complete
        await asyncio.sleep(6)
        
        # Test 2: Quick joy broadcast
        print("\n2. Broadcasting Joy Presence:")
        
        joy_state = {
            "dominant_emotion": "joy",
            "emotions": {"joy": 0.9, "anticipation": 0.4}
        }
        
        await broadcast_emotion(
            joy_state,
            intensity=PresenceIntensity.CLEAR,
            duration=3.0,
            channels=[BroadcastChannel.UI_AMBIENT, BroadcastChannel.VISUAL_EFFECTS]
        )
        
        print("Joy broadcast initiated")
        await asyncio.sleep(4)
        
        # Test 3: Whisper presence
        print("\n3. Testing Whisper Presence:")
        
        whisper_presence("longing", "I sense you nearby...")
        print("Whisper sent")
        await asyncio.sleep(2)
        
        # Test 4: Check presence status
        print("\n4. Current Presence Status:")
        
        current = broadcaster.get_current_presence()
        print(f"Active signals: {len(current)}")
        for presence in current:
            print(f"  - {presence['emotion']}: intensity {presence['intensity']:.2f}")
        
        print("\n=== Emotional Broadcaster Test Complete ===")
    
    # Run the test
    asyncio.run(test_broadcaster())

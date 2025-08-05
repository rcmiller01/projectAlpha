"""
Unified Emotional Broadcast System

This module integrates all emotional presence components into a cohesive
system that can broadcast emotional states across multiple channels
simultaneously: UI, voice, ambient sound, whispers, and external devices.
"""

import asyncio
import json
from typing import Dict, Any, Optional, List, Set, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
import logging
from enum import Enum

# Import our presence modules
from .presence_signal import EmotionalBroadcaster, PresenceIntensity, BroadcastChannel
from .ui_integration import PresenceUIManager
from .voice_integration import EmotionalVoicePresence

logger = logging.getLogger(__name__)

@dataclass
class BroadcastProfile:
    """Configuration profile for emotional broadcasting"""
    name: str
    enabled_channels: Set[str]
    intensity_multiplier: float = 1.0
    ui_effects: bool = True
    voice_modulation: bool = True
    ambient_sounds: bool = True
    whisper_frequency: float = 1.0  # Multiplier for whisper frequency
    external_devices: bool = False
    cross_session_persistence: bool = True

class UnifiedEmotionalBroadcast:
    """
    Unified system for broadcasting emotional presence across all channels
    
    Manages coordination between:
    - UI ambient effects (colors, particles, overlays)
    - Voice tone modulation and whispers
    - Ambient sound generation
    - External device integration (lights, haptics)
    - Cross-session emotional state persistence
    """
    
    def __init__(self, config_path: str = "data/emotional_signatures.json"):
        # Initialize all subsystems
        # Extract data directory from config path
        data_dir = str(Path(config_path).parent)
        
        self.core_broadcaster = EmotionalBroadcaster(data_dir)
        self.ui_manager = PresenceUIManager()
        self.voice_presence = EmotionalVoicePresence()
        
        # System state
        self.active_broadcasts: Dict[str, Dict[str, Any]] = {}
        self.broadcast_profiles = self._initialize_profiles()
        self.current_profile = "default"
        self.session_state_file = Path("data/emotional_session_state.json")
        
        # Performance monitoring
        self.performance_metrics = {
            'broadcasts_started': 0,
            'broadcasts_completed': 0,
            'average_duration': 0.0,
            'channels_used': set(),
            'errors_encountered': 0
        }
        
        # Load persistent state
        self.load_session_state()
        
    def _initialize_profiles(self) -> Dict[str, BroadcastProfile]:
        """Initialize broadcast profiles for different use cases"""
        return {
            "default": BroadcastProfile(
                name="Default",
                enabled_channels={"ui_ambient", "voice_tone", "whispers"},
                intensity_multiplier=0.7,
                ui_effects=True,
                voice_modulation=True,
                ambient_sounds=True,
                whisper_frequency=1.0
            ),
            "minimal": BroadcastProfile(
                name="Minimal",
                enabled_channels={"ui_ambient"},
                intensity_multiplier=0.3,
                ui_effects=True,
                voice_modulation=False,
                ambient_sounds=False,
                whisper_frequency=0.2
            ),
            "immersive": BroadcastProfile(
                name="Immersive",
                enabled_channels={"ui_ambient", "voice_tone", "whispers", "ambient_audio", "external_devices"},
                intensity_multiplier=1.0,
                ui_effects=True,
                voice_modulation=True,
                ambient_sounds=True,
                whisper_frequency=1.5,
                external_devices=True
            ),
            "voice_only": BroadcastProfile(
                name="Voice Only",
                enabled_channels={"voice_tone", "whispers"},
                intensity_multiplier=0.8,
                ui_effects=False,
                voice_modulation=True,
                ambient_sounds=False,
                whisper_frequency=1.2
            ),
            "silent": BroadcastProfile(
                name="Silent",
                enabled_channels={"ui_ambient"},
                intensity_multiplier=0.5,
                ui_effects=True,
                voice_modulation=False,
                ambient_sounds=False,
                whisper_frequency=0.0
            )
        }
    
    async def broadcast_emotion(self, emotion: str, intensity: float = 0.7, 
                              duration: float = 60.0, profile: Optional[str] = None,
                              custom_channels: Optional[Set[str]] = None) -> str:
        """
        Start a unified emotional broadcast across all enabled channels
        
        Args:
            emotion: The emotion to broadcast
            intensity: Intensity level (0.0 to 1.0)
            duration: Duration in seconds
            profile: Broadcast profile to use (default uses current profile)
            custom_channels: Override channels for this broadcast
            
        Returns:
            Broadcast ID for tracking and control
        """
        
        # Generate unique broadcast ID
        broadcast_id = f"{emotion}_{datetime.now().isoformat()}"
        
        # Get profile configuration
        profile_name = profile or self.current_profile
        broadcast_profile = self.broadcast_profiles.get(profile_name, self.broadcast_profiles["default"])
        
        # Determine channels to use
        channels = custom_channels or broadcast_profile.enabled_channels
        
        # Apply profile intensity multiplier
        effective_intensity = min(1.0, intensity * broadcast_profile.intensity_multiplier)
        
        logger.info(f"Starting emotional broadcast: {emotion} (intensity: {effective_intensity}, duration: {duration}s)")
        logger.debug(f"Using profile: {profile_name}, channels: {channels}")
        
        # Initialize broadcast tracking
        broadcast_data = {
            'broadcast_id': broadcast_id,
            'emotion': emotion,
            'intensity': effective_intensity,
            'duration': duration,
            'profile': profile_name,
            'channels': list(channels),
            'start_time': datetime.now(),
            'status': 'starting',
            'results': {}
        }
        
        self.active_broadcasts[broadcast_id] = broadcast_data
        
        try:
            # Start core emotional broadcaster
            emotion_state = {
                'emotion': emotion,
                'intensity': effective_intensity,
                'duration': duration
            }
            
            # Convert intensity to PresenceIntensity enum
            if effective_intensity >= 0.9:
                presence_intensity = PresenceIntensity.OVERWHELMING
            elif effective_intensity >= 0.7:
                presence_intensity = PresenceIntensity.STRONG
            elif effective_intensity >= 0.5:
                presence_intensity = PresenceIntensity.CLEAR
            elif effective_intensity >= 0.3:
                presence_intensity = PresenceIntensity.GENTLE
            else:
                presence_intensity = PresenceIntensity.WHISPER
            
            # Convert channels to BroadcastChannel enum
            broadcast_channels = []
            channel_mapping = {
                'ui_ambient': BroadcastChannel.UI_AMBIENT,
                'voice_tone': BroadcastChannel.VOICE_TONE,
                'visual_effects': BroadcastChannel.VISUAL_EFFECTS,
                'ambient_audio': BroadcastChannel.AUDIO_AMBIENT,
                'whispers': BroadcastChannel.VOICE_TONE,  # Whispers use voice channel
                'external_devices': BroadcastChannel.HAPTIC
            }
            
            for channel in channels:
                if channel in channel_mapping:
                    broadcast_channels.append(channel_mapping[channel])
            
            presence_signal = self.core_broadcaster.create_presence_signal(
                emotion_state, presence_intensity, duration, broadcast_channels
            )
            self.core_broadcaster.start_broadcasting(presence_signal)
            broadcast_data['results']['core'] = {
                'status': 'started', 
                'emotion': presence_signal.primary_emotion,
                'intensity': presence_signal.intensity.value,
                'channels': [ch.value for ch in presence_signal.channels]
            }
            
            # Start UI presence if enabled
            if "ui_ambient" in channels and broadcast_profile.ui_effects:
                ui_result = await self.ui_manager.activate_emotional_presence(
                    emotion, effective_intensity, duration
                )
                broadcast_data['results']['ui'] = ui_result
                logger.debug("UI presence activated")
            
            # Start voice presence if enabled
            if any(ch in channels for ch in ["voice_tone", "whispers", "ambient_audio"]):
                voice_result = await self.voice_presence.activate_voice_presence(
                    emotion, 
                    effective_intensity,
                    duration,
                    include_whispers=("whispers" in channels),
                    include_ambient=("ambient_audio" in channels and broadcast_profile.ambient_sounds)
                )
                broadcast_data['results']['voice'] = voice_result
                logger.debug("Voice presence activated")
            
            # External device integration (placeholder for future implementation)
            if "external_devices" in channels and broadcast_profile.external_devices:
                external_result = await self._activate_external_devices(
                    emotion, effective_intensity, duration
                )
                broadcast_data['results']['external'] = external_result
                logger.debug("External devices activated")
            
            # Start monitoring and coordination task
            asyncio.create_task(self._monitor_broadcast(broadcast_id))
            
            broadcast_data['status'] = 'active'
            self.performance_metrics['broadcasts_started'] += 1
            self.performance_metrics['channels_used'].update(channels)
            
            # Save state for cross-session persistence
            if broadcast_profile.cross_session_persistence:
                self.save_session_state()
            
            return broadcast_id
            
        except Exception as e:
            logger.error(f"Error starting emotional broadcast: {e}")
            broadcast_data['status'] = 'failed'
            broadcast_data['error'] = str(e)
            self.performance_metrics['errors_encountered'] += 1
            raise
    
    async def _monitor_broadcast(self, broadcast_id: str):
        """Monitor and coordinate a broadcast across all channels"""
        broadcast_data = self.active_broadcasts.get(broadcast_id)
        if not broadcast_data:
            return
            
        start_time = broadcast_data['start_time']
        duration = broadcast_data['duration']
        
        try:
            while (datetime.now() - start_time).total_seconds() < duration:
                # Update UI frame if active
                if 'ui' in broadcast_data['results']:
                    frame_data = await self.ui_manager.update_presence_frame()
                    if frame_data:
                        broadcast_data['results']['ui']['current_frame'] = frame_data
                
                # Check for voice whispers if active
                if 'voice' in broadcast_data['results']:
                    whisper = await self.voice_presence.check_whisper_queue()
                    if whisper:
                        logger.debug(f"Whisper processed: {whisper.get('original_text', '')}")
                
                # Update core broadcaster - just continue monitoring
                # The core broadcaster handles its own timing internally
                pass
                
                # Brief pause before next update
                await asyncio.sleep(1.0)
            
            # Broadcast completed naturally
            await self._complete_broadcast(broadcast_id, 'completed')
            
        except Exception as e:
            logger.error(f"Error monitoring broadcast {broadcast_id}: {e}")
            await self._complete_broadcast(broadcast_id, 'error')
    
    async def _complete_broadcast(self, broadcast_id: str, status: str):
        """Complete and clean up a broadcast"""
        broadcast_data = self.active_broadcasts.get(broadcast_id)
        if not broadcast_data:
            return
            
        broadcast_data['status'] = status
        broadcast_data['end_time'] = datetime.now()
        duration = (broadcast_data['end_time'] - broadcast_data['start_time']).total_seconds()
        
        # Update metrics
        self.performance_metrics['broadcasts_completed'] += 1
        current_avg = self.performance_metrics['average_duration']
        completed = self.performance_metrics['broadcasts_completed']
        self.performance_metrics['average_duration'] = (current_avg * (completed - 1) + duration) / completed
        
        logger.info(f"Broadcast {broadcast_id} {status} after {duration:.1f}s")
        
        # Clean up active broadcast
        del self.active_broadcasts[broadcast_id]
    
    async def stop_broadcast(self, broadcast_id: str) -> bool:
        """Stop a specific broadcast"""
        if broadcast_id not in self.active_broadcasts:
            logger.warning(f"Broadcast {broadcast_id} not found")
            return False
            
        # Clear all subsystems
        self.ui_manager.clear_emotional_presence()
        self.voice_presence.clear_voice_presence()
        self.core_broadcaster.stop_broadcasting()  # Stop all broadcasting
        
        await self._complete_broadcast(broadcast_id, 'stopped')
        return True
    
    async def stop_all_broadcasts(self):
        """Stop all active broadcasts"""
        broadcast_ids = list(self.active_broadcasts.keys())
        for broadcast_id in broadcast_ids:
            await self.stop_broadcast(broadcast_id)
            
        logger.info("All broadcasts stopped")
    
    def set_profile(self, profile_name: str) -> bool:
        """Set the active broadcast profile"""
        if profile_name not in self.broadcast_profiles:
            logger.warning(f"Unknown broadcast profile: {profile_name}")
            return False
            
        self.current_profile = profile_name
        logger.info(f"Broadcast profile set to: {profile_name}")
        return True
    
    async def process_speech(self, text: str) -> Dict[str, Any]:
        """Process speech through the voice presence system"""
        return await self.voice_presence.process_speech(text)
    
    def add_spontaneous_whisper(self, text: str, delay: float = 0.0, priority: int = 3):
        """Add a spontaneous whisper to the active broadcast"""
        self.voice_presence.add_spontaneous_whisper(text, delay, priority)
    
    async def _activate_external_devices(self, emotion: str, intensity: float, duration: float) -> Dict[str, Any]:
        """Activate external device integration (placeholder)"""
        # This would integrate with smart home devices, LED strips, haptic feedback, etc.
        # For now, return a placeholder response
        return {
            'status': 'simulated',
            'devices': ['smart_lights', 'ambient_speaker'],
            'emotion': emotion,
            'intensity': intensity,
            'note': 'External device integration not yet implemented'
        }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        ui_status = self.ui_manager.get_status()
        voice_status = self.voice_presence.get_voice_status()
        core_status = self.core_broadcaster.get_current_presence()  # Get list of current presence signals
        
        return {
            'active_broadcasts': len(self.active_broadcasts),
            'current_profile': self.current_profile,
            'subsystems': {
                'ui': ui_status,
                'voice': voice_status,
                'core': core_status
            },
            'performance': self.performance_metrics.copy(),
            'available_profiles': list(self.broadcast_profiles.keys()),
            'session_persistence': self.session_state_file.exists()
        }
    
    def get_active_broadcasts(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all active broadcasts"""
        return self.active_broadcasts.copy()
    
    def save_session_state(self):
        """Save current emotional state for cross-session persistence"""
        try:
            state_data = {
                'timestamp': datetime.now().isoformat(),
                'current_profile': self.current_profile,
                'active_broadcasts': {
                    bid: {
                        'emotion': data['emotion'],
                        'intensity': data['intensity'],
                        'start_time': data['start_time'].isoformat(),
                        'duration': data['duration'],
                        'channels': data['channels']
                    }
                    for bid, data in self.active_broadcasts.items()
                },
                'performance_metrics': {
                    **self.performance_metrics,
                    'channels_used': list(self.performance_metrics['channels_used'])
                }
            }
            
            self.session_state_file.parent.mkdir(exist_ok=True)
            with open(self.session_state_file, 'w') as f:
                json.dump(state_data, f, indent=2)
                
            logger.debug("Session state saved")
            
        except Exception as e:
            logger.error(f"Error saving session state: {e}")
    
    def load_session_state(self):
        """Load persistent emotional state from previous session"""
        try:
            if not self.session_state_file.exists():
                return
                
            with open(self.session_state_file, 'r') as f:
                state_data = json.load(f)
            
            # Restore profile
            if 'current_profile' in state_data:
                self.current_profile = state_data['current_profile']
            
            # Restore performance metrics
            if 'performance_metrics' in state_data:
                metrics = state_data['performance_metrics']
                self.performance_metrics.update(metrics)
                if 'channels_used' in metrics:
                    self.performance_metrics['channels_used'] = set(metrics['channels_used'])
            
            logger.info(f"Session state loaded from {state_data.get('timestamp', 'unknown time')}")
            
        except Exception as e:
            logger.error(f"Error loading session state: {e}")

# High-level convenience functions
async def create_emotional_moment(emotion: str, intensity: float = 0.7, 
                                duration: float = 30.0, include_whispers: bool = True) -> str:
    """
    Create a complete emotional moment with default settings
    
    This is the main entry point for creating emotional presence.
    """
    broadcast_system = UnifiedEmotionalBroadcast()
    
    # Choose appropriate profile based on parameters
    if include_whispers and intensity > 0.7:
        profile = "immersive"
    elif intensity < 0.4:
        profile = "minimal"
    else:
        profile = "default"
    
    return await broadcast_system.broadcast_emotion(
        emotion, intensity, duration, profile
    )

async def whisper_to_user(text: str, emotion: str = "warmth", delay: float = 0.0):
    """Send a spontaneous whisper to the user"""
    broadcast_system = UnifiedEmotionalBroadcast()
    broadcast_system.add_spontaneous_whisper(text, delay, priority=4)

async def demo_unified_broadcast():
    """Comprehensive demo of the unified broadcast system"""
    print("🎭 Unified Emotional Broadcast System Demo")
    print("=" * 50)
    
    broadcast_system = UnifiedEmotionalBroadcast()
    
    # Test different profiles
    profiles_to_test = ["minimal", "default", "immersive"]
    emotions_to_test = ["longing", "joy", "peace", "anticipation"]
    
    for profile in profiles_to_test:
        print(f"\n🎯 Testing profile: {profile.upper()}")
        broadcast_system.set_profile(profile)
        
        for emotion in emotions_to_test:
            print(f"\n  💫 Broadcasting {emotion}...")
            
            # Start broadcast
            broadcast_id = await broadcast_system.broadcast_emotion(
                emotion, intensity=0.8, duration=8.0
            )
            
            # Add a spontaneous whisper
            broadcast_system.add_spontaneous_whisper(
                f"Feeling {emotion} with you...", delay=2.0
            )
            
            # Monitor for a few seconds
            await asyncio.sleep(5)
            
            # Show status
            status = broadcast_system.get_system_status()
            print(f"    Active broadcasts: {status['active_broadcasts']}")
            print(f"    UI active: {status['subsystems']['ui']['active']}")
            print(f"    Voice active: {status['subsystems']['voice']['active']}")
            
            # Stop the broadcast
            await broadcast_system.stop_broadcast(broadcast_id)
            await asyncio.sleep(1)
    
    # Final system status
    print(f"\n📊 Final Performance Metrics:")
    final_status = broadcast_system.get_system_status()
    metrics = final_status['performance']
    print(f"  Broadcasts started: {metrics['broadcasts_started']}")
    print(f"  Broadcasts completed: {metrics['broadcasts_completed']}")
    print(f"  Average duration: {metrics['average_duration']:.1f}s")
    print(f"  Channels used: {', '.join(metrics['channels_used'])}")
    print(f"  Errors: {metrics['errors_encountered']}")

if __name__ == "__main__":
    # Run the comprehensive demo
    asyncio.run(demo_unified_broadcast())

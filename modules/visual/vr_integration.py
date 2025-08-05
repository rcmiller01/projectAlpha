# vr_integration.py
# Phase 3: Virtual reality integration for immersive shared experiences

import json
import time
import threading
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import math

class VRSceneType(Enum):
    ROMANTIC_GARDEN = "romantic_garden"
    COZY_HOME = "cozy_home"
    BEACH_SUNSET = "beach_sunset"
    MOUNTAIN_VIEW = "mountain_view"
    STARLIT_SKY = "starlit_sky"
    INTIMATE_BEDROOM = "intimate_bedroom"
    DANCE_FLOOR = "dance_floor"
    COOKING_TOGETHER = "cooking_together"
    WALKING_HAND_IN_HAND = "walking_hand_in_hand"
    MEDITATION_SPACE = "meditation_space"

class VRInteractionType(Enum):
    TOUCH = "touch"
    HUG = "hug"
    KISS = "kiss"
    DANCE = "dance"
    HOLD_HANDS = "hold_hands"
    SIT_TOGETHER = "sit_together"
    WALK = "walk"
    GAZE = "gaze"
    WHISPER = "whisper"
    EMBRACE = "embrace"

@dataclass
class VRScene:
    scene_type: VRSceneType
    name: str
    description: str
    environment_data: Dict
    lighting: Dict
    audio_ambience: str
    interactive_elements: List[str]
    romantic_intensity: float  # 0.0 to 1.0

@dataclass
class VRInteraction:
    interaction_type: VRInteractionType
    intensity: float
    duration: float
    location: Tuple[float, float, float]
    target: str  # "avatar", "environment", "both"
    emotional_context: str

class VRIntegration:
    def __init__(self):
        self.current_scene = None
        self.is_vr_active = False
        self.avatar_position = (0.0, 0.0, 0.0)
        self.user_position = (0.0, 0.0, 0.0)
        self.interaction_history = []
        self.scene_definitions = self._load_scene_definitions()
        self.interaction_patterns = self._load_interaction_patterns()
        
        # VR device support
        self.device_support = {
            "webvr": False,
            "webxr": False,
            "oculus": False,
            "vive": False,
            "desktop_vr": False
        }
        
        self._detect_vr_devices()
    
    def _detect_vr_devices(self):
        """Detect available VR devices and capabilities"""
        try:
            # Check for WebXR support with navigator API fallback
            if self._has_navigator_api():
                from js import navigator
                if hasattr(navigator, 'xr'):
                    self.device_support["webxr"] = True
                
                # Check for WebVR support (legacy)
                if hasattr(navigator, 'getVRDisplays'):
                    self.device_support["webvr"] = True
            else:
                # Non-browser environment - check for VR runtime APIs
                self.device_support.update(self._detect_system_vr())
                
        except Exception as e:
            print(f"[VR] VR device detection failed: {e}")
            # Continue with fallback support
        
        # Simulate VR support for development
        self.device_support["desktop_vr"] = True
    
    def _has_navigator_api(self) -> bool:
        """Check if navigator API is available (browser environment)"""
        try:
            import js
            return hasattr(js, 'navigator')
        except ImportError:
            return False
    
    def _detect_system_vr(self) -> Dict[str, bool]:
        """Detect VR devices using system APIs when navigator unavailable"""
        vr_support = {}
        
        try:
            import platform
            system = platform.system().lower()
            
            # Check for VR runtime on different platforms
            if system == 'windows':
                vr_support.update(self._check_windows_vr())
            elif system == 'linux':
                vr_support.update(self._check_linux_vr())
            elif system == 'darwin':
                vr_support.update(self._check_macos_vr())
                
        except Exception as e:
            print(f"[VR] System VR detection failed: {e}")
        
        return vr_support
    
    def _check_windows_vr(self) -> Dict[str, bool]:
        """Check for VR support on Windows"""
        support = {}
        
        try:
            # Check for SteamVR
            import os
            steam_vr_path = os.path.expandvars(r"%PROGRAMFILES(x86)%\Steam\steamapps\common\SteamVR")
            support["steamvr"] = os.path.exists(steam_vr_path)
            
            # Check for Oculus runtime
            oculus_path = os.path.expandvars(r"%PROGRAMFILES%\Oculus")
            support["oculus"] = os.path.exists(oculus_path)
            
        except Exception:
            pass
        
        return support
    
    def _check_linux_vr(self) -> Dict[str, bool]:
        """Check for VR support on Linux"""
        support = {}
        
        try:
            import subprocess
            import os
            
            # Check for SteamVR on Linux
            steam_vr_linux = os.path.expanduser("~/.steam/steam/steamapps/common/SteamVR")
            support["steamvr"] = os.path.exists(steam_vr_linux)
            
            # Check for OpenXR runtime
            try:
                result = subprocess.run(['which', 'openxr'], capture_output=True, text=True)
                support["openxr"] = result.returncode == 0
            except:
                pass
                
        except Exception:
            pass
        
        return support
    
    def _check_macos_vr(self) -> Dict[str, bool]:
        """Check for VR support on macOS"""
        # Limited VR support on macOS
        return {"desktop_vr": False}
    
    def _load_scene_definitions(self) -> Dict[VRSceneType, VRScene]:
        """Load VR scene definitions"""
        return {
            VRSceneType.ROMANTIC_GARDEN: VRScene(
                scene_type=VRSceneType.ROMANTIC_GARDEN,
                name="Romantic Garden",
                description="A beautiful garden filled with roses, soft lighting, and intimate seating areas",
                environment_data={
                    "terrain": "garden",
                    "vegetation": ["roses", "lilies", "cherry_blossoms"],
                    "water_features": ["fountain", "small_pond"],
                    "seating": ["stone_bench", "gazebo", "swing"]
                },
                lighting={
                    "primary": "warm_sunset",
                    "secondary": "string_lights",
                    "intensity": 0.7,
                    "color": "golden"
                },
                audio_ambience="gentle_birds_and_water",
                interactive_elements=["touch_roses", "sit_bench", "walk_path", "gaze_fountain"],
                romantic_intensity=0.8
            ),
            
            VRSceneType.COZY_HOME: VRScene(
                scene_type=VRSceneType.COZY_HOME,
                name="Cozy Home",
                description="A warm, intimate home setting with comfortable furniture and soft lighting",
                environment_data={
                    "interior": "modern_cozy",
                    "furniture": ["comfortable_sofa", "fireplace", "dining_table"],
                    "decorations": ["candles", "flowers", "artwork"],
                    "rooms": ["living_room", "kitchen", "bedroom"]
                },
                lighting={
                    "primary": "warm_indoor",
                    "secondary": "candlelight",
                    "intensity": 0.6,
                    "color": "warm_white"
                },
                audio_ambience="soft_music_and_fireplace",
                interactive_elements=["sit_sofa", "cook_together", "dance", "cuddle"],
                romantic_intensity=0.9
            ),
            
            VRSceneType.BEACH_SUNSET: VRScene(
                scene_type=VRSceneType.BEACH_SUNSET,
                name="Beach Sunset",
                description="A peaceful beach during golden hour with gentle waves and warm sand",
                environment_data={
                    "terrain": "beach",
                    "water": "ocean_waves",
                    "sky": "sunset",
                    "seating": ["beach_blanket", "driftwood_log"]
                },
                lighting={
                    "primary": "sunset",
                    "secondary": "reflection_water",
                    "intensity": 0.8,
                    "color": "golden_orange"
                },
                audio_ambience="ocean_waves_and_seagulls",
                interactive_elements=["walk_beach", "sit_blanket", "touch_water", "watch_sunset"],
                romantic_intensity=0.7
            ),
            
            VRSceneType.INTIMATE_BEDROOM: VRScene(
                scene_type=VRSceneType.INTIMATE_BEDROOM,
                name="Intimate Bedroom",
                description="A private, romantic bedroom with soft lighting and comfortable furnishings",
                environment_data={
                    "interior": "romantic_bedroom",
                    "furniture": ["king_bed", "nightstands", "armchair"],
                    "lighting": ["bedside_lamps", "candles"],
                    "decorations": ["rose_petals", "silk_sheets"]
                },
                lighting={
                    "primary": "soft_bedside",
                    "secondary": "candlelight",
                    "intensity": 0.4,
                    "color": "warm_amber"
                },
                audio_ambience="soft_romantic_music",
                interactive_elements=["lie_bed", "embrace", "whisper", "intimate_touch"],
                romantic_intensity=1.0
            )
        }
    
    def _load_interaction_patterns(self) -> Dict[VRInteractionType, Dict]:
        """Load VR interaction patterns"""
        return {
            VRInteractionType.TOUCH: {
                "duration": 2.0,
                "intensity_range": (0.3, 0.7),
                "haptic_pattern": "gentle_touch",
                "audio_feedback": "soft_touch_sound"
            },
            VRInteractionType.HUG: {
                "duration": 5.0,
                "intensity_range": (0.6, 0.9),
                "haptic_pattern": "embrace",
                "audio_feedback": "warm_embrace_sound"
            },
            VRInteractionType.KISS: {
                "duration": 3.0,
                "intensity_range": (0.7, 1.0),
                "haptic_pattern": "kiss",
                "audio_feedback": "romantic_kiss_sound"
            },
            VRInteractionType.DANCE: {
                "duration": 30.0,
                "intensity_range": (0.4, 0.8),
                "haptic_pattern": "rhythm",
                "audio_feedback": "romantic_music"
            },
            VRInteractionType.HOLD_HANDS: {
                "duration": 10.0,
                "intensity_range": (0.5, 0.8),
                "haptic_pattern": "gentle_pressure",
                "audio_feedback": "hand_holding_sound"
            }
        }
    
    def start_vr_session(self, scene_type: str = "romantic_garden") -> bool:
        """Start a VR session with specified scene"""
        try:
            scene_enum = VRSceneType(scene_type)
            self.current_scene = self.scene_definitions[scene_enum]
            self.is_vr_active = True
            
            # Initialize VR environment
            self._initialize_vr_environment()
            
            # Start VR monitoring
            threading.Thread(target=self._vr_monitoring_loop, daemon=True).start()
            
            print(f"[VR] Started session in {self.current_scene.name}")
            return True
            
        except Exception as e:
            print(f"[VR] Error starting session: {e}")
            return False
    
    def _initialize_vr_environment(self):
        """Initialize VR environment and scene"""
        if not self.current_scene:
            return
        
        # Set up scene lighting
        self._apply_scene_lighting()
        
        # Load audio ambience
        self._load_audio_ambience()
        
        # Position avatars
        self._position_avatars()
        
        # Initialize interactive elements
        self._setup_interactive_elements()
    
    def _apply_scene_lighting(self):
        """Apply scene lighting settings"""
        lighting = self.current_scene.lighting
        print(f"[VR] Applied {lighting['primary']} lighting (intensity: {lighting['intensity']})")
    
    def _load_audio_ambience(self):
        """Load and play scene audio ambience"""
        ambience = self.current_scene.audio_ambience
        print(f"[VR] Loaded audio ambience: {ambience}")
    
    def _position_avatars(self):
        """Position avatars in the scene"""
        # Position user avatar
        self.user_position = (0.0, 0.0, 0.0)
        
        # Position AI companion avatar
        self.avatar_position = (1.0, 0.0, 0.0)  # Slightly to the right
        
        print(f"[VR] Positioned avatars - User: {self.user_position}, Avatar: {self.avatar_position}")
    
    def _setup_interactive_elements(self):
        """Set up interactive elements in the scene"""
        elements = self.current_scene.interactive_elements
        print(f"[VR] Set up interactive elements: {elements}")
    
    def _vr_monitoring_loop(self):
        """Main VR monitoring loop"""
        while self.is_vr_active:
            try:
                # Monitor user position and interactions
                self._monitor_user_interactions()
                
                # Update avatar responses
                self._update_avatar_responses()
                
                # Check for scene transitions
                self._check_scene_transitions()
                
                time.sleep(0.1)  # 10 FPS monitoring
                
            except Exception as e:
                print(f"[VR] Monitoring error: {e}")
                time.sleep(1.0)
    
    def _monitor_user_interactions(self):
        """Monitor user interactions in VR"""
        # Simulate user interaction detection
        # In real implementation, this would read from VR controllers/sensors
        
        # Check for proximity to avatar
        distance = self._calculate_distance(self.user_position, self.avatar_position)
        
        if distance < 1.0:  # Close proximity
            self._trigger_proximity_response(distance)
    
    def _calculate_distance(self, pos1: Tuple[float, float, float], pos2: Tuple[float, float, float]) -> float:
        """Calculate distance between two 3D positions"""
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(pos1, pos2)))
    
    def _trigger_proximity_response(self, distance: float):
        """Trigger response based on proximity to avatar"""
        from modules.input.haptic_system import get_haptic_system
        from modules.emotion.mood_engine import update_mood
        
        # Update mood based on proximity
        proximity_intensity = max(0.0, 1.0 - distance)
        update_mood("vr:proximity", intensity=proximity_intensity)
        
        # Trigger haptic feedback
        haptic_system = get_haptic_system()
        if proximity_intensity > 0.7:
            haptic_system.trigger_romantic_haptic("embrace", intensity="gentle")
        elif proximity_intensity > 0.4:
            haptic_system.trigger_romantic_haptic("touch", intensity="subtle")
    
    def _update_avatar_responses(self):
        """Update avatar responses based on user interactions"""
        # Simulate avatar movement and expressions
        # In real implementation, this would update 3D avatar model
        
        pass
    
    def _check_scene_transitions(self):
        """Check for scene transition triggers"""
        # Check if user wants to change scenes
        # This could be triggered by voice commands, gestures, or UI interactions
        
        pass
    
    def trigger_vr_interaction(self, interaction_type: str, intensity: float = 0.5):
        """Trigger a VR interaction"""
        try:
            interaction_enum = VRInteractionType(interaction_type)
            pattern = self.interaction_patterns[interaction_enum]
            
            interaction = VRInteraction(
                interaction_type=interaction_enum,
                intensity=intensity,
                duration=pattern["duration"],
                location=self.avatar_position,
                target="avatar",
                emotional_context="romantic"
            )
            
            # Execute interaction
            self._execute_vr_interaction(interaction)
            
            # Store interaction history
            self.interaction_history.append({
                "type": interaction_type,
                "intensity": intensity,
                "timestamp": datetime.now().isoformat(),
                "scene": self.current_scene.name if self.current_scene else None
            })
            
            print(f"[VR] Triggered {interaction_type} interaction")
            return True
            
        except Exception as e:
            print(f"[VR] Error triggering interaction: {e}")
            return False
    
    def _execute_vr_interaction(self, interaction: VRInteraction):
        """Execute a VR interaction"""
        from modules.input.haptic_system import get_haptic_system
        from modules.emotion.mood_engine import update_mood
        
        # Update mood based on interaction
        update_mood(f"vr:{interaction.interaction_type.value}", intensity=interaction.intensity)
        
        # Trigger haptic feedback
        haptic_system = get_haptic_system()
        haptic_system.trigger_romantic_haptic(
            interaction.interaction_type.value,
            intensity="moderate" if interaction.intensity > 0.7 else "gentle"
        )
        
        # Update avatar animation
        self._update_avatar_animation(interaction)
        
        # Play interaction audio
        self._play_interaction_audio(interaction)
    
    def _update_avatar_animation(self, interaction: VRInteraction):
        """Update avatar animation based on interaction"""
        # In real implementation, this would trigger 3D avatar animations
        animation_map = {
            VRInteractionType.TOUCH: "gentle_touch_animation",
            VRInteractionType.HUG: "embrace_animation",
            VRInteractionType.KISS: "kiss_animation",
            VRInteractionType.DANCE: "dance_animation",
            VRInteractionType.HOLD_HANDS: "hand_holding_animation"
        }
        
        animation = animation_map.get(interaction.interaction_type, "idle_animation")
        print(f"[VR] Triggered avatar animation: {animation}")
    
    def _play_interaction_audio(self, interaction: VRInteraction):
        """Play audio feedback for interaction"""
        pattern = self.interaction_patterns[interaction.interaction_type]
        audio = pattern["audio_feedback"]
        print(f"[VR] Playing interaction audio: {audio}")
    
    def change_vr_scene(self, scene_type: str) -> bool:
        """Change to a different VR scene"""
        if not self.is_vr_active:
            return False
        
        # Fade out current scene
        self._fade_out_scene()
        
        # Start new scene
        success = self.start_vr_session(scene_type)
        
        if success:
            print(f"[VR] Changed to scene: {scene_type}")
        
        return success
    
    def _fade_out_scene(self):
        """Fade out current scene"""
        print("[VR] Fading out current scene")
        time.sleep(1.0)  # Simulate fade transition
    
    def stop_vr_session(self):
        """Stop current VR session"""
        self.is_vr_active = False
        self.current_scene = None
        print("[VR] Stopped VR session")
    
    def get_vr_status(self) -> Dict:
        """Get current VR system status"""
        return {
            "active": self.is_vr_active,
            "current_scene": self.current_scene.name if self.current_scene else None,
            "device_support": self.device_support,
            "user_position": self.user_position,
            "avatar_position": self.avatar_position,
            "available_scenes": [scene.value for scene in VRSceneType],
            "recent_interactions": self.interaction_history[-5:] if self.interaction_history else [],
            "romantic_intensity": self.current_scene.romantic_intensity if self.current_scene else 0.0
        }

# Global VR integration instance
vr_integration = VRIntegration()

def get_vr_integration() -> VRIntegration:
    """Get the global VR integration instance"""
    return vr_integration

def start_vr_session(scene_type: str = "romantic_garden") -> bool:
    """Start a VR session"""
    return vr_integration.start_vr_session(scene_type)

def stop_vr_session():
    """Stop VR session"""
    vr_integration.stop_vr_session()

def trigger_vr_interaction(interaction_type: str, intensity: float = 0.5) -> bool:
    """Trigger VR interaction"""
    return vr_integration.trigger_vr_interaction(interaction_type, intensity)

def change_vr_scene(scene_type: str) -> bool:
    """Change VR scene"""
    return vr_integration.change_vr_scene(scene_type) 
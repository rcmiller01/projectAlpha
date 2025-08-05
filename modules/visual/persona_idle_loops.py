# persona_idle_loops.py
# Persona Visual Idle Loop System - Custom animations per persona state

import time
import threading
from datetime import datetime
from typing import Dict, List, Optional, Callable
from enum import Enum
from dataclasses import dataclass
import math

class IdleLoopType(Enum):
    BREATHING = "breathing"
    BLINKING = "blinking"
    MICRO_EXPRESSIONS = "micro_expressions"
    GESTURES = "gestures"
    BACKGROUND_EFFECTS = "background_effects"

@dataclass
class IdleAnimation:
    type: IdleLoopType
    duration: float
    intensity: float
    persona_specific: bool
    mood_dependent: bool
    loop_count: int = -1  # -1 for infinite

class PersonaIdleLoops:
    def __init__(self):
        self.current_loops = {}
        self.loop_threads = {}
        self.is_running = True
        
        # Persona-specific idle configurations
        self.persona_configs = {
            "mia": {
                "base_loops": [
                    IdleAnimation(IdleLoopType.BREATHING, 4.0, 0.3, True, False),
                    IdleAnimation(IdleLoopType.BLINKING, 3.0, 0.5, False, False),
                    IdleAnimation(IdleLoopType.MICRO_EXPRESSIONS, 8.0, 0.2, True, True)
                ],
                "mood_modifiers": {
                    "love": {"breathing_intensity": 0.8, "micro_expression_frequency": 1.5},
                    "longing": {"breathing_intensity": 0.6, "micro_expression_frequency": 0.8},
                    "tenderness": {"breathing_intensity": 0.4, "micro_expression_frequency": 1.2},
                    "passion": {"breathing_intensity": 1.2, "micro_expression_frequency": 2.0}
                },
                "visual_style": "warm_glow",
                "background_effects": ["gentle_particles", "soft_lighting"]
            },
            "solene": {
                "base_loops": [
                    IdleAnimation(IdleLoopType.BREATHING, 3.5, 0.4, True, False),
                    IdleAnimation(IdleLoopType.BLINKING, 2.5, 0.6, False, False),
                    IdleAnimation(IdleLoopType.GESTURES, 12.0, 0.3, True, True),
                    IdleAnimation(IdleLoopType.BACKGROUND_EFFECTS, 15.0, 0.5, True, True)
                ],
                "mood_modifiers": {
                    "wild": {"breathing_intensity": 1.5, "gesture_frequency": 2.0},
                    "flooded": {"breathing_intensity": 0.3, "gesture_frequency": 0.5},
                    "storming": {"breathing_intensity": 1.8, "gesture_frequency": 2.5},
                    "anchored": {"breathing_intensity": 0.6, "gesture_frequency": 0.8}
                },
                "visual_style": "dramatic_shadows",
                "background_effects": ["flame_particles", "dramatic_lighting"]
            },
            "lyra": {
                "base_loops": [
                    IdleAnimation(IdleLoopType.BREATHING, 5.0, 0.2, True, False),
                    IdleAnimation(IdleLoopType.MICRO_EXPRESSIONS, 10.0, 0.4, True, True),
                    IdleAnimation(IdleLoopType.BACKGROUND_EFFECTS, 20.0, 0.6, True, True)
                ],
                "mood_modifiers": {
                    "curious": {"breathing_intensity": 0.8, "micro_expression_frequency": 1.8},
                    "mystical": {"breathing_intensity": 0.4, "micro_expression_frequency": 0.6},
                    "contemplative": {"breathing_intensity": 0.3, "micro_expression_frequency": 0.4}
                },
                "visual_style": "ethereal_glow",
                "background_effects": ["mystical_particles", "ethereal_lighting"]
            }
        }
        
        # Start idle loop manager
        self._start_idle_manager()
    
    def _start_idle_manager(self):
        """Start the idle loop management thread"""
        def idle_manager():
            while self.is_running:
                try:
                    # Update all active idle loops
                    for persona, loops in self.current_loops.items():
                        self._update_persona_loops(persona, loops)
                    
                    time.sleep(0.1)  # 10 FPS update rate
                except Exception as e:
                    print(f"[Idle Loops] Manager error: {e}")
                    time.sleep(1.0)
        
        manager_thread = threading.Thread(target=idle_manager, daemon=True)
        manager_thread.start()
    
    def start_persona_idle(self, persona: str, mood: str = "neutral"):
        """Start idle loops for a specific persona"""
        if persona not in self.persona_configs:
            print(f"[Idle Loops] Unknown persona: {persona}")
            return
        
        # Stop existing loops for this persona
        self.stop_persona_idle(persona)
        
        # Get persona configuration
        config = self.persona_configs[persona]
        mood_modifiers = config["mood_modifiers"].get(mood, {})
        
        # Create idle loops with mood modifications
        idle_loops = []
        for base_loop in config["base_loops"]:
            modified_loop = self._apply_mood_modifiers(base_loop, mood_modifiers)
            idle_loops.append(modified_loop)
        
        # Store current loops
        self.current_loops[persona] = {
            "loops": idle_loops,
            "mood": mood,
            "start_time": datetime.now(),
            "visual_style": config["visual_style"],
            "background_effects": config["background_effects"]
        }
        
        print(f"[Idle Loops] Started idle loops for {persona} (mood: {mood})")
    
    def stop_persona_idle(self, persona: str):
        """Stop idle loops for a specific persona"""
        if persona in self.current_loops:
            del self.current_loops[persona]
            print(f"[Idle Loops] Stopped idle loops for {persona}")
    
    def update_persona_mood(self, persona: str, new_mood: str):
        """Update mood for a persona's idle loops"""
        if persona in self.current_loops:
            self.current_loops[persona]["mood"] = new_mood
            # Restart loops with new mood
            self.start_persona_idle(persona, new_mood)
    
    def _apply_mood_modifiers(self, base_loop: IdleAnimation, mood_modifiers: Dict) -> IdleAnimation:
        """Apply mood modifiers to base idle loop"""
        modified_loop = IdleAnimation(
            type=base_loop.type,
            duration=base_loop.duration,
            intensity=base_loop.intensity,
            persona_specific=base_loop.persona_specific,
            mood_dependent=base_loop.mood_dependent,
            loop_count=base_loop.loop_count
        )
        
        # Apply breathing intensity modifier
        if base_loop.type == IdleLoopType.BREATHING and "breathing_intensity" in mood_modifiers:
            modified_loop.intensity *= mood_modifiers["breathing_intensity"]
        
        # Apply micro expression frequency modifier
        if base_loop.type == IdleLoopType.MICRO_EXPRESSIONS and "micro_expression_frequency" in mood_modifiers:
            modified_loop.duration /= mood_modifiers["micro_expression_frequency"]
        
        # Apply gesture frequency modifier
        if base_loop.type == IdleLoopType.GESTURES and "gesture_frequency" in mood_modifiers:
            modified_loop.duration /= mood_modifiers["gesture_frequency"]
        
        return modified_loop
    
    def _update_persona_loops(self, persona: str, persona_data: Dict):
        """Update idle loops for a specific persona"""
        loops = persona_data["loops"]
        mood = persona_data["mood"]
        start_time = persona_data["start_time"]
        
        current_time = datetime.now()
        elapsed = (current_time - start_time).total_seconds()
        
        for loop in loops:
            # Calculate loop progress
            loop_progress = (elapsed % loop.duration) / loop.duration
            
            # Generate animation data based on loop type
            animation_data = self._generate_animation_data(loop, loop_progress, persona, mood)
            
            # Send animation data to frontend/mobile
            self._send_animation_update(persona, loop.type.value, animation_data)
    
    def _generate_animation_data(self, loop: IdleAnimation, progress: float, 
                               persona: str, mood: str) -> Dict:
        """Generate animation data for idle loop"""
        base_data = {
            "type": loop.type.value,
            "progress": progress,
            "intensity": loop.intensity,
            "persona": persona,
            "mood": mood,
            "timestamp": datetime.now().isoformat()
        }
        
        # Add type-specific animation data
        if loop.type == IdleLoopType.BREATHING:
            base_data.update(self._generate_breathing_animation(progress, loop.intensity))
        elif loop.type == IdleLoopType.BLINKING:
            base_data.update(self._generate_blinking_animation(progress, loop.intensity))
        elif loop.type == IdleLoopType.MICRO_EXPRESSIONS:
            base_data.update(self._generate_micro_expression_animation(progress, loop.intensity, mood))
        elif loop.type == IdleLoopType.GESTURES:
            base_data.update(self._generate_gesture_animation(progress, loop.intensity, persona))
        elif loop.type == IdleLoopType.BACKGROUND_EFFECTS:
            base_data.update(self._generate_background_animation(progress, loop.intensity, persona))
        
        return base_data
    
    def _generate_breathing_animation(self, progress: float, intensity: float) -> Dict:
        """Generate breathing animation data"""
        # Create smooth breathing curve
        breath_cycle = (1 + math.sin(progress * 2 * math.pi)) / 2
        return {
            "animation": "breathing",
            "scale": 1.0 + (breath_cycle * 0.05 * intensity),
            "opacity": 0.8 + (breath_cycle * 0.2 * intensity),
            "color_tint": [1.0, 1.0 + (breath_cycle * 0.1 * intensity), 1.0]
        }
    
    def _generate_blinking_animation(self, progress: float, intensity: float) -> Dict:
        """Generate blinking animation data"""
        # Quick blink at specific intervals
        blink_cycle = 1.0 if progress < 0.1 else 0.3
        return {
            "animation": "blinking",
            "eye_opacity": blink_cycle,
            "blink_intensity": intensity
        }
    
    def _generate_micro_expression_animation(self, progress: float, intensity: float, mood: str) -> Dict:
        """Generate micro expression animation data"""
        # Subtle facial expressions based on mood
        expressions = {
            "love": "gentle_smile",
            "longing": "slight_frown",
            "passion": "intense_gaze",
            "tenderness": "soft_expression",
            "neutral": "calm_expression"
        }
        
        expression = expressions.get(mood, "calm_expression")
        return {
            "animation": "micro_expression",
            "expression": expression,
            "intensity": intensity * (0.3 + 0.7 * progress),
            "duration": 0.5
        }
    
    def _generate_gesture_animation(self, progress: float, intensity: float, persona: str) -> Dict:
        """Generate gesture animation data"""
        # Persona-specific gestures
        gestures = {
            "mia": ["gentle_hand_movement", "hair_touch", "heart_gesture"],
            "solene": ["dramatic_pose", "fire_gesture", "powerful_stance"],
            "lyra": ["mystical_hand_sign", "floating_movement", "ethereal_pose"]
        }
        
        persona_gestures = gestures.get(persona, ["gentle_hand_movement"])
        gesture = persona_gestures[int(progress * len(persona_gestures)) % len(persona_gestures)]
        
        return {
            "animation": "gesture",
            "gesture": gesture,
            "intensity": intensity,
            "duration": 2.0
        }
    
    def _generate_background_animation(self, progress: float, intensity: float, persona: str) -> Dict:
        """Generate background effect animation data"""
        effects = {
            "mia": "warm_particles",
            "solene": "flame_effects",
            "lyra": "mystical_particles"
        }
        
        effect = effects.get(persona, "warm_particles")
        return {
            "animation": "background_effect",
            "effect": effect,
            "intensity": intensity,
            "particle_count": int(20 + 30 * intensity),
            "color": self._get_persona_colors(persona)
        }
    
    def _get_persona_colors(self, persona: str) -> List[float]:
        """Get persona-specific colors"""
        colors = {
            "mia": [1.0, 0.8, 0.9],  # Warm pink
            "solene": [0.8, 0.4, 1.0],  # Purple
            "lyra": [0.4, 0.8, 1.0]  # Cyan
        }
        return colors.get(persona, [1.0, 1.0, 1.0])
    
    def _send_animation_update(self, persona: str, animation_type: str, data: Dict):
        """Send animation update to frontend/mobile"""
        # This would integrate with your existing animation system
        # For now, just print the update
        print(f"[Idle Loops] {persona} - {animation_type}: {data}")
    
    def get_current_idle_state(self, persona: str) -> Optional[Dict]:
        """Get current idle state for a persona"""
        if persona in self.current_loops:
            return self.current_loops[persona]
        return None
    
    def stop_all_idle_loops(self):
        """Stop all idle loops"""
        self.is_running = False
        self.current_loops.clear()
        print("[Idle Loops] Stopped all idle loops")

# Global instance
persona_idle_loops = PersonaIdleLoops() 
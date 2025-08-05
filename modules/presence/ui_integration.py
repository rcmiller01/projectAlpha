"""
Emotional Presence UI Integration

This module provides frontend components and utilities for displaying
emotional presence signals in the user interface through ambient visual
effects, overlays, and atmospheric elements.
"""

import json
import asyncio
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from pathlib import Path
import logging
import random
import math
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class UIPresenceState:
    """Current UI presence state for rendering"""
    emotion: str
    intensity: float
    primary_color: str
    secondary_color: str
    effects: Dict[str, Any]
    timestamp: datetime
    duration_remaining: float

class EmotionalUIRenderer:
    """Renders emotional presence signals in the user interface"""
    
    def __init__(self, config_path: str = "data/emotional_signatures.json"):
        self.config_path = Path(config_path)
        self.signatures = {}
        self.current_state: Optional[UIPresenceState] = None
        self.animation_frame = 0
        self.load_signatures()
        
    def load_signatures(self):
        """Load emotional signature configurations"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    self.signatures = json.load(f)
                logger.info(f"Loaded {len(self.signatures)} emotional signatures")
            else:
                logger.warning(f"Emotional signatures file not found: {self.config_path}")
                self.signatures = {}
        except Exception as e:
            logger.error(f"Error loading emotional signatures: {e}")
            self.signatures = {}
    
    def set_emotional_state(self, emotion: str, intensity: float = 0.7, duration: float = 30.0):
        """Set the current emotional state for UI rendering"""
        if emotion not in self.signatures:
            logger.warning(f"Unknown emotion for UI rendering: {emotion}")
            return
            
        signature = self.signatures[emotion]
        effects = signature.get('ui_effects', {})
        
        self.current_state = UIPresenceState(
            emotion=emotion,
            intensity=intensity,
            primary_color=signature['primary_color'],
            secondary_color=signature['secondary_color'],
            effects=effects,
            timestamp=datetime.now(),
            duration_remaining=duration
        )
        
        logger.info(f"UI emotional state set to {emotion} (intensity: {intensity})")
    
    def get_current_css_variables(self) -> Dict[str, str]:
        """Generate CSS variables for current emotional state"""
        if not self.current_state:
            return {}
            
        state = self.current_state
        effects = state.effects
        
        # Calculate time-based modulations
        elapsed = (datetime.now() - state.timestamp).total_seconds()
        pulse_phase = (elapsed * effects.get('pulse_frequency', 1.0)) % (2 * math.pi)
        wave_phase = (elapsed * 0.5) % (2 * math.pi)
        
        # Base colors with intensity modulation
        primary_opacity = state.intensity * effects.get('background_overlay', 0.1)
        glow_intensity = state.intensity * (0.5 + 0.3 * math.sin(pulse_phase))
        
        css_vars = {
            '--emotion-primary': state.primary_color,
            '--emotion-secondary': state.secondary_color,
            '--emotion-intensity': str(state.intensity),
            '--emotion-overlay-opacity': str(primary_opacity),
            '--emotion-glow-intensity': str(glow_intensity),
            '--emotion-pulse-phase': str(pulse_phase),
            '--emotion-wave-phase': str(wave_phase),
        }
        
        # Animation speed mappings
        speed_map = {
            'very_slow': '8s',
            'slow': '4s',
            'medium': '2s',
            'fast': '1s',
            'very_fast': '0.5s'
        }
        
        animation_speed = effects.get('animation_speed', 'medium')
        css_vars['--emotion-animation-duration'] = speed_map.get(animation_speed, '2s')
        
        # Particle density
        density_map = {
            'sparse': 5,
            'light': 15,
            'medium': 30,
            'dense': 50
        }
        
        particle_density = effects.get('particle_density', 'light')
        particle_count = density_map.get(particle_density, 15)
        css_vars['--emotion-particle-count'] = str(particle_count)
        
        # Special effect parameters
        if 'glow_radius' in effects:
            css_vars['--emotion-glow-radius'] = f"{effects['glow_radius']}px"
        
        if 'sparkle_count' in effects:
            css_vars['--emotion-sparkle-count'] = str(effects['sparkle_count'])
        
        if 'brightness_boost' in effects:
            css_vars['--emotion-brightness'] = str(effects['brightness_boost'])
        
        return css_vars
    
    def generate_css_classes(self) -> str:
        """Generate dynamic CSS classes for current emotional state"""
        if not self.current_state:
            return ""
            
        state = self.current_state
        effects = state.effects
        emotion = state.emotion
        
        css = f"""
        .emotion-{emotion} {{
            position: relative;
            transition: all 0.5s ease-in-out;
        }}
        
        .emotion-{emotion}::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: linear-gradient(45deg, 
                {state.primary_color}10, 
                {state.secondary_color}05);
            opacity: var(--emotion-overlay-opacity);
            pointer-events: none;
            z-index: -1;
        }}
        """
        
        # Add glow effects
        if effects.get('edge_glow'):
            css += f"""
            .emotion-{emotion} {{
                box-shadow: 
                    0 0 var(--emotion-glow-radius, 10px) {state.primary_color}40,
                    inset 0 0 var(--emotion-glow-radius, 10px) {state.secondary_color}20;
            }}
            """
        
        # Add animation patterns
        pattern = self.signatures[emotion].get('visual_pattern', 'gentle_glow')
        
        if pattern == 'slow_pulse':
            css += f"""
            .emotion-{emotion} {{
                animation: pulse-{emotion} var(--emotion-animation-duration) infinite ease-in-out;
            }}
            
            @keyframes pulse-{emotion} {{
                0%, 100% {{ opacity: 0.7; transform: scale(1); }}
                50% {{ opacity: 1; transform: scale(1.02); }}
            }}
            """
        
        elif pattern == 'sparkle':
            css += f"""
            .emotion-{emotion}::after {{
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background-image: 
                    radial-gradient(2px 2px at 20% 30%, {state.primary_color}, transparent),
                    radial-gradient(1px 1px at 40% 70%, {state.secondary_color}, transparent),
                    radial-gradient(1px 1px at 80% 10%, {state.primary_color}, transparent);
                animation: sparkle-{emotion} var(--emotion-animation-duration) infinite linear;
                pointer-events: none;
            }}
            
            @keyframes sparkle-{emotion} {{
                0%, 100% {{ opacity: 0; transform: scale(0.5) rotate(0deg); }}
                50% {{ opacity: 1; transform: scale(1) rotate(180deg); }}
            }}
            """
        
        elif pattern == 'gentle_wave':
            css += f"""
            .emotion-{emotion} {{
                animation: wave-{emotion} var(--emotion-animation-duration) infinite ease-in-out;
            }}
            
            @keyframes wave-{emotion} {{
                0%, 100% {{ transform: translateY(0px); }}
                25% {{ transform: translateY(-2px); }}
                75% {{ transform: translateY(2px); }}
            }}
            """
        
        elif pattern == 'electric_pulse':
            css += f"""
            .emotion-{emotion} {{
                animation: electric-{emotion} var(--emotion-animation-duration) infinite;
            }}
            
            @keyframes electric-{emotion} {{
                0% {{ box-shadow: 0 0 5px {state.primary_color}60; }}
                25% {{ box-shadow: 0 0 15px {state.primary_color}80, 0 0 25px {state.secondary_color}40; }}
                50% {{ box-shadow: 0 0 10px {state.primary_color}70; }}
                75% {{ box-shadow: 0 0 20px {state.primary_color}90, 0 0 30px {state.secondary_color}50; }}
                100% {{ box-shadow: 0 0 5px {state.primary_color}60; }}
            }}
            """
        
        return css
    
    def generate_particle_system(self) -> Dict[str, Any]:
        """Generate particle system configuration for current emotion"""
        if not self.current_state:
            return {}
            
        state = self.current_state
        effects = state.effects
        
        # Map particle density strings to numbers
        density_map = {
            'sparse': 5,
            'light': 15,
            'medium': 30,
            'dense': 50
        }
        
        particle_density_str = effects.get('particle_density', 'light')
        particle_count = density_map.get(particle_density_str, 15)
        
        particles = []
        for i in range(particle_count):
            particle = {
                'id': i,
                'x': random.uniform(0, 100),  # Percentage
                'y': random.uniform(0, 100),  # Percentage
                'size': random.uniform(1, 4),
                'opacity': random.uniform(0.3, 0.8),
                'speed': random.uniform(0.5, 2.0),
                'direction': random.uniform(0, 360),
                'color': random.choice([state.primary_color, state.secondary_color]),
                'lifetime': random.uniform(5, 15),  # seconds
            }
            particles.append(particle)
        
        return {
            'particles': particles,
            'gravity': effects.get('gravity', 0.1),
            'wind': effects.get('wind', 0.05),
            'emission_rate': effects.get('emission_rate', 2.0),
            'blend_mode': effects.get('blend_mode', 'normal'),
        }
    
    def update_animation_frame(self):
        """Update animation frame counter"""
        self.animation_frame += 1
        
        # Update duration
        if self.current_state:
            elapsed = (datetime.now() - self.current_state.timestamp).total_seconds()
            self.current_state.duration_remaining = max(0, 
                self.current_state.duration_remaining - elapsed)
            
            # Clear state if duration expired
            if self.current_state.duration_remaining <= 0:
                self.current_state = None
    
    def get_whisper_overlay_data(self) -> Optional[Dict[str, Any]]:
        """Get data for whisper text overlay"""
        if not self.current_state:
            return None
            
        emotion = self.current_state.emotion
        if emotion not in self.signatures:
            return None
            
        phrases = self.signatures[emotion].get('whisper_phrases', [])
        if not phrases:
            return None
            
        # Select phrase based on animation frame for variety
        phrase_index = (self.animation_frame // 60) % len(phrases)  # Change every 60 frames
        selected_phrase = phrases[phrase_index]
        
        return {
            'text': selected_phrase,
            'color': self.current_state.secondary_color,
            'opacity': self.current_state.intensity * 0.6,
            'position': 'bottom-right',  # Can be configurable
            'duration': 3.0,  # seconds to display
            'fade_in': 0.5,
            'fade_out': 0.5,
        }
    
    def get_ambient_light_config(self) -> Dict[str, Any]:
        """Get ambient lighting configuration for external devices"""
        if not self.current_state:
            return {}
            
        state = self.current_state
        
        # Convert hex colors to RGB
        def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
            hex_color = hex_color.lstrip('#')
            rgb_values = [int(hex_color[i:i+2], 16) for i in (0, 2, 4)]
            return (rgb_values[0], rgb_values[1], rgb_values[2])
        
        primary_rgb = hex_to_rgb(state.primary_color)
        secondary_rgb = hex_to_rgb(state.secondary_color)
        
        return {
            'primary_color': {
                'r': primary_rgb[0],
                'g': primary_rgb[1],
                'b': primary_rgb[2],
                'intensity': state.intensity
            },
            'secondary_color': {
                'r': secondary_rgb[0],
                'g': secondary_rgb[1],
                'b': secondary_rgb[2],
                'intensity': state.intensity * 0.7
            },
            'pattern': self.signatures[state.emotion].get('visual_pattern', 'gentle_glow'),
            'duration': state.duration_remaining,
            'transition_speed': 1.0  # seconds
        }

class PresenceUIManager:
    """High-level manager for emotional presence in UI"""
    
    def __init__(self):
        self.renderer = EmotionalUIRenderer()
        self.active_overlays: List[str] = []
        self.css_injected = False
        
    async def activate_emotional_presence(self, emotion: str, intensity: float = 0.7, duration: float = 30.0):
        """Activate emotional presence with full UI integration"""
        
        # Set the emotional state
        self.renderer.set_emotional_state(emotion, intensity, duration)
        
        # Generate and inject CSS
        css_vars = self.renderer.get_current_css_variables()
        css_classes = self.renderer.generate_css_classes()
        
        # In a real implementation, this would inject CSS into the DOM
        logger.info(f"Activating emotional presence: {emotion}")
        logger.debug(f"CSS Variables: {css_vars}")
        
        # Set up particle system
        particles = self.renderer.generate_particle_system()
        logger.debug(f"Particle system: {len(particles.get('particles', []))} particles")
        
        # Configure ambient lighting for external devices
        ambient_config = self.renderer.get_ambient_light_config()
        logger.debug(f"Ambient lighting: {ambient_config}")
        
        # Start whisper overlay system
        whisper_data = self.renderer.get_whisper_overlay_data()
        if whisper_data:
            logger.debug(f"Whisper overlay: {whisper_data['text']}")
        
        return {
            'css_variables': css_vars,
            'css_classes': css_classes,
            'particle_system': particles,
            'ambient_lighting': ambient_config,
            'whisper_overlay': whisper_data,
            'status': 'activated'
        }
    
    async def update_presence_frame(self):
        """Update presence system for one animation frame"""
        self.renderer.update_animation_frame()
        
        if self.renderer.current_state:
            # Return updated frame data
            return {
                'css_variables': self.renderer.get_current_css_variables(),
                'whisper_overlay': self.renderer.get_whisper_overlay_data(),
                'time_remaining': self.renderer.current_state.duration_remaining,
            }
        return None
    
    def clear_emotional_presence(self):
        """Clear all emotional presence effects"""
        self.renderer.current_state = None
        self.active_overlays.clear()
        logger.info("Emotional presence cleared")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current presence system status"""
        if not self.renderer.current_state:
            return {'active': False}
            
        state = self.renderer.current_state
        return {
            'active': True,
            'emotion': state.emotion,
            'intensity': state.intensity,
            'duration_remaining': state.duration_remaining,
            'effects_active': len(self.active_overlays),
            'animation_frame': self.renderer.animation_frame
        }

# Example integration functions
async def demo_emotional_ui():
    """Demonstrate emotional UI presence system"""
    manager = PresenceUIManager()
    
    emotions = ['longing', 'joy', 'peace', 'anticipation', 'melancholy', 'warmth']
    
    for emotion in emotions:
        print(f"\n=== Demonstrating {emotion.title()} Presence ===")
        
        # Activate emotional presence
        result = await manager.activate_emotional_presence(emotion, intensity=0.8, duration=10.0)
        print(f"Activated: {result['status']}")
        print(f"CSS Variables: {len(result['css_variables'])} properties")
        print(f"Particles: {len(result['particle_system'].get('particles', []))} active")
        
        if result['whisper_overlay']:
            print(f"Whisper: '{result['whisper_overlay']['text']}'")
        
        # Simulate a few animation frames
        for frame in range(3):
            await asyncio.sleep(1)
            frame_data = await manager.update_presence_frame()
            if frame_data:
                print(f"Frame {frame + 1}: {frame_data['time_remaining']:.1f}s remaining")
        
        manager.clear_emotional_presence()
        await asyncio.sleep(0.5)

if __name__ == "__main__":
    # Run the demo
    asyncio.run(demo_emotional_ui())

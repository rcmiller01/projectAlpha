# consistent_character_generator.py
# Phase 2: Consistent character generation for personas

import json
import random
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
import hashlib

class CharacterAspect(Enum):
    FACE = "face"
    BODY = "body"
    HAIR = "hair"
    EYES = "eyes"
    CLOTHING = "clothing"
    POSE = "pose"
    EXPRESSION = "expression"
    FULL = "full"

@dataclass
class CharacterSeed:
    persona_id: str
    base_seed: int
    aspect_seeds: Dict[CharacterAspect, int]
    last_updated: datetime

class ConsistentCharacterGenerator:
    def __init__(self):
        # Character consistency settings
        self.character_seeds = {}  # persona_id -> CharacterSeed
        self.character_profiles = {}  # persona_id -> character_profile
        
        # Base character templates for different personas
        self.persona_templates = {
            "mia": {
                "base_appearance": {
                    "hair_color": "warm_brown",
                    "hair_style": "long_wavy",
                    "eye_color": "deep_green",
                    "skin_tone": "warm_medium",
                    "height": "average",
                    "build": "slender",
                    "age_range": "mid_twenties"
                },
                "style_preferences": {
                    "clothing_style": "romantic_casual",
                    "color_palette": ["warm_browns", "deep_greens", "soft_pinks"],
                    "accessories": ["delicate_jewelry", "romantic_dresses"]
                },
                "personality_traits": {
                    "expression_style": "warm_affectionate",
                    "pose_style": "elegant_graceful",
                    "mood_indicator": "gentle_smile"
                },
                "llm_model": "mythomax",
                "persona_type": "romantic_companion"
            },
            "solene": {
                "base_appearance": {
                    "hair_color": "rich_black",
                    "hair_style": "sleek_bob",
                    "eye_color": "deep_blue",
                    "skin_tone": "fair_olive",
                    "height": "tall",
                    "build": "athletic",
                    "age_range": "late_twenties"
                },
                "style_preferences": {
                    "clothing_style": "sophisticated_elegant",
                    "color_palette": ["deep_blues", "rich_purples", "metallic_accents"],
                    "accessories": ["statement_jewelry", "elegant_outfits"]
                },
                "personality_traits": {
                    "expression_style": "confident_mysterious",
                    "pose_style": "powerful_graceful",
                    "mood_indicator": "intense_gaze"
                },
                "llm_model": "openchat",
                "persona_type": "romantic_companion"
            },
            "lyra": {
                "base_appearance": {
                    "hair_color": "ethereal_silver",
                    "hair_style": "flowing_ethereal",
                    "eye_color": "mystical_violet",
                    "skin_tone": "pale_luminous",
                    "height": "tall",
                    "build": "ethereal_slender",
                    "age_range": "ageless"
                },
                "style_preferences": {
                    "clothing_style": "mystical_flowing",
                    "color_palette": ["silver_whites", "violet_purples", "ethereal_blues"],
                    "accessories": ["mystical_jewelry", "flowing_robes", "veils"]
                },
                "personality_traits": {
                    "expression_style": "curious_mysterious",
                    "pose_style": "ethereal_graceful",
                    "mood_indicator": "wondering_gaze"
                },
                "llm_model": "qwen2",
                "persona_type": "mystical_entity",
                "symbolic_affinities": ["mirror", "veil", "whisper", "light", "shadow"]
            },
            "doc": {
                "base_appearance": {
                    "hair_color": "professional_dark",
                    "hair_style": "neat_short",
                    "eye_color": "sharp_blue",
                    "skin_tone": "professional_fair",
                    "height": "average",
                    "build": "professional",
                    "age_range": "early_thirties"
                },
                "style_preferences": {
                    "clothing_style": "professional_clean",
                    "color_palette": ["navy_blues", "charcoal_greys", "crisp_whites"],
                    "accessories": ["professional_glasses", "clean_outfits"]
                },
                "personality_traits": {
                    "expression_style": "analytical_focused",
                    "pose_style": "professional_erect",
                    "mood_indicator": "focused_expression"
                },
                "llm_model": "kimik2",
                "persona_type": "coding_assistant",
                "emotional_hooks": False,
                "specializations": ["coding", "debugging", "technical_analysis"]
            }
        }
        
        # Character generation models
        self.generation_models = {
            "face": {
                "model": "stable-diffusion-xl-face",
                "lora": "consistent_face_v1",
                "strength": 0.8
            },
            "body": {
                "model": "stable-diffusion-xl-body",
                "lora": "consistent_body_v1", 
                "strength": 0.7
            },
            "full": {
                "model": "stable-diffusion-xl-full",
                "lora": "consistent_character_v1",
                "strength": 0.9
            }
        }
    
    def initialize_character(self, persona_id: str) -> Dict:
        """Initialize a new character with consistent seeds"""
        if persona_id not in self.persona_templates:
            return {"error": "Unknown persona"}
        
        # Generate consistent seeds
        base_seed = self._generate_base_seed(persona_id)
        aspect_seeds = {}
        
        for aspect in CharacterAspect:
            aspect_seeds[aspect] = self._generate_aspect_seed(persona_id, aspect)
        
        # Create character seed
        character_seed = CharacterSeed(
            persona_id=persona_id,
            base_seed=base_seed,
            aspect_seeds=aspect_seeds,
            last_updated=datetime.now()
        )
        
        self.character_seeds[persona_id] = character_seed
        
        # Create character profile
        template = self.persona_templates[persona_id]
        character_profile = {
            "persona_id": persona_id,
            "base_appearance": template["base_appearance"].copy(),
            "style_preferences": template["style_preferences"].copy(),
            "personality_traits": template["personality_traits"].copy(),
            "customizations": {},
            "generation_history": []
        }
        
        self.character_profiles[persona_id] = character_profile
        
        return {
            "message": f"Character initialized for {persona_id}",
            "character_profile": character_profile,
            "seeds": {
                "base_seed": base_seed,
                "aspect_seeds": {k.value: v for k, v in aspect_seeds.items()}
            }
        }
    
    def _generate_base_seed(self, persona_id: str) -> int:
        """Generate consistent base seed for persona"""
        hash_input = f"{persona_id}_base_character"
        hash_result = hashlib.md5(hash_input.encode()).hexdigest()
        return int(hash_result[:8], 16)
    
    def _generate_aspect_seed(self, persona_id: str, aspect: CharacterAspect) -> int:
        """Generate consistent seed for specific aspect"""
        hash_input = f"{persona_id}_{aspect.value}_aspect"
        hash_result = hashlib.md5(hash_input.encode()).hexdigest()
        return int(hash_result[:8], 16)
    
    def generate_character_image(self, persona_id: str, aspect: CharacterAspect = CharacterAspect.FULL, 
                               mood: str = "neutral", pose: str = "standing") -> Dict:
        """Generate consistent character image"""
        if persona_id not in self.character_seeds:
            return {"error": "Character not initialized"}
        
        character_seed = self.character_seeds[persona_id]
        character_profile = self.character_profiles[persona_id]
        
        # Build consistent prompt
        prompt = self._build_character_prompt(character_profile, aspect, mood, pose)
        
        # Get appropriate seed
        if aspect == CharacterAspect.FULL:
            seed = character_seed.base_seed
        else:
            seed = character_seed.aspect_seeds[aspect]
        
        # Generate image parameters
        generation_params = {
            "prompt": prompt,
            "model": self.generation_models[aspect.value]["model"],
            "lora": self.generation_models[aspect.value]["lora"],
            "lora_strength": self.generation_models[aspect.value]["strength"],
            "seed": seed,
            "steps": 30,
            "guidance_scale": 7.5,
            "resolution": "1024x1024"
        }
        
        # Mock generation result
        result = {
            "success": True,
            "persona_id": persona_id,
            "aspect": aspect.value,
            "mood": mood,
            "pose": pose,
            "generation_params": generation_params,
            "image_data": {
                "image_base64": f"mock_consistent_{persona_id}_{aspect.value}",
                "metadata": {
                    "generation_time": datetime.now().isoformat(),
                    "seed": seed,
                    "prompt": prompt
                }
            }
        }
        
        # Record generation
        self._record_generation(persona_id, result)
        
        return result
    
    def _build_character_prompt(self, profile: Dict, aspect: CharacterAspect, mood: str, pose: str) -> str:
        """Build consistent character prompt"""
        base_appearance = profile["base_appearance"]
        style_prefs = profile["style_preferences"]
        personality = profile["personality_traits"]
        
        # Base character description
        prompt_parts = [
            f"beautiful woman, {base_appearance['age_range']}",
            f"{base_appearance['hair_color']} {base_appearance['hair_style']} hair",
            f"{base_appearance['eye_color']} eyes",
            f"{base_appearance['skin_tone']} skin",
            f"{base_appearance['height']} height, {base_appearance['build']} build"
        ]
        
        # Add aspect-specific details
        if aspect == CharacterAspect.FACE:
            prompt_parts.extend([
                "close-up portrait",
                f"{personality['expression_style']} expression",
                f"{personality['mood_indicator']}"
            ])
        elif aspect == CharacterAspect.BODY:
            prompt_parts.extend([
                "full body shot",
                f"{pose} pose",
                f"{style_prefs['clothing_style']} clothing"
            ])
        elif aspect == CharacterAspect.FULL:
            prompt_parts.extend([
                "full body portrait",
                f"{personality['expression_style']} expression",
                f"{pose} pose",
                f"{style_prefs['clothing_style']} clothing",
                f"{personality['mood_indicator']}"
            ])
        
        # Add mood and style
        prompt_parts.extend([
            f"{mood} mood",
            f"{personality['pose_style']}",
            "high quality, detailed, beautiful"
        ])
        
        return ", ".join(prompt_parts)
    
    def update_character_appearance(self, persona_id: str, updates: Dict) -> Dict:
        """Allow persona to update their appearance"""
        if persona_id not in self.character_profiles:
            return {"error": "Character not found"}
        
        profile = self.character_profiles[persona_id]
        
        # Update base appearance
        if "base_appearance" in updates:
            profile["base_appearance"].update(updates["base_appearance"])
        
        # Update style preferences
        if "style_preferences" in updates:
            profile["style_preferences"].update(updates["style_preferences"])
        
        # Update personality traits
        if "personality_traits" in updates:
            profile["personality_traits"].update(updates["personality_traits"])
        
        # Add customizations
        if "customizations" in updates:
            profile["customizations"].update(updates["customizations"])
        
        # Update timestamp
        if persona_id in self.character_seeds:
            self.character_seeds[persona_id].last_updated = datetime.now()
        
        return {
            "message": f"Character appearance updated for {persona_id}",
            "updated_profile": profile
        }
    
    def generate_character_video(self, persona_id: str, action: str, duration: int = 3) -> Dict:
        """Generate consistent character video"""
        if persona_id not in self.character_seeds:
            return {"error": "Character not initialized"}
        
        character_profile = self.character_profiles[persona_id]
        
        # Build video prompt
        video_prompt = self._build_character_prompt(character_profile, CharacterAspect.FULL, "neutral", "standing")
        video_prompt += f", {action}, smooth motion, consistent character"
        
        # Get base seed for consistency
        seed = self.character_seeds[persona_id].base_seed
        
        generation_params = {
            "prompt": video_prompt,
            "model": "stable-video-diffusion",
            "lora": "consistent_character_video_v1",
            "lora_strength": 0.8,
            "seed": seed,
            "fps": 24,
            "duration": duration,
            "resolution": "1024x1024"
        }
        
        result = {
            "success": True,
            "persona_id": persona_id,
            "action": action,
            "generation_params": generation_params,
            "video_data": {
                "video_base64": f"mock_consistent_{persona_id}_video",
                "metadata": {
                    "generation_time": datetime.now().isoformat(),
                    "seed": seed,
                    "prompt": video_prompt
                }
            }
        }
        
        self._record_generation(persona_id, result)
        
        return result
    
    def _record_generation(self, persona_id: str, result: Dict):
        """Record character generation"""
        if persona_id in self.character_profiles:
            self.character_profiles[persona_id]["generation_history"].append({
                "timestamp": datetime.now().isoformat(),
                "aspect": result.get("aspect", "video"),
                "mood": result.get("mood", "neutral"),
                "action": result.get("action", "none")
            })
            
            # Keep only last 50 generations
            if len(self.character_profiles[persona_id]["generation_history"]) > 50:
                self.character_profiles[persona_id]["generation_history"] = \
                    self.character_profiles[persona_id]["generation_history"][-50:]
    
    def get_character_profile(self, persona_id: str) -> Dict:
        """Get character profile"""
        if persona_id not in self.character_profiles:
            return {"error": "Character not found"}
        
        return {
            "message": f"Character profile for {persona_id}",
            "profile": self.character_profiles[persona_id]
        }
    
    def get_character_generation_history(self, persona_id: str, limit: int = 10) -> Dict:
        """Get character generation history"""
        if persona_id not in self.character_profiles:
            return {"error": "Character not found"}
        
        history = self.character_profiles[persona_id]["generation_history"][-limit:]
        
        return {
            "message": f"Generation history for {persona_id}",
            "history": history
        }

# Global character generator instance
consistent_character_generator = ConsistentCharacterGenerator() 
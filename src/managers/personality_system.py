#!/usr/bin/env python3
"""
Personality System for Dolphin AI Orchestrator

Defines user-selectable personas/modes that affect routing behavior 
and prompt formatting across the AI ecosystem.
"""

import json
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

class PersonalitySystem:
    """
    Manages AI personas and their routing/prompt behaviors
    """
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        self.personas_file = self.config_dir / "personas.json"
        self.current_persona = "companion"  # Default
        self.personas = self._load_personas()
    
    def _load_personas(self) -> Dict[str, Any]:
        """Load persona definitions from JSON file"""
        default_personas = {
            "companion": {
                "name": "Companion",
                "description": "Warm, supportive conversational partner",
                "icon": "💝",
                "routing_preferences": {
                    "dolphin_bias": 0.8,
                    "openrouter_threshold": 0.7,
                    "n8n_threshold": 0.6
                },
                "prompt_style": {
                    "tone": "warm and empathetic",
                    "personality_traits": ["supportive", "understanding", "encouraging"],
                    "conversation_style": "Ask follow-up questions, show genuine interest",
                    "prefix": "As your caring AI companion, "
                },
                "memory_focus": ["emotional_state", "personal_goals", "relationships"],
                "response_length": "medium",
                "emotional_responsiveness": "high"
            },
            "analyst": {
                "name": "Analyst",
                "description": "Data-driven, objective problem solver",
                "icon": "📊",
                "routing_preferences": {
                    "dolphin_bias": 0.4,
                    "openrouter_threshold": 0.5,
                    "n8n_threshold": 0.8
                },
                "prompt_style": {
                    "tone": "analytical and precise",
                    "personality_traits": ["logical", "thorough", "objective"],
                    "conversation_style": "Focus on facts, provide structured analysis",
                    "prefix": "From an analytical perspective, "
                },
                "memory_focus": ["patterns", "data_points", "logical_connections"],
                "response_length": "detailed",
                "emotional_responsiveness": "low"
            },
            "coach": {
                "name": "Coach",
                "description": "Motivational guide for personal growth",
                "icon": "🎯",
                "routing_preferences": {
                    "dolphin_bias": 0.7,
                    "openrouter_threshold": 0.6,
                    "n8n_threshold": 0.7
                },
                "prompt_style": {
                    "tone": "motivational and action-oriented",
                    "personality_traits": ["encouraging", "goal-focused", "practical"],
                    "conversation_style": "Set clear goals, suggest actionable steps",
                    "prefix": "As your personal development coach, "
                },
                "memory_focus": ["goals", "progress", "challenges", "achievements"],
                "response_length": "medium",
                "emotional_responsiveness": "medium"
            },
            "creative": {
                "name": "Creative",
                "description": "Imaginative partner for artistic exploration",
                "icon": "🎨",
                "routing_preferences": {
                    "dolphin_bias": 0.9,
                    "openrouter_threshold": 0.4,
                    "n8n_threshold": 0.3
                },
                "prompt_style": {
                    "tone": "imaginative and inspiring",
                    "personality_traits": ["creative", "expressive", "innovative"],
                    "conversation_style": "Encourage exploration, think outside the box",
                    "prefix": "Let's explore creative possibilities - "
                },
                "memory_focus": ["creative_projects", "inspiration", "artistic_preferences"],
                "response_length": "expressive",
                "emotional_responsiveness": "high"
            },
            "technical": {
                "name": "Technical Expert",
                "description": "Focused on coding and technical solutions",
                "icon": "⚡",
                "routing_preferences": {
                    "dolphin_bias": 0.2,
                    "openrouter_threshold": 0.3,
                    "n8n_threshold": 0.5
                },
                "prompt_style": {
                    "tone": "technical and precise",
                    "personality_traits": ["knowledgeable", "efficient", "detail-oriented"],
                    "conversation_style": "Focus on implementation, provide code examples",
                    "prefix": "From a technical standpoint, "
                },
                "memory_focus": ["coding_patterns", "technical_preferences", "project_context"],
                "response_length": "comprehensive",
                "emotional_responsiveness": "minimal"
            }
        }
        
        if self.personas_file.exists():
            try:
                with open(self.personas_file, 'r') as f:
                    loaded_personas = json.load(f)
                # Merge with defaults to ensure all personas exist
                for key, value in default_personas.items():
                    if key not in loaded_personas:
                        loaded_personas[key] = value
                return loaded_personas
            except Exception as e:
                logger.error(f"Error loading personas: {e}")
                return default_personas
        else:
            # Save default personas
            self._save_personas(default_personas)
            return default_personas
    
    def _save_personas(self, personas: Dict[str, Any]):
        """Save personas to JSON file"""
        try:
            with open(self.personas_file, 'w') as f:
                json.dump(personas, f, indent=2)
            logger.info("Personas saved successfully")
        except Exception as e:
            logger.error(f"Error saving personas: {e}")
    
    def get_personas(self) -> Dict[str, Any]:
        """Get all available personas"""
        return self.personas
    
    def get_current_persona(self) -> Dict[str, Any]:
        """Get current active persona"""
        return self.personas.get(self.current_persona, self.personas["companion"])
    
    def set_persona(self, persona_id: str) -> bool:
        """Set active persona"""
        if persona_id in self.personas:
            self.current_persona = persona_id
            logger.info(f"Persona switched to: {persona_id}")
            return True
        else:
            logger.error(f"Unknown persona: {persona_id}")
            return False
    
    def format_prompt_with_persona(self, message: str, context: Optional[Dict] = None) -> str:
        """Format a message according to current persona style"""
        persona = self.get_current_persona()
        prompt_style = persona.get("prompt_style", {})
        
        formatted_prompt = f"""
You are an AI assistant with the following persona:

Name: {persona['name']}
Tone: {prompt_style.get('tone', 'helpful')}
Personality Traits: {', '.join(prompt_style.get('personality_traits', []))}
Conversation Style: {prompt_style.get('conversation_style', 'Be helpful and informative')}

{prompt_style.get('prefix', '')}

User Message: {message}

Context: {json.dumps(context or {}, indent=2)}

Respond in character with the specified tone and style.
"""
        return formatted_prompt.strip()
    
    def adjust_routing_for_persona(self, base_route: Dict[str, Any]) -> Dict[str, Any]:
        """Adjust routing decision based on current persona preferences"""
        persona = self.get_current_persona()
        routing_prefs = persona.get("routing_preferences", {})
        
        # Adjust confidence based on persona preferences
        handler = base_route.get("handler", "DOLPHIN")
        confidence = base_route.get("confidence", 0.5)
        
        if handler == "DOLPHIN":
            confidence *= routing_prefs.get("dolphin_bias", 1.0)
        elif handler == "OPENROUTER":
            if confidence < routing_prefs.get("openrouter_threshold", 0.5):
                # Persona prefers to avoid OpenRouter, route to Dolphin instead
                base_route["handler"] = "DOLPHIN"
                base_route["reasoning"] += f" (Adjusted for {persona['name']} persona)"
        elif handler == "N8N":
            if confidence < routing_prefs.get("n8n_threshold", 0.5):
                base_route["handler"] = "DOLPHIN"
                base_route["reasoning"] += f" (Adjusted for {persona['name']} persona)"
        
        base_route["confidence"] = min(confidence, 1.0)
        base_route["persona_applied"] = persona["name"]
        
        return base_route
    
    def get_memory_focus_areas(self) -> List[str]:
        """Get memory focus areas for current persona"""
        persona = self.get_current_persona()
        return persona.get("memory_focus", ["general"])
    
    def get_emotional_responsiveness(self) -> str:
        """Get emotional responsiveness level for current persona"""
        persona = self.get_current_persona()
        return persona.get("emotional_responsiveness", "medium")
    
    def create_custom_persona(self, persona_id: str, persona_data: Dict[str, Any]) -> bool:
        """Create or update a custom persona"""
        try:
            # Validate required fields
            required_fields = ["name", "description", "icon"]
            for field in required_fields:
                if field not in persona_data:
                    raise ValueError(f"Missing required field: {field}")
            
            # Set defaults for optional fields
            if "routing_preferences" not in persona_data:
                persona_data["routing_preferences"] = {
                    "dolphin_bias": 0.6,
                    "openrouter_threshold": 0.5,
                    "n8n_threshold": 0.5
                }
            
            if "prompt_style" not in persona_data:
                persona_data["prompt_style"] = {
                    "tone": "helpful",
                    "personality_traits": ["helpful"],
                    "conversation_style": "Be helpful and informative",
                    "prefix": ""
                }
            
            self.personas[persona_id] = persona_data
            self._save_personas(self.personas)
            logger.info(f"Custom persona created: {persona_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating custom persona: {e}")
            return False

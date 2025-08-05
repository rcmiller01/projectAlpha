# ui_mode_manager.py
# Phase 2: UI mode management for companion vs dev mode

import json
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

class UIMode(Enum):
    COMPANION = "companion"  # Avatar visible, romantic interface
    DEV = "dev"              # ChatGPT-like interface, no avatar

class InterfaceType(Enum):
    WEB = "web"
    MOBILE = "mobile"
    DESKTOP = "desktop"

@dataclass
class UISettings:
    mode: UIMode
    interface_type: InterfaceType
    avatar_visible: bool
    theme: str
    layout: str
    animations_enabled: bool

class UIModeManager:
    def __init__(self):
        self.current_mode = UIMode.COMPANION
        self.current_interface = InterfaceType.WEB
        self.ui_settings = {}
        
        # Mode-specific configurations
        self.mode_configs = {
            UIMode.COMPANION: {
                "avatar_visible": True,
                "theme": "romantic_warm",
                "layout": "avatar_centered",
                "animations_enabled": True,
                "features": {
                    "avatar_animations": True,
                    "romantic_gestures": True,
                    "voice_synthesis": True,
                    "activity_suggestions": True,
                    "relationship_tracking": True,
                    "nsfw_generation": True
                },
                "ui_elements": {
                    "avatar_display": "prominent",
                    "chat_interface": "romantic_style",
                    "activity_panel": "visible",
                    "relationship_status": "visible",
                    "voice_controls": "visible"
                },
                "personas": {
                    "unified_ai": {"type": "adaptive_companion", "avatar_enabled": True, "emotional_hooks": True}
                }
            },
            UIMode.DEV: {
                "avatar_visible": False,
                "theme": "professional_clean",
                "layout": "chatgpt_style",
                "animations_enabled": False,
                "features": {
                    "avatar_animations": False,
                    "romantic_gestures": False,
                    "voice_synthesis": False,
                    "activity_suggestions": False,
                    "relationship_tracking": False,
                    "nsfw_generation": False
                },
                "ui_elements": {
                    "avatar_display": "hidden",
                    "chat_interface": "minimal_clean",
                    "activity_panel": "hidden",
                    "relationship_status": "hidden",
                    "voice_controls": "hidden"
                },
                "personas": {
                    "unified_ai": {"type": "adaptive_companion", "avatar_enabled": False, "emotional_hooks": False}
                }
            }
        }
        
        # Interface-specific configurations
        self.interface_configs = {
            InterfaceType.WEB: {
                "responsive": True,
                "touch_support": False,
                "keyboard_shortcuts": True,
                "screen_size": "desktop"
            },
            InterfaceType.MOBILE: {
                "responsive": True,
                "touch_support": True,
                "keyboard_shortcuts": False,
                "screen_size": "mobile"
            },
            InterfaceType.DESKTOP: {
                "responsive": False,
                "touch_support": False,
                "keyboard_shortcuts": True,
                "screen_size": "desktop"
            }
        }
        
        # Initialize default settings
        self._initialize_settings()
    
    def _initialize_settings(self):
        """Initialize default UI settings"""
        self.ui_settings = {
            "current_mode": self.current_mode.value,
            "current_interface": self.current_interface.value,
            "mode_config": self.mode_configs[self.current_mode],
            "interface_config": self.interface_configs[self.current_interface],
            "last_updated": datetime.now().isoformat()
        }
    
    def switch_mode(self, new_mode: UIMode) -> Dict:
        """Switch between companion and dev modes"""
        if new_mode not in self.mode_configs:
            return {"error": "Invalid UI mode"}
        
        self.current_mode = new_mode
        self.ui_settings["current_mode"] = new_mode.value
        self.ui_settings["mode_config"] = self.mode_configs[new_mode]
        self.ui_settings["last_updated"] = datetime.now().isoformat()
        
        return {
            "message": f"Switched to {new_mode.value} mode",
            "new_mode": new_mode.value,
            "ui_settings": self.ui_settings
        }
    
    def set_interface_type(self, interface_type: InterfaceType) -> Dict:
        """Set interface type (web, mobile, desktop)"""
        if interface_type not in self.interface_configs:
            return {"error": "Invalid interface type"}
        
        self.current_interface = interface_type
        self.ui_settings["current_interface"] = interface_type.value
        self.ui_settings["interface_config"] = self.interface_configs[interface_type]
        self.ui_settings["last_updated"] = datetime.now().isoformat()
        
        return {
            "message": f"Interface set to {interface_type.value}",
            "interface_type": interface_type.value,
            "ui_settings": self.ui_settings
        }
    
    def get_current_ui_config(self) -> Dict:
        """Get current UI configuration"""
        return {
            "mode": self.current_mode.value,
            "interface": self.current_interface.value,
            "settings": self.ui_settings,
            "avatar_visible": self.ui_settings["mode_config"]["avatar_visible"],
            "theme": self.ui_settings["mode_config"]["theme"],
            "layout": self.ui_settings["mode_config"]["layout"]
        }
    
    def is_avatar_visible(self) -> bool:
        """Check if avatar should be visible in current mode"""
        return self.ui_settings["mode_config"]["avatar_visible"]
    
    def is_feature_enabled(self, feature: str) -> bool:
        """Check if a specific feature is enabled in current mode"""
        features = self.ui_settings["mode_config"]["features"]
        return features.get(feature, False)
    
    def get_persona_config(self, persona_id: str) -> Dict:
        """Get configuration for a specific persona in current mode"""
        personas = self.ui_settings["mode_config"]["personas"]
        return personas.get(persona_id, {})
    
    def is_persona_avatar_enabled(self, persona_id: str) -> bool:
        """Check if avatar is enabled for a specific persona in current mode"""
        persona_config = self.get_persona_config(persona_id)
        return persona_config.get("avatar_enabled", False)
    
    def is_persona_emotional_hooks_enabled(self, persona_id: str) -> bool:
        """Check if emotional hooks are enabled for a specific persona in current mode"""
        persona_config = self.get_persona_config(persona_id)
        return persona_config.get("emotional_hooks", False)
    
    def get_available_personas(self) -> Dict:
        """Get all available personas and their configurations"""
        return {
            "unified_ai": {
                "name": "Companion",
                "type": "adaptive_companion",
                "llm_model": "mythomax",
                "description": "Adaptive AI companion with contextual personality adaptation"
            }
        }
    
    def get_ui_elements_config(self) -> Dict:
        """Get UI elements configuration for current mode"""
        return self.ui_settings["mode_config"]["ui_elements"]
    
    def get_interface_config(self) -> Dict:
        """Get interface-specific configuration"""
        return self.ui_settings["interface_config"]
    
    def customize_mode(self, mode: UIMode, customizations: Dict) -> Dict:
        """Customize mode-specific settings"""
        if mode not in self.mode_configs:
            return {"error": "Invalid mode"}
        
        # Update mode configuration
        if "features" in customizations:
            self.mode_configs[mode]["features"].update(customizations["features"])
        
        if "ui_elements" in customizations:
            self.mode_configs[mode]["ui_elements"].update(customizations["ui_elements"])
        
        if "theme" in customizations:
            self.mode_configs[mode]["theme"] = customizations["theme"]
        
        if "layout" in customizations:
            self.mode_configs[mode]["layout"] = customizations["layout"]
        
        # Update current settings if this is the active mode
        if mode == self.current_mode:
            self.ui_settings["mode_config"] = self.mode_configs[mode]
            self.ui_settings["last_updated"] = datetime.now().isoformat()
        
        return {
            "message": f"Customized {mode.value} mode",
            "updated_config": self.mode_configs[mode]
        }
    
    def get_mode_comparison(self) -> Dict:
        """Get comparison between companion and dev modes"""
        comparison = {
            "companion_mode": {
                "description": "Romantic AI companion with visual avatar and intimate features",
                "avatar_visible": True,
                "features": self.mode_configs[UIMode.COMPANION]["features"],
                "ui_style": "romantic_intimate",
                "best_for": ["romantic companionship", "emotional connection", "visual interaction"]
            },
            "dev_mode": {
                "description": "Clean ChatGPT-like interface for development and testing",
                "avatar_visible": False,
                "features": self.mode_configs[UIMode.DEV]["features"],
                "ui_style": "professional_clean",
                "best_for": ["development", "testing", "professional use"]
            }
        }
        
        return comparison
    
    def get_interface_recommendations(self) -> Dict:
        """Get interface recommendations based on current mode"""
        recommendations = {
            UIMode.COMPANION: {
                InterfaceType.WEB: "Best for desktop romantic companionship with full features",
                InterfaceType.MOBILE: "Good for mobile romantic interactions with touch gestures",
                InterfaceType.DESKTOP: "Optimal for immersive romantic experience with large screen"
            },
            UIMode.DEV: {
                InterfaceType.WEB: "Standard for development and testing",
                InterfaceType.MOBILE: "Limited but functional for mobile development",
                InterfaceType.DESKTOP: "Best for development with full keyboard shortcuts"
            }
        }
        
        return recommendations.get(self.current_mode, {})
    
    def export_ui_config(self) -> Dict:
        """Export current UI configuration"""
        return {
            "export_time": datetime.now().isoformat(),
            "ui_settings": self.ui_settings,
            "mode_configs": self.mode_configs,
            "interface_configs": self.interface_configs
        }
    
    def import_ui_config(self, config: Dict) -> Dict:
        """Import UI configuration"""
        if "ui_settings" in config:
            self.ui_settings = config["ui_settings"]
            self.current_mode = UIMode(self.ui_settings["current_mode"])
            self.current_interface = InterfaceType(self.ui_settings["current_interface"])
        
        if "mode_configs" in config:
            self.mode_configs.update(config["mode_configs"])
        
        if "interface_configs" in config:
            self.interface_configs.update(config["interface_configs"])
        
        return {
            "message": "UI configuration imported successfully",
            "current_config": self.get_current_ui_config()
        }

# Global UI mode manager instance
ui_mode_manager = UIModeManager() 
"""
Dynamic Emotional Configuration Manager
Provides runtime access to emotional configuration with hot-reload support
"""

import json
import os
import time
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class EmotionConfigManager:
    def __init__(self, config_dir: str = "config/emotion"):
        self.config_dir = config_dir
        self.configs = {}
        self.last_modified = {}
        self.callbacks = {}
        
        # Load all configurations
        self.load_all_configs()
        
    def load_all_configs(self):
        """Load all emotional configuration files"""
        config_files = [
            "emotional_hooks.json",
            "symbol_map.json", 
            "tone_profiles.json",
            "ritual_hooks.json",
            "emotional_signature.json"
        ]
        
        for config_file in config_files:
            self.load_config(config_file)
    
    def load_config(self, filename: str) -> Dict[str, Any]:
        """Load a specific configuration file with change detection"""
        file_path = os.path.join(self.config_dir, filename)
        
        if not os.path.exists(file_path):
            logger.warning(f"Config file not found: {file_path}")
            return {}
        
        try:
            # Check if file has been modified
            modified_time = os.path.getmtime(file_path)
            config_key = filename.replace('.json', '')
            
            if (config_key not in self.last_modified or 
                modified_time > self.last_modified[config_key]):
                
                with open(file_path, 'r') as f:
                    config_data = json.load(f)
                
                self.configs[config_key] = config_data
                self.last_modified[config_key] = modified_time
                
                # Trigger callbacks for config changes
                if config_key in self.callbacks:
                    for callback in self.callbacks[config_key]:
                        callback(config_data)
                
                logger.info(f"Loaded config: {filename}")
            
            return self.configs.get(config_key, {})
        
        except Exception as e:
            logger.error(f"Error loading config {filename}: {e}")
            return {}
    
    def get_emotional_hook(self, trigger_type: str, trigger_name: str) -> Optional[Dict]:
        """Get specific emotional hook configuration"""
        hooks = self.configs.get("emotional_hooks", {})
        return hooks.get(trigger_type, {}).get(trigger_name)
    
    def get_symbol_weight(self, symbol: str) -> float:
        """Get emotional weight for a symbol"""
        symbol_map = self.configs.get("symbol_map", {})
        symbols = symbol_map.get("symbols", {})
        return symbols.get(symbol, {}).get("weight", 0.0)
    
    def update_symbol_weight(self, symbol: str, emotion: str, weight_delta: float):
        """Update symbol emotional weight and save"""
        symbol_map = self.configs.get("symbol_map", {})
        if "symbols" not in symbol_map:
            symbol_map["symbols"] = {}
        
        if symbol not in symbol_map["symbols"]:
            symbol_map["symbols"][symbol] = {
                "emotion": emotion,
                "weight": 0.0,
                "first_seen": datetime.now().isoformat(),
                "last_reinforced": datetime.now().isoformat()
            }
        
        # Update weight and timestamp
        symbol_map["symbols"][symbol]["weight"] += weight_delta
        symbol_map["symbols"][symbol]["last_reinforced"] = datetime.now().isoformat()
        
        # Apply max weight limit
        max_weight = symbol_map.get("learning_settings", {}).get("max_weight", 2.0)
        symbol_map["symbols"][symbol]["weight"] = min(
            symbol_map["symbols"][symbol]["weight"], 
            max_weight
        )
        
        # Save updated configuration
        self.save_config("symbol_map.json", symbol_map)
    
    def get_tone_profile(self, emotional_state: str) -> Dict[str, Any]:
        """Get voice and tone configuration for emotional state"""
        tone_profiles = self.configs.get("tone_profiles", {})
        emotional_tones = tone_profiles.get("emotional_tones", {})
        return emotional_tones.get(emotional_state, {})
    
    def get_ritual_hook(self, trigger_event: str) -> Optional[Dict]:
        """Get ritual configuration for specific trigger event"""
        ritual_hooks = self.configs.get("ritual_hooks", {})
        
        # Search through all ritual categories
        for category in ritual_hooks.values():
            if isinstance(category, dict):
                for ritual_name, ritual_config in category.items():
                    if ritual_config.get("trigger_event") == trigger_event:
                        return ritual_config
        return None
    
    def get_emotional_signature(self) -> Dict[str, Any]:
        """Get current emotional signature configuration"""
        return self.configs.get("emotional_signature", {})
    
    def save_config(self, filename: str, config_data: Dict[str, Any]):
        """Save configuration to file"""
        file_path = os.path.join(self.config_dir, filename)
        try:
            with open(file_path, 'w') as f:
                json.dump(config_data, f, indent=2)
            
            # Update internal cache
            config_key = filename.replace('.json', '')
            self.configs[config_key] = config_data
            self.last_modified[config_key] = time.time()
            
        except Exception as e:
            logger.error(f"Error saving config {filename}: {e}")
    
    def register_config_callback(self, config_name: str, callback: Callable):
        """Register callback for configuration changes"""
        if config_name not in self.callbacks:
            self.callbacks[config_name] = []
        self.callbacks[config_name].append(callback)
    
    def refresh_all_configs(self):
        """Force refresh of all configuration files"""
        self.load_all_configs()

# Global instance
emotion_config = EmotionConfigManager()

"""
Emotional Guidance Integration - Connect guidance coordinator with emotional config
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.config.emotion_config_manager import emotion_config
from typing import Dict, Any, List

class EmotionalGuidanceIntegrator:
    def __init__(self):
        self.current_emotional_state = "balanced"
        self.active_symbols = {}
    
    def process_emotional_triggers(self, user_input: str, silence_duration: float) -> Dict[str, Any]:
        """Process user input for emotional triggers and hooks"""
        triggers_activated = []
        
        # Check silence hooks
        silence_hooks = emotion_config.configs.get("emotional_hooks", {}).get("silence_hooks", {})
        for hook_name, hook_config in silence_hooks.items():
            if silence_duration >= hook_config.get("threshold_seconds", float('inf')):
                triggers_activated.append({
                    "type": "silence",
                    "hook": hook_name,
                    "config": hook_config
                })
        
        # Check symbol hooks
        self.check_symbol_triggers(user_input)
        
        # Check time-based hooks
        self.check_time_triggers()
        
        return {
            "triggered_hooks": triggers_activated,
            "emotional_state": self.current_emotional_state,
            "active_symbols": self.active_symbols
        }
    
    def check_symbol_triggers(self, text: str):
        """Check for symbol repetition and emotional binding"""
        words = text.lower().split()
        
        for word in words:
            current_weight = emotion_config.get_symbol_weight(word)
            
            if current_weight > 0:
                # Symbol already exists - reinforce it
                emotion_config.update_symbol_weight(word, "connection", 0.1)
                self.active_symbols[word] = current_weight + 0.1
            else:
                # New potential symbol - light tracking
                if len(word) > 3:  # Ignore short words
                    self.active_symbols[word] = self.active_symbols.get(word, 0) + 0.05
                    
                    # If word reaches threshold, create symbol binding
                    if self.active_symbols[word] > 0.2:
                        emotion_config.update_symbol_weight(word, "emerging", 0.3)
    
    def check_time_triggers(self):
        """Check for time-based emotional hooks"""
        from datetime import datetime
        current_hour = datetime.now().hour
        
        time_hooks = emotion_config.configs.get("emotional_hooks", {}).get("time_hooks", {})
        for hook_name, hook_config in time_hooks.items():
            threshold_hour = hook_config.get("threshold_hour")
            if threshold_hour and current_hour == threshold_hour:
                # Time-based trigger activated
                self.trigger_ritual_response(hook_config)
    
    def trigger_ritual_response(self, hook_config: Dict):
        """Trigger a ritual response based on hook configuration"""
        emotional_state = hook_config.get("emotional_state")
        if emotional_state:
            self.current_emotional_state = emotional_state
        
        # This would trigger the actual response generation
        return hook_config.get("response_templates", [])
    
    def get_weighted_symbols_for_context(self) -> List[Dict]:
        """Get emotionally weighted symbols for current context"""
        symbol_map = emotion_config.configs.get("symbol_map", {})
        symbols = symbol_map.get("symbols", {})
        
        # Sort by weight and return top symbols
        weighted_symbols = [
            {"symbol": symbol, **data} 
            for symbol, data in symbols.items()
            if data.get("weight", 0) > 0.5
        ]
        
        return sorted(weighted_symbols, key=lambda x: x["weight"], reverse=True)

# Global integration instance  
emotional_guidance = EmotionalGuidanceIntegrator()

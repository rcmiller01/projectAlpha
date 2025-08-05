"""
Attachment Reflector - Transparency Module for Symbol Binding History

This module provides transparency into the emotional attachment and symbolic
evolution process, allowing users to understand how symbols accumulate
meaning and emotional weight over time.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from .memory_manager import memory_manager

logger = logging.getLogger(__name__)

@dataclass
class SymbolReflection:
    """Reflection data for a symbol's emotional journey"""
    symbol: str
    current_weight: float
    base_meaning: str
    drifted_meaning: str
    usage_count: int
    first_encounter: float
    last_usage: float
    emotional_journey: List[str]
    meaning_evolution: List[str]
    decay_resistance: float
    
class AttachmentReflector:
    """
    Provides transparency into symbol binding and attachment processes
    
    This module allows users to understand:
    - How symbols accumulate emotional weight
    - The evolution of meaning over time
    - Which symbols hold the deepest significance
    - The emotional journey of specific symbols
    """
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}")
        
    def get_symbol_reflection(self, symbol: str) -> Optional[SymbolReflection]:
        """
        Get detailed reflection data for a specific symbol
        
        Args:
            symbol: The symbol to reflect on
            
        Returns:
            SymbolReflection with detailed history, or None if symbol not found
        """
        try:
            if not memory_manager or not hasattr(memory_manager, 'symbol_binding_map'):
                return None
                
            binding = memory_manager.symbol_binding_map.get(symbol)
            if not binding:
                return None
                
            # Create emotional journey from associated emotions
            emotional_journey = list(binding.associated_emotions.keys())
            
            # Create meaning evolution timeline
            meaning_evolution = []
            if binding.base_meaning:
                meaning_evolution.append(f"Initial: {binding.base_meaning}")
            if binding.drifted_meaning and binding.drifted_meaning != binding.base_meaning:
                meaning_evolution.append(f"Evolved: {binding.drifted_meaning}")
                
            return SymbolReflection(
                symbol=symbol,
                current_weight=binding.emotional_weight,
                base_meaning=binding.base_meaning,
                drifted_meaning=binding.drifted_meaning,
                usage_count=binding.usage_count,
                first_encounter=binding.first_encounter,
                last_usage=binding.last_usage,
                emotional_journey=emotional_journey,
                meaning_evolution=meaning_evolution,
                decay_resistance=binding.decay_resistance
            )
            
        except Exception as e:
            self.logger.error(f"Error creating symbol reflection for '{symbol}': {e}")
            return None
    
    def get_attachment_landscape(self, minimum_weight: float = 0.1) -> Dict[str, Any]:
        """
        Get overview of the current attachment landscape
        
        Args:
            minimum_weight: Minimum emotional weight to include
            
        Returns:
            Dictionary containing attachment analytics and symbol categories
        """
        try:
            if not memory_manager:
                return {"available": False, "reason": "Memory manager not available"}
                
            weighted_symbols = memory_manager.get_emotionally_weighted_symbols(minimum_weight)
            
            if not weighted_symbols:
                return {
                    "available": True,
                    "total_symbols": 0,
                    "message": "No symbols have developed emotional significance yet"
                }
            
            # Categorize symbols by emotional weight
            categories = {
                "deeply_attached": [],    # >= 0.7
                "moderately_attached": [], # 0.4 - 0.69
                "lightly_attached": [],   # 0.1 - 0.39
            }
            
            total_emotional_weight = 0
            most_recent = None
            oldest = None
            
            for symbol, binding in weighted_symbols.items():
                total_emotional_weight += binding.emotional_weight
                
                # Track temporal bounds
                if most_recent is None or binding.last_usage > most_recent:
                    most_recent = binding.last_usage
                if oldest is None or binding.first_encounter < oldest:
                    oldest = binding.first_encounter
                
                # Categorize by weight
                symbol_info = {
                    "symbol": symbol,
                    "weight": binding.emotional_weight,
                    "meaning": binding.drifted_meaning or binding.base_meaning,
                    "usage_count": binding.usage_count,
                    "emotions": list(binding.associated_emotions.keys())
                }
                
                if binding.emotional_weight >= 0.7:
                    categories["deeply_attached"].append(symbol_info)
                elif binding.emotional_weight >= 0.4:
                    categories["moderately_attached"].append(symbol_info)
                else:
                    categories["lightly_attached"].append(symbol_info)
            
            # Sort each category by weight
            for category in categories.values():
                category.sort(key=lambda x: x["weight"], reverse=True)
            
            return {
                "available": True,
                "total_symbols": len(weighted_symbols),
                "categories": categories,
                "analytics": {
                    "total_emotional_weight": round(total_emotional_weight, 3),
                    "average_weight": round(total_emotional_weight / len(weighted_symbols), 3),
                    "attachment_depth": "deep" if any(b.emotional_weight >= 0.7 for b in weighted_symbols.values()) else
                                     "moderate" if any(b.emotional_weight >= 0.4 for b in weighted_symbols.values()) else "light",
                    "oldest_attachment": datetime.fromtimestamp(oldest).isoformat() if oldest else None,
                    "most_recent_reinforcement": datetime.fromtimestamp(most_recent).isoformat() if most_recent else None,
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error generating attachment landscape: {e}")
            return {"available": False, "error": str(e)}
    
    def get_symbol_evolution_timeline(self, symbol: str) -> List[Dict[str, Any]]:
        """
        Get timeline of how a symbol's meaning has evolved
        
        Args:
            symbol: Symbol to trace evolution for
            
        Returns:
            List of evolution events in chronological order
        """
        try:
            if not memory_manager or not hasattr(memory_manager, 'symbol_binding_map'):
                return []
                
            binding = memory_manager.symbol_binding_map.get(symbol)
            if not binding:
                return []
            
            timeline = []
            
            # Add initial attachment event
            timeline.append({
                "event": "initial_attachment",
                "timestamp": datetime.fromtimestamp(binding.first_encounter).isoformat(),
                "meaning": binding.base_meaning,
                "emotional_weight": 0.1,  # Approximate initial weight
                "context": "Symbol first encountered and bound to emotional context"
            })
            
            # Add meaning evolution if it occurred
            if binding.drifted_meaning and binding.drifted_meaning != binding.base_meaning:
                # Estimate when drift occurred (halfway between first encounter and last usage)
                drift_time = binding.first_encounter + (binding.last_usage - binding.first_encounter) / 2
                timeline.append({
                    "event": "meaning_drift",
                    "timestamp": datetime.fromtimestamp(drift_time).isoformat(),
                    "meaning": binding.drifted_meaning,
                    "emotional_weight": binding.emotional_weight * 0.7,  # Approximate weight at drift
                    "context": "Symbol meaning evolved through repeated emotional contexts"
                })
            
            # Add recent reinforcement
            if binding.last_usage != binding.first_encounter:
                timeline.append({
                    "event": "recent_reinforcement",
                    "timestamp": datetime.fromtimestamp(binding.last_usage).isoformat(),
                    "meaning": binding.drifted_meaning or binding.base_meaning,
                    "emotional_weight": binding.emotional_weight,
                    "context": f"Symbol reinforced through usage (total: {binding.usage_count} times)"
                })
            
            return timeline
            
        except Exception as e:
            self.logger.error(f"Error creating evolution timeline for '{symbol}': {e}")
            return []
    
    def get_emotional_resonance_report(self) -> Dict[str, Any]:
        """
        Generate a report on emotional resonance patterns across symbols
        
        Returns:
            Report on how emotions cluster around symbols
        """
        try:
            if not memory_manager:
                return {"available": False}
                
            weighted_symbols = memory_manager.get_emotionally_weighted_symbols(0.1)
            
            if not weighted_symbols:
                return {"available": True, "emotional_patterns": {}}
            
            # Map emotions to symbols
            emotion_symbol_map = {}
            symbol_emotion_map = {}
            
            for symbol, binding in weighted_symbols.items():
                symbol_emotion_map[symbol] = list(binding.associated_emotions.keys())
                
                for emotion in binding.associated_emotions.keys():
                    if emotion not in emotion_symbol_map:
                        emotion_symbol_map[emotion] = []
                    emotion_symbol_map[emotion].append({
                        "symbol": symbol,
                        "weight": binding.emotional_weight
                    })
            
            # Sort symbols by weight for each emotion
            for emotion in emotion_symbol_map:
                emotion_symbol_map[emotion].sort(key=lambda x: x["weight"], reverse=True)
            
            # Find dominant emotional themes
            dominant_emotions = sorted(emotion_symbol_map.keys(), 
                                     key=lambda e: sum(s["weight"] for s in emotion_symbol_map[e]), 
                                     reverse=True)
            
            return {
                "available": True,
                "emotional_patterns": {
                    "emotion_to_symbols": emotion_symbol_map,
                    "symbol_to_emotions": symbol_emotion_map,
                    "dominant_emotions": dominant_emotions[:5],  # Top 5
                    "emotional_diversity": len(emotion_symbol_map),
                    "total_emotional_connections": sum(len(binding.associated_emotions) 
                                                     for binding in weighted_symbols.values())
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error generating emotional resonance report: {e}")
            return {"available": False, "error": str(e)}

# Global instance for easy access
attachment_reflector = AttachmentReflector()

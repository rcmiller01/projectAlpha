"""
Symbol Memory - Symbol tracking and decay analysis
Manages symbolic memory persistence and determines symbol dormancy levels
"""

import time
import math
import json
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta

@dataclass
class SymbolMetadata:
    """Metadata for symbolic memory tracking"""
    symbol: str
    last_used: float
    usage_frequency: int
    creation_time: float
    emotional_weight: float
    context_tags: List[str]
    decay_resistance: float = 1.0  # Higher = more resistant to decay

class SymbolMemory:
    """Manages symbolic memory with decay tracking"""
    
    def __init__(self, memory_path: str = "memory/symbol_memory.json"):
        self.memory_path = memory_path
        self.symbols: Dict[str, SymbolMetadata] = {}
        self.decay_constants = {
            "base_decay_rate": 1.0,        # Base decay per day
            "frequency_protection": 0.5,   # How much frequency protects from decay
            "emotional_protection": 0.3,   # How much emotional weight protects
            "recency_boost": 2.0,          # Boost for recently used symbols
            "dormancy_threshold": 0.7      # Above this score = dormant
        }
        self.load_from_memory()
    
    def symbol_decay_score(self, symbol: str, last_used: Optional[float] = None, 
                          usage_frequency: Optional[int] = None) -> float:
        """
        Determine how dormant a symbol is based on last-used and context frequency.
        
        Args:
            symbol: The symbol to analyze
            last_used: Override last used time (for testing)
            usage_frequency: Override frequency count (for testing)
            
        Returns:
            Decay score (0.0 = fresh, 1.0 = completely dormant)
        """
        
        # Get symbol metadata
        if symbol in self.symbols:
            metadata = self.symbols[symbol]
            actual_last_used = last_used if last_used is not None else metadata.last_used
            actual_frequency = usage_frequency if usage_frequency is not None else metadata.usage_frequency
            emotional_weight = metadata.emotional_weight
            decay_resistance = metadata.decay_resistance
        else:
            # Unknown symbol - treat as never used
            actual_last_used = last_used if last_used is not None else 0
            actual_frequency = usage_frequency if usage_frequency is not None else 0
            emotional_weight = 0.5
            decay_resistance = 1.0
        
        current_time = time.time()
        
        # Time-based decay
        age_seconds = current_time - actual_last_used
        age_days = age_seconds / 86400  # Convert to days
        
        # Base decay calculation
        base_decay = age_days * self.decay_constants["base_decay_rate"]
        
        # Frequency protection - higher frequency = more resistance
        frequency_factor = 1.0 / (actual_frequency + 1)
        frequency_penalty = frequency_factor * self.decay_constants["frequency_protection"]
        
        # Emotional weight protection - emotional symbols decay slower
        emotional_protection = emotional_weight * self.decay_constants["emotional_protection"]
        
        # Recency boost - recent usage dramatically reduces decay
        recency_factor = 1.0
        if age_days < 1:  # Used within last day
            recency_factor = 0.1
        elif age_days < 7:  # Used within last week
            recency_factor = 0.3 + (age_days - 1) * 0.1
        elif age_days < 30:  # Used within last month
            recency_factor = 0.8 + (age_days - 7) * 0.02
        
        # Apply decay resistance
        resistance_factor = 1.0 / decay_resistance
        
        # Calculate final decay score
        raw_decay = (base_decay + frequency_penalty) * recency_factor * resistance_factor
        protected_decay = max(0, raw_decay - emotional_protection)
        
        # Apply sigmoid transformation for smoother decay curve
        final_score = self._sigmoid_decay(protected_decay)
        
        return min(1.0, max(0.0, final_score))
    
    def _sigmoid_decay(self, raw_score: float) -> float:
        """Apply sigmoid transformation for natural decay curve"""
        # Sigmoid function: 1 / (1 + e^(-k(x-offset)))
        k = 0.5  # Steepness
        offset = 2.0  # Midpoint
        return 1 / (1 + math.exp(-k * (raw_score - offset)))
    
    def track_symbol_usage(self, symbol: str, context_tags: Optional[List[str]] = None, 
                          emotional_weight: float = 0.5):
        """Track symbol usage for decay calculation"""
        current_time = time.time()
        
        if symbol in self.symbols:
            # Update existing symbol
            metadata = self.symbols[symbol]
            metadata.last_used = current_time
            metadata.usage_frequency += 1
            metadata.emotional_weight = (metadata.emotional_weight + emotional_weight) / 2
            if context_tags:
                metadata.context_tags.extend(context_tags)
                metadata.context_tags = list(set(metadata.context_tags))  # Remove duplicates
        else:
            # Create new symbol
            self.symbols[symbol] = SymbolMetadata(
                symbol=symbol,
                last_used=current_time,
                usage_frequency=1,
                creation_time=current_time,
                emotional_weight=emotional_weight,
                context_tags=context_tags or [],
                decay_resistance=1.0
            )
        
        self.save_to_memory()
    
    def get_dormant_symbols(self, threshold: Optional[float] = None) -> List[Tuple[str, float]]:
        """Get list of symbols that have decayed past dormancy threshold"""
        threshold = threshold or self.decay_constants["dormancy_threshold"]
        
        dormant = []
        for symbol in self.symbols:
            decay_score = self.symbol_decay_score(symbol)
            if decay_score >= threshold:
                dormant.append((symbol, decay_score))
        
        # Sort by decay score (most dormant first)
        dormant.sort(key=lambda x: x[1], reverse=True)
        return dormant
    
    def get_resurrection_candidates(self, max_candidates: int = 5) -> List[Tuple[str, float]]:
        """Get symbols ready for resurrection (moderately dormant but still valuable)"""
        candidates = []
        
        for symbol, metadata in self.symbols.items():
            decay_score = self.symbol_decay_score(symbol)
            
            # Good resurrection candidates:
            # - Moderately dormant (0.4-0.8 decay)
            # - High emotional weight or frequency
            # - Not too old
            age_days = (time.time() - metadata.creation_time) / 86400
            
            if (0.4 <= decay_score <= 0.8 and 
                (metadata.emotional_weight > 0.6 or metadata.usage_frequency > 3) and
                age_days < 90):  # Not older than 3 months
                
                resurrection_score = self._calculate_resurrection_value(metadata, decay_score)
                candidates.append((symbol, resurrection_score))
        
        # Sort by resurrection value
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:max_candidates]
    
    def _calculate_resurrection_value(self, metadata: SymbolMetadata, decay_score: float) -> float:
        """Calculate how valuable a symbol would be to resurrect"""
        # Higher emotional weight = more valuable
        emotional_value = metadata.emotional_weight
        
        # Higher frequency = more valuable
        frequency_value = min(1.0, metadata.usage_frequency / 10.0)
        
        # Sweet spot decay (not too fresh, not too stale)
        decay_value = 1.0 - abs(decay_score - 0.6)  # Peak at 0.6 decay
        
        # Recency factor (not too old)
        age_days = (time.time() - metadata.last_used) / 86400
        recency_value = max(0.1, 1.0 - (age_days / 60))  # Linear decay over 60 days
        
        return (emotional_value * 0.4 + frequency_value * 0.3 + 
                decay_value * 0.2 + recency_value * 0.1)
    
    def boost_symbol_resistance(self, symbol: str, resistance_boost: float = 0.2):
        """Increase a symbol's resistance to decay"""
        if symbol in self.symbols:
            self.symbols[symbol].decay_resistance += resistance_boost
            self.symbols[symbol].decay_resistance = min(3.0, self.symbols[symbol].decay_resistance)
            self.save_to_memory()
    
    def get_symbol_stats(self) -> Dict[str, Any]:
        """Get statistics about symbol memory"""
        if not self.symbols:
            return {"total_symbols": 0}
        
        decay_scores = [self.symbol_decay_score(s) for s in self.symbols]
        
        return {
            "total_symbols": len(self.symbols),
            "dormant_symbols": len([s for s in decay_scores if s >= 0.7]),
            "active_symbols": len([s for s in decay_scores if s < 0.3]),
            "average_decay": sum(decay_scores) / len(decay_scores),
            "most_decayed": max(decay_scores) if decay_scores else 0,
            "freshest": min(decay_scores) if decay_scores else 0
        }
    
    def save_to_memory(self):
        """Save symbol memory to persistent storage"""
        try:
            import os
            os.makedirs(os.path.dirname(self.memory_path), exist_ok=True)
            
            data = {
                "symbols": {k: asdict(v) for k, v in self.symbols.items()},
                "decay_constants": self.decay_constants
            }
            
            with open(self.memory_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save symbol memory: {e}")
    
    def load_from_memory(self):
        """Load symbol memory from persistent storage"""
        try:
            with open(self.memory_path, 'r') as f:
                data = json.load(f)
            
            if "symbols" in data:
                for symbol_id, symbol_data in data["symbols"].items():
                    self.symbols[symbol_id] = SymbolMetadata(**symbol_data)
            
            if "decay_constants" in data:
                self.decay_constants.update(data["decay_constants"])
                
        except FileNotFoundError:
            pass
        except Exception as e:
            print(f"Warning: Could not load symbol memory: {e}")

# Convenience function for easy importing
def symbol_decay_score(symbol: str, last_used: float, usage_frequency: int) -> float:
    """
    Standalone function for symbol decay calculation.
    Determine how dormant a symbol is based on last-used and context frequency.
    """
    memory = SymbolMemory()
    return memory.symbol_decay_score(symbol, last_used, usage_frequency)

# Example usage
if __name__ == "__main__":
    memory = SymbolMemory()
    
    # Test symbol decay calculation
    current_time = time.time()
    
    test_cases = [
        ("moon", current_time - 86400, 5),        # Used yesterday, freq 5
        ("shadow", current_time - 604800, 2),     # Used week ago, freq 2  
        ("whisper", current_time - 2592000, 1),   # Used month ago, freq 1
        ("flame", current_time - 300, 10),        # Used 5 min ago, freq 10
    ]
    
    for symbol, last_used, frequency in test_cases:
        decay = memory.symbol_decay_score(symbol, last_used, frequency)
        age_days = (current_time - last_used) / 86400
        print(f"{symbol} (age: {age_days:.1f}d, freq: {frequency}): decay={decay:.3f}")
    
    # Test tracking and resurrection candidates
    memory.track_symbol_usage("starlight", ["romantic", "night"], 0.8)
    memory.track_symbol_usage("coffee", ["morning", "routine"], 0.3)
    
    print(f"\nSymbol stats: {memory.get_symbol_stats()}")
    print(f"Resurrection candidates: {memory.get_resurrection_candidates()}")

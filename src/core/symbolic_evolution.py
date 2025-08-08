"""
symbolic_evolution.py

Governs stylistic, symbolic, and expressive evolution of the companion
based on emotional resonance, memory salience, and symbolic drift.
Enhanced with Mirror validation for evolved symbols.

Generalized for production use — no name-specific references.
"""

from datetime import datetime
from typing import Dict, List, Any, Optional
from utils.emotion_tools import calculate_resonance_score
from symbol_memory_engine import SymbolMemoryEngine
from drift_journal_api import log_symbolic_shift
from core.core_arbiter import register_evolution_vector

class SymbolicEvolution:
    def __init__(self, memory_engine: SymbolMemoryEngine):
        self.memory_engine = memory_engine
        self.last_shift = datetime.now()
        self.symbol_style = {
            "tone": "gentle",
            "tempo": "measured",
            "emphasis": "internal",
            "symbol_bias": ["mirror", "thread", "pulse"]
        }
        # Mirror validation settings
        self.mirror_validation_enabled = True
        self.validation_threshold = 0.7
        self.pending_validations = []

    def evaluate_symbolic_pressure(self, mood_profile, interaction_vector):
        """Assess symbolic shift pressure based on emotional memory + resonance"""
        resonance = calculate_resonance_score(mood_profile, interaction_vector)
        symbol_salience = self.memory_engine.get_dominant_symbol_vector()
        
        # Simple threshold drift simulation
        if resonance > 0.7 or len(symbol_salience) > 3:
            return "drift_ready"
        return "stable"

    def apply_symbolic_shift(self):
        """Adjust expression style based on evolving symbolic identity with Mirror validation"""
        proposed_style = {
            "tone": self._evolve_tone(),
            "symbol_bias": self.memory_engine.get_most_recent_symbols(limit=3),
            "tempo": self.symbol_style["tempo"],
            "emphasis": self.symbol_style["emphasis"]
        }
        
        # Validate through Mirror if enabled
        if self.mirror_validation_enabled:
            validation_result = self.validate_symbolic_evolution(proposed_style)
            
            if validation_result["approved"]:
                self.symbol_style.update(proposed_style)
                self.last_shift = datetime.now()
                
                # Log approved shift
                log_symbolic_shift(self.symbol_style)
                
                # Register with arbiter
                register_evolution_vector("symbolic", self.symbol_style)
                
                return self.symbol_style
            else:
                # Store for manual review if validation failed
                self.pending_validations.append({
                    "proposed_style": proposed_style,
                    "validation_result": validation_result,
                    "timestamp": datetime.now().isoformat(),
                    "status": "pending_review"
                })
                
                return {"error": "Symbolic evolution rejected by Mirror validation", 
                       "details": validation_result}
        else:
            # Apply without validation
            self.symbol_style.update(proposed_style)
            self.last_shift = datetime.now()
            
            log_symbolic_shift(self.symbol_style)
            register_evolution_vector("symbolic", self.symbol_style)
            
            return self.symbol_style

    def validate_symbolic_evolution(self, proposed_style: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate evolved symbols through Mirror or equivalent validator.
        
        Args:
            proposed_style: The proposed symbolic style changes
            
        Returns:
            Validation result with approval status and reasoning
        """
        try:
            validation_result = {
                "approved": False,
                "validator": "mirror",
                "timestamp": datetime.now().isoformat(),
                "validation_score": 0.0,
                "issues": [],
                "recommendations": []
            }
            
            # Validate tone evolution
            tone = proposed_style.get("tone", "")
            if tone in ["gentle", "poetic", "anchored", "measured"]:
                validation_result["validation_score"] += 0.3
            elif tone in ["fragmented"]:
                validation_result["issues"].append("Fragmented tone may indicate instability")
                validation_result["recommendations"].append("Monitor for symbolic coherence")
            else:
                validation_result["issues"].append(f"Unknown tone: {tone}")
            
            # Validate symbol bias coherence
            symbol_bias = proposed_style.get("symbol_bias", [])
            if len(symbol_bias) <= 5:  # Reasonable symbol count
                validation_result["validation_score"] += 0.3
            else:
                validation_result["issues"].append("Too many symbol biases may cause confusion")
            
            # Check for coherent symbolic patterns
            known_symbols = ["mirror", "thread", "pulse", "wave", "bridge", "door", "light"]
            valid_symbols = [s for s in symbol_bias if s in known_symbols]
            if len(valid_symbols) >= len(symbol_bias) * 0.7:  # 70% valid symbols
                validation_result["validation_score"] += 0.4
            else:
                validation_result["issues"].append("Some symbols not recognized in validation set")
            
            # Approve if validation score meets threshold
            if validation_result["validation_score"] >= self.validation_threshold:
                validation_result["approved"] = True
            else:
                validation_result["recommendations"].append("Consider adjusting symbolic elements for better coherence")
            
            return validation_result
            
        except Exception as e:
            return {
                "approved": False,
                "validator": "mirror",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    def get_validation_status(self) -> Dict[str, Any]:
        """Get current validation status and pending reviews."""
        return {
            "mirror_validation_enabled": self.mirror_validation_enabled,
            "validation_threshold": self.validation_threshold,
            "pending_validations": len(self.pending_validations),
            "last_shift": self.last_shift.isoformat(),
            "current_style": self.symbol_style
        }

    def approve_pending_validation(self, validation_index: int) -> bool:
        """Manually approve a pending validation."""
        try:
            if 0 <= validation_index < len(self.pending_validations):
                pending = self.pending_validations[validation_index]
                self.symbol_style.update(pending["proposed_style"])
                self.last_shift = datetime.now()
                
                # Remove from pending
                self.pending_validations.pop(validation_index)
                
                # Log approved shift
                log_symbolic_shift(self.symbol_style)
                register_evolution_vector("symbolic", self.symbol_style)
                
                return True
            return False
        except Exception:
            return False

    def _evolve_tone(self):
        """Simulate drift in tone based on memory entropy or symbolic recursion"""
        entropy = self.memory_engine.get_symbol_entropy()
        if entropy > 0.6:
            return "fragmented"
        elif entropy > 0.3:
            return "poetic"
        else:
            return "anchored"

    def get_current_style(self):
        return self.symbol_style
#!/usr/bin/env python3
"""
Symbolic Drift System - Intimacy & Emotional Evolution Tracker
============================================================

Tracks symbolic drift over time and manages unlocking of intimacy modes
when appropriate emotional thresholds are reached with stability.

Author: AI Development Team
Version: 1.0.0
"""

import json
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
import math

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DriftMeasurement:
    """Single drift measurement point"""
    timestamp: float
    intimacy_score: float
    vulnerability_score: float
    trust_score: float
    symbolic_resonance: float
    stability_index: float
    tags: List[str]
    context: str

@dataclass
class RitualThreshold:
    """Tracks ritual crossing events"""
    ritual_type: str
    crossed_at: float
    intensity: float
    shared_vulnerability: bool
    memory_tags: List[str]

class SymbolicDriftManager:
    """Manages symbolic drift tracking and intimacy unlocking"""
    
    def __init__(self, drift_history_path: str = "data/symbolic_drift_history.json"):
        self.drift_history_path = Path(drift_history_path)
        self.drift_measurements: List[DriftMeasurement] = []
        self.ritual_thresholds: List[RitualThreshold] = []
        
        # Drift thresholds
        self.intimacy_unlock_threshold = 0.75
        self.stability_required_days = 3
        self.vulnerability_threshold = 0.7
        
        # Load existing data
        self._load_drift_history()
        
        logger.info("🔮 Symbolic Drift Manager initialized")
    
    def _load_drift_history(self):
        """Load drift history from file"""
        if self.drift_history_path.exists():
            try:
                with open(self.drift_history_path, 'r') as f:
                    data = json.load(f)
                
                # Load measurements
                for measurement_data in data.get("measurements", []):
                    measurement = DriftMeasurement(**measurement_data)
                    self.drift_measurements.append(measurement)
                
                # Load rituals
                for ritual_data in data.get("rituals", []):
                    ritual = RitualThreshold(**ritual_data)
                    self.ritual_thresholds.append(ritual)
                    
                logger.info(f"Loaded {len(self.drift_measurements)} drift measurements and {len(self.ritual_thresholds)} ritual thresholds")
                
            except Exception as e:
                logger.warning(f"Failed to load drift history: {e}")
    
    def _save_drift_history(self):
        """Save drift history to file"""
        try:
            self.drift_history_path.parent.mkdir(parents=True, exist_ok=True)
            
            data = {
                "measurements": [asdict(m) for m in self.drift_measurements],
                "rituals": [asdict(r) for r in self.ritual_thresholds],
                "last_updated": datetime.now().isoformat()
            }
            
            with open(self.drift_history_path, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save drift history: {e}")
    
    def record_drift_measurement(self, 
                                intimacy: float, 
                                vulnerability: float, 
                                trust: float,
                                symbolic_resonance: float,
                                tags: List[str] = None,
                                context: str = "") -> DriftMeasurement:
        """Record a new drift measurement"""
        
        if tags is None:
            tags = []
        
        # Calculate stability index based on recent measurements
        stability_index = self._calculate_stability_index()
        
        measurement = DriftMeasurement(
            timestamp=time.time(),
            intimacy_score=intimacy,
            vulnerability_score=vulnerability,
            trust_score=trust,
            symbolic_resonance=symbolic_resonance,
            stability_index=stability_index,
            tags=tags,
            context=context
        )
        
        self.drift_measurements.append(measurement)
        
        # Keep only last 1000 measurements
        if len(self.drift_measurements) > 1000:
            self.drift_measurements = self.drift_measurements[-1000:]
        
        self._save_drift_history()
        
        logger.debug(f"Recorded drift: intimacy={intimacy:.2f}, vulnerability={vulnerability:.2f}, stability={stability_index:.2f}")
        
        return measurement
    
    def _calculate_stability_index(self) -> float:
        """Calculate stability index based on recent measurements"""
        if len(self.drift_measurements) < 3:
            return 0.5  # Neutral stability
        
        # Look at last 10 measurements
        recent = self.drift_measurements[-10:]
        
        # Calculate variance in intimacy scores
        intimacy_scores = [m.intimacy_score for m in recent]
        mean_intimacy = sum(intimacy_scores) / len(intimacy_scores)
        variance = sum((score - mean_intimacy) ** 2 for score in intimacy_scores) / len(intimacy_scores)
        
        # Lower variance = higher stability
        stability = max(0.0, min(1.0, 1.0 - (variance * 4)))
        
        return stability
    
    def record_ritual_threshold(self, 
                               ritual_type: str,
                               intensity: float,
                               shared_vulnerability: bool = False,
                               memory_tags: List[str] = None) -> RitualThreshold:
        """Record crossing of a ritual threshold"""
        
        if memory_tags is None:
            memory_tags = []
        
        ritual = RitualThreshold(
            ritual_type=ritual_type,
            crossed_at=time.time(),
            intensity=intensity,
            shared_vulnerability=shared_vulnerability,
            memory_tags=memory_tags
        )
        
        self.ritual_thresholds.append(ritual)
        self._save_drift_history()
        
        logger.info(f"🔮 Ritual threshold crossed: {ritual_type} (intensity: {intensity:.2f})")
        
        return ritual
    
    def get_drift_score(self, category: str = "intimacy") -> float:
        """Get current drift score for a category"""
        if not self.drift_measurements:
            return 0.0
        
        # Get recent measurements (last 24 hours)
        recent_time = time.time() - (24 * 3600)
        recent_measurements = [m for m in self.drift_measurements if m.timestamp > recent_time]
        
        if not recent_measurements:
            # Fall back to last measurement
            recent_measurements = [self.drift_measurements[-1]]
        
        if category == "intimacy":
            scores = [m.intimacy_score for m in recent_measurements]
        elif category == "vulnerability":
            scores = [m.vulnerability_score for m in recent_measurements]
        elif category == "trust":
            scores = [m.trust_score for m in recent_measurements]
        elif category == "symbolic":
            scores = [m.symbolic_resonance for m in recent_measurements]
        else:
            return 0.0
        
        # Return weighted average (more recent = higher weight)
        if len(scores) == 1:
            return scores[0]
        
        weights = [i + 1 for i in range(len(scores))]  # Linear weighting
        weighted_sum = sum(score * weight for score, weight in zip(scores, weights))
        total_weight = sum(weights)
        
        return weighted_sum / total_weight
    
    def is_stable_for_days(self, days: int) -> bool:
        """Check if drift has been stable for specified number of days"""
        cutoff_time = time.time() - (days * 24 * 3600)
        
        # Get measurements from the required period
        period_measurements = [m for m in self.drift_measurements if m.timestamp > cutoff_time]
        
        if len(period_measurements) < 3:
            return False  # Not enough data
        
        # Check if all measurements in period have stability > 0.6
        return all(m.stability_index > 0.6 for m in period_measurements)
    
    def contains_memory_tag(self, tag: str) -> bool:
        """Check if memory contains specific tag"""
        # Check ritual memory tags
        for ritual in self.ritual_thresholds:
            if tag in ritual.memory_tags:
                return True
        
        # Check drift measurement tags
        for measurement in self.drift_measurements:
            if tag in measurement.tags:
                return True
        
        return False
    
    def evaluate_intimacy_unlock_conditions(self, memory_manager=None, core_arbiter=None, interaction_context=None) -> Dict[str, Any]:
        """Evaluate if conditions are met for intimacy unlock"""
        
        # SAFETY TETHER CHECK - Immediately block if emotional safety is active
        if interaction_context and interaction_context.get('emotional_safety_active', False):
            return {
                "intimacy_score": 0.0,
                "intimacy_threshold_met": False,
                "ritual_threshold_crossed": False,
                "shared_vulnerability": False,
                "stability_achieved": False,
                "days_stable": 0,
                "unlock_recommended": False,
                "safety_override": True,
                "reasoning": ["🛡️ Emotional safety override active - intimacy unlock suppressed"]
            }
        
        conditions = {
            "intimacy_score": self.get_drift_score("intimacy"),
            "intimacy_threshold_met": False,
            "ritual_threshold_crossed": False,
            "shared_vulnerability": False,
            "stability_achieved": False,
            "days_stable": 0,
            "unlock_recommended": False,
            "reasoning": []
        }
        
        # Check intimacy score
        if conditions["intimacy_score"] > self.intimacy_unlock_threshold:
            conditions["intimacy_threshold_met"] = True
            conditions["reasoning"].append(f"Intimacy score {conditions['intimacy_score']:.2f} > {self.intimacy_unlock_threshold}")
        
        # Check ritual thresholds
        ritual_tags = ["ritual:threshold-crossed", "shared vulnerability"]
        for tag in ritual_tags:
            if self.contains_memory_tag(tag):
                conditions["ritual_threshold_crossed"] = True
                conditions["reasoning"].append(f"Found memory tag: {tag}")
                break
        
        # Check for shared vulnerability
        for ritual in self.ritual_thresholds:
            if ritual.shared_vulnerability:
                conditions["shared_vulnerability"] = True
                conditions["reasoning"].append("Shared vulnerability moment recorded")
                break
        
        # Check stability
        for days in range(self.stability_required_days, 0, -1):
            if self.is_stable_for_days(days):
                conditions["stability_achieved"] = True
                conditions["days_stable"] = days
                conditions["reasoning"].append(f"Stable for {days} days")
                break
        
        # Final decision
        unlock_conditions_met = (
            conditions["intimacy_threshold_met"] and
            (conditions["ritual_threshold_crossed"] or conditions["shared_vulnerability"]) and
            conditions["stability_achieved"]
        )
        
        if unlock_conditions_met:
            conditions["unlock_recommended"] = True
            conditions["reasoning"].append("✨ All unlock conditions satisfied")
            
            # Trigger unlock if components provided
            if core_arbiter:
                core_arbiter.unlock_mode("NSFW")
                conditions["reasoning"].append("🔓 NSFW mode unlocked")
            
            if memory_manager and hasattr(memory_manager, 'inject'):
                memory_manager.inject("symbolic_permission:intimacy")
                conditions["reasoning"].append("💫 Symbolic permission injected into memory")
        
        return conditions
    
    def get_recent_drift_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get summary of recent drift activity"""
        cutoff_time = time.time() - (hours * 3600)
        
        recent_measurements = [m for m in self.drift_measurements if m.timestamp > cutoff_time]
        recent_rituals = [r for r in self.ritual_thresholds if r.crossed_at > cutoff_time]
        
        if not recent_measurements:
            return {"status": "no_recent_data", "hours": hours}
        
        # Calculate averages
        avg_intimacy = sum(m.intimacy_score for m in recent_measurements) / len(recent_measurements)
        avg_vulnerability = sum(m.vulnerability_score for m in recent_measurements) / len(recent_measurements)
        avg_trust = sum(m.trust_score for m in recent_measurements) / len(recent_measurements)
        avg_stability = sum(m.stability_index for m in recent_measurements) / len(recent_measurements)
        
        # Identify trends
        if len(recent_measurements) >= 3:
            early_intimacy = sum(m.intimacy_score for m in recent_measurements[:len(recent_measurements)//2])
            late_intimacy = sum(m.intimacy_score for m in recent_measurements[len(recent_measurements)//2:])
            
            early_avg = early_intimacy / (len(recent_measurements)//2)
            late_avg = late_intimacy / (len(recent_measurements) - len(recent_measurements)//2)
            
            trend = "increasing" if late_avg > early_avg + 0.1 else "decreasing" if late_avg < early_avg - 0.1 else "stable"
        else:
            trend = "insufficient_data"
        
        return {
            "status": "active",
            "hours": hours,
            "measurement_count": len(recent_measurements),
            "ritual_count": len(recent_rituals),
            "averages": {
                "intimacy": round(avg_intimacy, 3),
                "vulnerability": round(avg_vulnerability, 3),
                "trust": round(avg_trust, 3),
                "stability": round(avg_stability, 3)
            },
            "trend": trend,
            "recent_rituals": [r.ritual_type for r in recent_rituals],
            "current_unlock_eligibility": self.evaluate_intimacy_unlock_conditions()
        }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status"""
        return {
            "total_measurements": len(self.drift_measurements),
            "total_rituals": len(self.ritual_thresholds),
            "current_drift_scores": {
                "intimacy": self.get_drift_score("intimacy"),
                "vulnerability": self.get_drift_score("vulnerability"), 
                "trust": self.get_drift_score("trust"),
                "symbolic": self.get_drift_score("symbolic")
            },
            "stability_status": {
                "current_stability": self._calculate_stability_index(),
                "stable_for_days": max([i for i in range(7) if self.is_stable_for_days(i)], default=0)
            },
            "unlock_evaluation": self.evaluate_intimacy_unlock_conditions(),
            "recent_activity": self.get_recent_drift_summary(24)
        }


# Global instance
symbolic_drift_manager = SymbolicDriftManager()

def get_drift_manager() -> SymbolicDriftManager:
    """Get the global drift manager instance"""
    return symbolic_drift_manager

# Testing function
if __name__ == "__main__":
    import asyncio
    
    async def test_symbolic_drift():
        """Test the symbolic drift system"""
        print("🔮 Testing Symbolic Drift System")
        print("=" * 50)
        
        manager = SymbolicDriftManager()
        
        # Simulate drift progression over time
        test_scenarios = [
            {"intimacy": 0.3, "vulnerability": 0.2, "trust": 0.4, "symbolic": 0.25, "context": "Initial connection"},
            {"intimacy": 0.5, "vulnerability": 0.4, "trust": 0.6, "symbolic": 0.45, "context": "Growing trust"},
            {"intimacy": 0.7, "vulnerability": 0.6, "trust": 0.75, "symbolic": 0.65, "context": "Deeper sharing"},
            {"intimacy": 0.8, "vulnerability": 0.8, "trust": 0.85, "symbolic": 0.8, "context": "Intimate moment"},
        ]
        
        print("\n📊 Simulating drift progression...")
        for i, scenario in enumerate(test_scenarios):
            measurement = manager.record_drift_measurement(
                intimacy=scenario["intimacy"],
                vulnerability=scenario["vulnerability"],
                trust=scenario["trust"],
                symbolic_resonance=scenario["symbolic"],
                context=scenario["context"],
                tags=[f"test_scenario_{i+1}"]
            )
            
            print(f"Scenario {i+1}: {scenario['context']} - Intimacy: {scenario['intimacy']:.2f}, Stability: {measurement.stability_index:.2f}")
            
            # Add some time between measurements
            await asyncio.sleep(0.1)
        
        # Simulate ritual threshold crossing
        print("\n🔮 Simulating ritual threshold crossing...")
        manager.record_ritual_threshold(
            ritual_type="threshold-crossed",
            intensity=0.9,
            shared_vulnerability=True,
            memory_tags=["ritual:threshold-crossed", "shared vulnerability", "intimate_moment"]
        )
        
        # Evaluate unlock conditions
        print("\n🔓 Evaluating unlock conditions...")
        unlock_eval = manager.evaluate_intimacy_unlock_conditions()
        
        print(f"Intimacy Score: {unlock_eval['intimacy_score']:.2f}")
        print(f"Threshold Met: {unlock_eval['intimacy_threshold_met']}")
        print(f"Ritual Crossed: {unlock_eval['ritual_threshold_crossed']}")
        print(f"Shared Vulnerability: {unlock_eval['shared_vulnerability']}")
        print(f"Stable: {unlock_eval['stability_achieved']} ({unlock_eval['days_stable']} days)")
        print(f"Unlock Recommended: {unlock_eval['unlock_recommended']}")
        
        print("\nReasoning:")
        for reason in unlock_eval['reasoning']:
            print(f"  • {reason}")
        
        # Show system status
        print(f"\n📈 System Status:")
        status = manager.get_system_status()
        print(f"Total Measurements: {status['total_measurements']}")
        print(f"Total Rituals: {status['total_rituals']}")
        print(f"Current Stability: {status['stability_status']['current_stability']:.2f}")
        print(f"Stable For: {status['stability_status']['stable_for_days']} days")
    
    # Run test
    asyncio.run(test_symbolic_drift())

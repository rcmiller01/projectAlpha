#!/usr/bin/env python3
"""
Symbolic Drift System - Intimacy & Emotional Evolution Tracker
============================================================

Tracks symbolic drift over time and manages unlocking of intimacy modes
when appropriate emotional thresholds are reached with stability.

Enhanced with security features:
- Symbolic change monitoring with anomaly alerts
- Input validation for all drift measurements
- Comprehensive logging for symbolic changes
- Rate limiting and session management for symbolic access

Author: AI Development Team
Version: 1.1.0 (Security Enhanced)
"""

import json
import time
import logging
import hashlib
import re
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import deque, defaultdict
import math

# Enhanced logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
)
logger = logging.getLogger(__name__)

# Security configuration
SYMBOLIC_SESSION_TOKEN_LENGTH = 32
MAX_DRIFT_MEASUREMENTS = 1000
DRIFT_ANOMALY_THRESHOLD = 0.9
SYMBOLIC_RATE_LIMIT = 20  # measurements per hour
MAX_CONTEXT_LENGTH = 500
MAX_TAGS_PER_MEASUREMENT = 10

# Thread safety
drift_lock = threading.Lock()

# Session management
symbolic_sessions = {}
session_expiry_hours = 24

# Rate limiting
drift_requests = defaultdict(lambda: deque())

# Anomaly detection
symbolic_anomalies = deque(maxlen=50)

def validate_symbolic_session(session_token: str) -> bool:
    """Validate symbolic drift session token"""
    if not session_token or len(session_token) != SYMBOLIC_SESSION_TOKEN_LENGTH:
        return False
    
    if session_token not in symbolic_sessions:
        return False
    
    # Check if session has expired
    session_data = symbolic_sessions[session_token]
    if datetime.now() > session_data['expires_at']:
        del symbolic_sessions[session_token]
        return False
    
    # Update last access time
    session_data['last_access'] = datetime.now()
    return True

def generate_symbolic_session() -> str:
    """Generate a secure symbolic session token"""
    timestamp = str(time.time())
    random_data = str(hash(datetime.now()))
    token_string = f"symbolic:{timestamp}:{random_data}"
    return hashlib.sha256(token_string.encode()).hexdigest()[:SYMBOLIC_SESSION_TOKEN_LENGTH]

def check_drift_rate_limit(session_token: str) -> bool:
    """Check if drift measurement rate limit is exceeded"""
    current_time = time.time()
    
    # Clean old requests
    while (drift_requests[session_token] and 
           drift_requests[session_token][0] < current_time - 3600):  # 1 hour window
        drift_requests[session_token].popleft()
    
    # Check limit
    if len(drift_requests[session_token]) >= SYMBOLIC_RATE_LIMIT:
        logger.warning(f"Symbolic drift rate limit exceeded for session: {session_token[:8]}...")
        return False
    
    # Add current request
    drift_requests[session_token].append(current_time)
    return True

def validate_drift_measurement(measurement_data: Dict[str, Any]) -> tuple[bool, str]:
    """Validate drift measurement data"""
    try:
        # Check required fields
        required_fields = ['intimacy_score', 'vulnerability_score', 'trust_score']
        for field in required_fields:
            if field not in measurement_data:
                return False, f"Missing required field: {field}"
        
        # Validate score ranges (0-1)
        score_fields = ['intimacy_score', 'vulnerability_score', 'trust_score', 
                       'symbolic_resonance', 'stability_index']
        
        for field in score_fields:
            if field in measurement_data:
                score = measurement_data[field]
                if not isinstance(score, (int, float)):
                    return False, f"{field} must be a number"
                
                if score < 0 or score > 1:
                    return False, f"{field} must be between 0 and 1"
        
        # Validate tags
        if 'tags' in measurement_data:
            tags = measurement_data['tags']
            if not isinstance(tags, list):
                return False, "Tags must be a list"
            
            if len(tags) > MAX_TAGS_PER_MEASUREMENT:
                return False, f"Too many tags (max {MAX_TAGS_PER_MEASUREMENT})"
            
            # Validate tag content
            for tag in tags:
                if not isinstance(tag, str):
                    return False, "All tags must be strings"
                
                if not re.match(r'^[a-zA-Z0-9_-]+$', tag):
                    return False, f"Invalid tag format: {tag}"
        
        # Validate context
        if 'context' in measurement_data:
            context = measurement_data['context']
            if not isinstance(context, str):
                return False, "Context must be a string"
            
            if len(context) > MAX_CONTEXT_LENGTH:
                return False, f"Context exceeds maximum length of {MAX_CONTEXT_LENGTH}"
        
        return True, "Valid"
    
    except Exception as e:
        logger.error(f"Error validating drift measurement: {str(e)}")
        return False, f"Validation error: {str(e)}"

def detect_symbolic_anomaly(measurement: 'DriftMeasurement') -> bool:
    """Detect if a drift measurement represents an anomaly"""
    try:
        # Check for extreme values
        extreme_threshold = 0.95
        if (measurement.intimacy_score > extreme_threshold or
            measurement.vulnerability_score > extreme_threshold or
            measurement.trust_score > extreme_threshold):
            
            logger.warning(f"Extreme symbolic drift detected: intimacy={measurement.intimacy_score}")
            symbolic_anomalies.append({
                'timestamp': datetime.now().isoformat(),
                'type': 'extreme_values',
                'intimacy_score': measurement.intimacy_score,
                'vulnerability_score': measurement.vulnerability_score,
                'trust_score': measurement.trust_score
            })
            return True
        
        # Check for rapid changes
        if hasattr(measurement, 'previous_measurement') and measurement.previous_measurement:
            prev = measurement.previous_measurement
            intimacy_change = abs(measurement.intimacy_score - prev.intimacy_score)
            
            if intimacy_change > 0.5:  # 50% change
                logger.warning(f"Rapid symbolic change detected: {intimacy_change}")
                symbolic_anomalies.append({
                    'timestamp': datetime.now().isoformat(),
                    'type': 'rapid_change',
                    'change_amount': intimacy_change,
                    'context': measurement.context[:100] if measurement.context else ""
                })
                return True
        
        return False
    
    except Exception as e:
        logger.error(f"Error detecting symbolic anomaly: {str(e)}")
        return False

def sanitize_symbolic_input(text: str, max_length: int) -> str:
    """Sanitize symbolic text input"""
    if not isinstance(text, str):
        return ""
    
    # Remove potential injection patterns
    text = re.sub(r'[<>"\']', '', text)
    
    # Limit length
    if len(text) > max_length:
        text = text[:max_length] + "..."
    
    return text.strip()

def log_symbolic_activity(activity_type: str, session_token: str, details: Dict[str, Any], status: str = "success"):
    """Log symbolic drift activities"""
    try:
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'activity_type': activity_type,
            'session': session_token[:8] + "..." if session_token else "none",
            'details': details,
            'status': status,
            'thread_id': threading.get_ident()
        }
        
        logger.info(f"Symbolic activity logged: {activity_type} ({status})")
        
        if status != "success":
            logger.warning(f"Symbolic activity issue: {activity_type} failed with {status}")
        
    except Exception as e:
        logger.error(f"Error logging symbolic activity: {str(e)}")

@dataclass
class DriftMeasurement:
    """Single drift measurement point with security enhancements"""
    timestamp: float
    intimacy_score: float
    vulnerability_score: float
    trust_score: float
    symbolic_resonance: float
    stability_index: float
    tags: List[str]
    context: str
    session_token: Optional[str] = None
    anomaly_detected: bool = False
    sanitized: bool = False
    
    def __post_init__(self):
        """Validate and sanitize measurement after initialization"""
        # Clamp scores to valid range
        self.intimacy_score = max(0.0, min(1.0, float(self.intimacy_score)))
        self.vulnerability_score = max(0.0, min(1.0, float(self.vulnerability_score)))
        self.trust_score = max(0.0, min(1.0, float(self.trust_score)))
        self.symbolic_resonance = max(0.0, min(1.0, float(self.symbolic_resonance)))
        self.stability_index = max(0.0, min(1.0, float(self.stability_index)))
        
        # Sanitize tags
        if self.tags:
            self.tags = [sanitize_symbolic_input(tag, 50) for tag in self.tags[:MAX_TAGS_PER_MEASUREMENT]]
        
        # Sanitize context
        if self.context:
            self.context = sanitize_symbolic_input(self.context, MAX_CONTEXT_LENGTH)
            self.sanitized = True

@dataclass  
class RitualThreshold:
    """Tracks ritual crossing events with validation"""
    ritual_type: str
    crossed_at: float
    intensity: float
    shared_vulnerability: bool
    memory_tags: List[str]
    session_token: Optional[str] = None
    
    def __post_init__(self):
        """Validate ritual threshold after initialization"""
        self.intensity = max(0.0, min(1.0, float(self.intensity)))
        
        # Sanitize ritual type
        if self.ritual_type:
            self.ritual_type = sanitize_symbolic_input(self.ritual_type, 100)
        
        # Sanitize memory tags
        if self.memory_tags:
            self.memory_tags = [sanitize_symbolic_input(tag, 50) for tag in self.memory_tags[:MAX_TAGS_PER_MEASUREMENT]]

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

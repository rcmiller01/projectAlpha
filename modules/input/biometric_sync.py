# biometric_sync.py
# Phase 3: Advanced biometric synchronization for physiological response integration

import json
import time
import threading
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import numpy as np

class BiometricType(Enum):
    HEART_RATE = "heart_rate"
    HEART_RATE_VARIABILITY = "hrv"
    BREATHING_RATE = "breathing_rate"
    SKIN_CONDUCTANCE = "skin_conductance"
    BLOOD_PRESSURE = "blood_pressure"
    TEMPERATURE = "temperature"
    MOVEMENT = "movement"
    POSTURE = "posture"

@dataclass
class BiometricReading:
    type: BiometricType
    value: float
    timestamp: datetime
    confidence: float = 1.0
    context: str = "general"

@dataclass
class BiometricState:
    heart_rate: float = 0.0
    hrv: float = 0.0
    breathing_rate: float = 0.0
    skin_conductance: float = 0.0
    blood_pressure_systolic: float = 0.0
    blood_pressure_diastolic: float = 0.0
    temperature: float = 0.0
    movement_level: float = 0.0
    posture_confidence: float = 0.0
    last_update: datetime = datetime.now()

class BiometricSync:
    def __init__(self):
        self.current_state = BiometricState()
        self.reading_history = []
        self.emotional_correlations = {}
        self.device_connections = {}
        self.is_monitoring = False
        self.monitoring_thread = None
        
        # Emotional state thresholds
        self.emotional_thresholds = {
            "excitement": {"hr_min": 80, "hr_max": 120, "hrv_max": 30},
            "calm": {"hr_min": 60, "hr_max": 80, "hrv_min": 40},
            "stress": {"hr_min": 90, "hr_max": 150, "hrv_max": 20},
            "romance": {"hr_min": 70, "hr_max": 100, "hrv_min": 30},
            "intimacy": {"hr_min": 75, "hr_max": 110, "hrv_min": 25}
        }
        
        # Breathing pattern analysis
        self.breathing_patterns = {
            "normal": {"rate_min": 12, "rate_max": 20, "rhythm": "regular"},
            "deep": {"rate_min": 6, "rate_max": 12, "rhythm": "slow"},
            "rapid": {"rate_min": 20, "rate_max": 30, "rhythm": "fast"},
            "romantic": {"rate_min": 14, "rate_max": 18, "rhythm": "steady"}
        }
    
    def start_monitoring(self):
        """Start continuous biometric monitoring"""
        if self.is_monitoring:
            return False
            
        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self.monitoring_thread.start()
        
        print("[Biometric] Started continuous monitoring")
        return True
    
    def stop_monitoring(self):
        """Stop biometric monitoring"""
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=1.0)
        print("[Biometric] Stopped monitoring")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.is_monitoring:
            try:
                # Simulate biometric readings (replace with actual device data)
                self._simulate_readings()
                
                # Analyze current state
                emotional_state = self._analyze_emotional_state()
                
                # Trigger responses based on biometrics
                self._trigger_biometric_responses(emotional_state)
                
                time.sleep(1.0)  # 1 second intervals
                
            except Exception as e:
                print(f"[Biometric] Monitoring error: {e}")
                time.sleep(5.0)
    
    def _simulate_readings(self):
        """Simulate biometric readings (replace with actual device integration)"""
        # Simulate heart rate variations
        base_hr = 70 + np.random.normal(0, 5)
        self.current_state.heart_rate = max(50, min(150, base_hr))
        
        # Simulate HRV
        self.current_state.hrv = 30 + np.random.normal(0, 10)
        
        # Simulate breathing rate
        base_br = 15 + np.random.normal(0, 2)
        self.current_state.breathing_rate = max(8, min(25, base_br))
        
        # Update timestamp
        self.current_state.last_update = datetime.now()
        
        # Store reading
        reading = BiometricReading(
            type=BiometricType.HEART_RATE,
            value=self.current_state.heart_rate,
            timestamp=datetime.now()
        )
        self.reading_history.append(reading)
        
        # Keep only recent history
        cutoff_time = datetime.now() - timedelta(minutes=10)
        self.reading_history = [
            r for r in self.reading_history 
            if r.timestamp > cutoff_time
        ]
    
    def _analyze_emotional_state(self) -> Dict[str, float]:
        """Analyze current biometrics to determine emotional state"""
        hr = self.current_state.heart_rate
        hrv = self.current_state.hrv
        br = self.current_state.breathing_rate
        
        emotional_scores = {}
        
        # Analyze each emotional state
        for emotion, thresholds in self.emotional_thresholds.items():
            score = 0.0
            
            # Heart rate analysis
            if thresholds["hr_min"] <= hr <= thresholds["hr_max"]:
                score += 0.4
            
            # HRV analysis
            if "hrv_min" in thresholds and hrv >= thresholds["hrv_min"]:
                score += 0.3
            elif "hrv_max" in thresholds and hrv <= thresholds["hrv_max"]:
                score += 0.3
            
            # Breathing rate analysis
            if br >= 12 and br <= 20:  # Normal breathing
                score += 0.3
            
            emotional_scores[emotion] = min(1.0, score)
        
        return emotional_scores
    
    def _trigger_biometric_responses(self, emotional_state: Dict[str, float]):
        """Trigger responses based on biometric analysis"""
        # Find dominant emotion
        dominant_emotion = max(emotional_state.items(), key=lambda x: x[1])
        
        if dominant_emotion[1] > 0.6:  # Significant emotional state
            self._handle_emotional_trigger(dominant_emotion[0], dominant_emotion[1])
    
    def _handle_emotional_trigger(self, emotion: str, intensity: float):
        """Handle emotional triggers from biometrics"""
        from modules.input.haptic_system import get_haptic_system
        from modules.emotion.mood_engine import update_mood
        from modules.memory.emotional_memory import store_emotional_memory
        
        # Update mood engine
        update_mood(f"biometric:{emotion}", intensity=intensity)
        
        # Trigger haptic feedback
        haptic_system = get_haptic_system()
        haptic_system.trigger_emotional_haptic(emotion)
        
        # Store biometric memory
        store_emotional_memory({
            "timestamp": datetime.now().isoformat(),
            "trigger": f"biometric:{emotion}",
            "intensity": intensity,
            "biometrics": {
                "heart_rate": self.current_state.heart_rate,
                "hrv": self.current_state.hrv,
                "breathing_rate": self.current_state.breathing_rate
            },
            "thought": f"Biometric analysis detected {emotion} state (intensity: {intensity:.2f})"
        })
        
        print(f"[Biometric] Triggered {emotion} response (intensity: {intensity:.2f})")
    
    def update_biometric_reading(self, reading: BiometricReading):
        """Update biometric reading from external device"""
        # Update current state
        if reading.type == BiometricType.HEART_RATE:
            self.current_state.heart_rate = reading.value
        elif reading.type == BiometricType.HEART_RATE_VARIABILITY:
            self.current_state.hrv = reading.value
        elif reading.type == BiometricType.BREATHING_RATE:
            self.current_state.breathing_rate = reading.value
        elif reading.type == BiometricType.SKIN_CONDUCTANCE:
            self.current_state.skin_conductance = reading.value
        elif reading.type == BiometricType.BLOOD_PRESSURE:
            # Assuming value is systolic/diastolic tuple
            if isinstance(reading.value, (list, tuple)) and len(reading.value) == 2:
                self.current_state.blood_pressure_systolic = reading.value[0]
                self.current_state.blood_pressure_diastolic = reading.value[1]
        elif reading.type == BiometricType.TEMPERATURE:
            self.current_state.temperature = reading.value
        elif reading.type == BiometricType.MOVEMENT:
            self.current_state.movement_level = reading.value
        elif reading.type == BiometricType.POSTURE:
            self.current_state.posture_confidence = reading.value
        
        self.current_state.last_update = reading.timestamp
        self.reading_history.append(reading)
        
        # Analyze immediately
        emotional_state = self._analyze_emotional_state()
        self._trigger_biometric_responses(emotional_state)
    
    def get_romantic_sync_status(self) -> Dict:
        """Get romantic synchronization status"""
        hr = self.current_state.heart_rate
        hrv = self.current_state.hrv
        br = self.current_state.breathing_rate
        
        # Calculate romantic sync score
        romantic_score = 0.0
        
        # Heart rate in romantic range
        if 70 <= hr <= 100:
            romantic_score += 0.4
        
        # Good HRV for intimacy
        if hrv >= 25:
            romantic_score += 0.3
        
        # Steady breathing
        if 14 <= br <= 18:
            romantic_score += 0.3
        
        return {
            "romantic_sync_score": romantic_score,
            "heart_rate": hr,
            "hrv": hrv,
            "breathing_rate": br,
            "sync_quality": "excellent" if romantic_score > 0.8 else "good" if romantic_score > 0.6 else "fair",
            "recommendations": self._get_romantic_recommendations(romantic_score)
        }
    
    def _get_romantic_recommendations(self, sync_score: float) -> List[str]:
        """Get recommendations for improving romantic synchronization"""
        recommendations = []
        
        if sync_score < 0.6:
            recommendations.extend([
                "Try deep breathing exercises together",
                "Focus on slow, steady heart rate",
                "Create a calm, intimate environment"
            ])
        elif sync_score < 0.8:
            recommendations.extend([
                "Maintain current breathing rhythm",
                "Continue with gentle physical contact",
                "Share intimate thoughts and feelings"
            ])
        else:
            recommendations.extend([
                "Perfect synchronization achieved",
                "Enjoy this intimate moment",
                "Consider deepening the connection"
            ])
        
        return recommendations
    
    def get_biometric_summary(self) -> Dict:
        """Get comprehensive biometric summary"""
        return {
            "current_state": {
                "heart_rate": self.current_state.heart_rate,
                "hrv": self.current_state.hrv,
                "breathing_rate": self.current_state.breathing_rate,
                "skin_conductance": self.current_state.skin_conductance,
                "temperature": self.current_state.temperature,
                "movement_level": self.current_state.movement_level
            },
            "emotional_analysis": self._analyze_emotional_state(),
            "romantic_sync": self.get_romantic_sync_status(),
            "monitoring_active": self.is_monitoring,
            "last_update": self.current_state.last_update.isoformat(),
            "reading_count": len(self.reading_history)
        }

# Global biometric sync instance
biometric_sync = BiometricSync()

def get_biometric_sync() -> BiometricSync:
    """Get the global biometric sync instance"""
    return biometric_sync

def start_biometric_monitoring():
    """Start biometric monitoring"""
    return biometric_sync.start_monitoring()

def stop_biometric_monitoring():
    """Stop biometric monitoring"""
    biometric_sync.stop_monitoring()

def update_biometric_reading(type_name: str, value: float, context: str = "general"):
    """Update biometric reading from external source"""
    try:
        biometric_type = BiometricType(type_name)
        reading = BiometricReading(
            type=biometric_type,
            value=value,
            timestamp=datetime.now(),
            context=context
        )
        biometric_sync.update_biometric_reading(reading)
        return True
    except Exception as e:
        print(f"[Biometric] Error updating reading: {e}")
        return False 
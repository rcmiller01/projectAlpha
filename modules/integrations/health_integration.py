"""
Apple HealthKit Integration for AI Companion

This module provides comprehensive health data integration including:
- Real-time biometrics from Apple Watch/iPhone
- Historical health data from HealthKit
- Health trend analysis for emotional context
- Privacy-compliant health data access
"""

import json
import asyncio
import logging
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
import statistics

logger = logging.getLogger(__name__)

@dataclass
class BiometricReading:
    """Individual biometric measurement"""
    metric_type: str  # heart_rate, blood_pressure, temperature, etc.
    value: Union[float, Dict[str, float]]  # Single value or dict for complex metrics
    unit: str
    timestamp: datetime
    source: str  # apple_watch, iphone, manual, etc.
    confidence: float = 1.0  # 0.0 to 1.0

@dataclass
class HealthTrend:
    """Health trend analysis over time"""
    metric_type: str
    period: str  # daily, weekly, monthly
    trend_direction: str  # increasing, decreasing, stable
    percentage_change: float
    significance: str  # low, moderate, high
    data_points: int
    last_updated: datetime

@dataclass
class HealthContext:
    """Current health context for emotional awareness"""
    stress_level: Optional[float] = None  # 0.0 to 1.0
    energy_level: Optional[float] = None  # 0.0 to 1.0
    sleep_quality: Optional[float] = None  # 0.0 to 1.0
    activity_level: Optional[float] = None  # 0.0 to 1.0
    heart_rate_variability: Optional[float] = None
    current_activity: Optional[str] = None  # walking, running, resting, etc.
    health_alerts: Optional[List[str]] = None
    
    def __post_init__(self):
        if self.health_alerts is None:
            self.health_alerts = []

class HealthKitConnector:
    """Core HealthKit data connector"""
    
    def __init__(self, config_path: str = "data/health_config.json"):
        self.config_path = Path(config_path)
        self.config = {}
        self.authorized_metrics = set()
        self.last_sync = None
        self.cache = {}
        self.load_config()
        
    def load_config(self):
        """Load health integration configuration"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    self.config = json.load(f)
                self.authorized_metrics = set(self.config.get('authorized_metrics', []))
                logger.info(f"Loaded health config with {len(self.authorized_metrics)} authorized metrics")
            else:
                self.config = self._get_default_config()
                self.save_config()
        except Exception as e:
            logger.error(f"Error loading health config: {e}")
            self.config = self._get_default_config()
    
    def save_config(self):
        """Save health configuration"""
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving health config: {e}")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default health configuration"""
        return {
            "authorized_metrics": [
                "heart_rate",
                "heart_rate_variability",
                "blood_pressure",
                "body_temperature",
                "respiratory_rate",
                "blood_oxygen",
                "steps",
                "distance_walked",
                "calories_burned",
                "sleep_analysis",
                "stress_level",
                "mindfulness_minutes",
                "stand_hours",
                "exercise_minutes",
                "environmental_audio_exposure"
            ],
            "sync_interval": 300,  # 5 minutes
            "privacy_mode": "anonymized",
            "data_retention_days": 90,
            "real_time_monitoring": True,
            "health_insights": True,
            "trend_analysis": True,
            "alert_thresholds": {
                "heart_rate_max": 180,
                "heart_rate_min": 50,
                "stress_level_high": 0.8,
                "sleep_hours_min": 6
            }
        }
    
    async def request_authorization(self, metrics: List[str]) -> bool:
        """Request authorization for specific health metrics"""
        try:
            # In a real implementation, this would use HealthKit APIs
            # For now, we'll simulate the authorization process
            
            logger.info(f"Requesting authorization for metrics: {metrics}")
            
            # Simulate authorization success
            self.authorized_metrics.update(metrics)
            self.config['authorized_metrics'] = list(self.authorized_metrics)
            self.save_config()
            
            logger.info(f"Authorization granted for {len(metrics)} metrics")
            return True
            
        except Exception as e:
            logger.error(f"Authorization request failed: {e}")
            return False
    
    async def get_real_time_biometrics(self) -> List[BiometricReading]:
        """Get current real-time biometric readings"""
        readings = []
        now = datetime.now()
        
        try:
            # Simulate real-time data from Apple Watch/iPhone
            if "heart_rate" in self.authorized_metrics:
                readings.append(BiometricReading(
                    metric_type="heart_rate",
                    value=72.0,  # Simulated value
                    unit="bpm",
                    timestamp=now,
                    source="apple_watch",
                    confidence=0.95
                ))
            
            if "heart_rate_variability" in self.authorized_metrics:
                readings.append(BiometricReading(
                    metric_type="heart_rate_variability",
                    value=45.2,  # Simulated HRV in ms
                    unit="ms",
                    timestamp=now,
                    source="apple_watch",
                    confidence=0.90
                ))
            
            if "blood_oxygen" in self.authorized_metrics:
                readings.append(BiometricReading(
                    metric_type="blood_oxygen",
                    value=98.5,  # Simulated SpO2
                    unit="percent",
                    timestamp=now,
                    source="apple_watch",
                    confidence=0.88
                ))
            
            if "stress_level" in self.authorized_metrics:
                readings.append(BiometricReading(
                    metric_type="stress_level",
                    value=0.3,  # Simulated stress (0.0-1.0)
                    unit="normalized",
                    timestamp=now,
                    source="apple_watch",
                    confidence=0.75
                ))
            
            logger.debug(f"Retrieved {len(readings)} real-time biometric readings")
            return readings
            
        except Exception as e:
            logger.error(f"Error getting real-time biometrics: {e}")
            return []
    
    async def get_historical_data(self, metric_type: str, days: int = 7) -> List[BiometricReading]:
        """Get historical health data for analysis"""
        if metric_type not in self.authorized_metrics:
            logger.warning(f"Not authorized for metric: {metric_type}")
            return []
        
        try:
            # Simulate historical data retrieval
            readings = []
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # Generate simulated historical data
            current_date = start_date
            while current_date <= end_date:
                if metric_type == "heart_rate":
                    value = 70 + (current_date.hour - 12) * 2  # Vary by time of day
                elif metric_type == "steps":
                    value = 8000 + (current_date.weekday() * 1000)  # More steps on weekends
                elif metric_type == "sleep_analysis":
                    value = 7.5 + (current_date.weekday() % 2) * 0.5  # Sleep variation
                else:
                    value = 50.0  # Default value
                
                readings.append(BiometricReading(
                    metric_type=metric_type,
                    value=value,
                    unit=self._get_metric_unit(metric_type),
                    timestamp=current_date,
                    source="healthkit",
                    confidence=0.85
                ))
                
                current_date += timedelta(hours=1)
            
            logger.info(f"Retrieved {len(readings)} historical {metric_type} readings")
            return readings
            
        except Exception as e:
            logger.error(f"Error getting historical data for {metric_type}: {e}")
            return []
    
    def _get_metric_unit(self, metric_type: str) -> str:
        """Get the standard unit for a metric type"""
        units = {
            "heart_rate": "bpm",
            "heart_rate_variability": "ms",
            "blood_pressure": "mmHg",
            "body_temperature": "°F",
            "respiratory_rate": "breaths/min",
            "blood_oxygen": "percent",
            "steps": "count",
            "distance_walked": "miles",
            "calories_burned": "kcal",
            "sleep_analysis": "hours",
            "stress_level": "normalized",
            "mindfulness_minutes": "minutes",
            "stand_hours": "hours",
            "exercise_minutes": "minutes"
        }
        return units.get(metric_type, "unit")

class HealthAnalyzer:
    """Analyzes health data for trends and insights"""
    
    def __init__(self, connector: HealthKitConnector):
        self.connector = connector
        
    async def analyze_health_trends(self, days: int = 30) -> List[HealthTrend]:
        """Analyze health trends over specified period"""
        trends = []
        
        try:
            key_metrics = ["heart_rate", "steps", "sleep_analysis", "stress_level"]
            
            for metric in key_metrics:
                if metric in self.connector.authorized_metrics:
                    readings = await self.connector.get_historical_data(metric, days)
                    if len(readings) >= 7:  # Need minimum data for trend analysis
                        trend = self._calculate_trend(readings, metric)
                        if trend:
                            trends.append(trend)
            
            logger.info(f"Analyzed {len(trends)} health trends")
            return trends
            
        except Exception as e:
            logger.error(f"Error analyzing health trends: {e}")
            return []
    
    def _calculate_trend(self, readings: List[BiometricReading], metric_type: str) -> Optional[HealthTrend]:
        """Calculate trend for a specific metric"""
        try:
            if len(readings) < 7:
                return None
            
            # Split data into first and second half for comparison
            mid_point = len(readings) // 2
            first_half = [r.value for r in readings[:mid_point] if isinstance(r.value, (int, float))]
            second_half = [r.value for r in readings[mid_point:] if isinstance(r.value, (int, float))]
            
            if not first_half or not second_half:
                return None
            
            first_avg = statistics.mean(first_half)
            second_avg = statistics.mean(second_half)
            
            percentage_change = ((second_avg - first_avg) / first_avg) * 100
            
            # Determine trend direction
            if abs(percentage_change) < 5:
                direction = "stable"
                significance = "low"
            elif percentage_change > 0:
                direction = "increasing"
                significance = "moderate" if percentage_change < 15 else "high"
            else:
                direction = "decreasing"
                significance = "moderate" if percentage_change > -15 else "high"
            
            return HealthTrend(
                metric_type=metric_type,
                period="weekly",
                trend_direction=direction,
                percentage_change=percentage_change,
                significance=significance,
                data_points=len(readings),
                last_updated=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error calculating trend for {metric_type}: {e}")
            return None
    
    async def get_current_health_context(self) -> HealthContext:
        """Get current health context for emotional awareness"""
        try:
            readings = await self.connector.get_real_time_biometrics()
            
            context = HealthContext()
            
            for reading in readings:
                if reading.metric_type == "stress_level" and isinstance(reading.value, (int, float)):
                    context.stress_level = float(reading.value)
                elif reading.metric_type == "heart_rate_variability" and isinstance(reading.value, (int, float)):
                    # Higher HRV generally indicates better stress management
                    context.energy_level = min(1.0, float(reading.value) / 50.0)
            
            # Get recent sleep data
            sleep_data = await self.connector.get_historical_data("sleep_analysis", 1)
            if sleep_data and isinstance(sleep_data[-1].value, (int, float)):
                latest_sleep = sleep_data[-1]
                context.sleep_quality = min(1.0, float(latest_sleep.value) / 8.0)  # Normalize to 8 hours
            
            # Get recent activity data
            steps_data = await self.connector.get_historical_data("steps", 1)
            if steps_data and isinstance(steps_data[-1].value, (int, float)):
                latest_steps = steps_data[-1]
                context.activity_level = min(1.0, float(latest_steps.value) / 10000.0)  # Normalize to 10k steps
            
            # Check for health alerts
            await self._check_health_alerts(context, readings)
            
            logger.debug("Generated current health context")
            return context
            
        except Exception as e:
            logger.error(f"Error getting health context: {e}")
            return HealthContext()
    
    async def _check_health_alerts(self, context: HealthContext, readings: List[BiometricReading]):
        """Check for health alerts based on thresholds"""
        if context.health_alerts is None:
            context.health_alerts = []
            
        thresholds = self.connector.config.get('alert_thresholds', {})
        
        for reading in readings:
            if reading.metric_type == "heart_rate" and isinstance(reading.value, (int, float)):
                if reading.value > thresholds.get('heart_rate_max', 180):
                    context.health_alerts.append(f"High heart rate detected: {reading.value} bpm")
                elif reading.value < thresholds.get('heart_rate_min', 50):
                    context.health_alerts.append(f"Low heart rate detected: {reading.value} bpm")
            
            elif reading.metric_type == "stress_level" and isinstance(reading.value, (int, float)):
                if reading.value > thresholds.get('stress_level_high', 0.8):
                    context.health_alerts.append("High stress level detected")

class EmotionalHealthIntegration:
    """Integrates health data with emotional state management"""
    
    def __init__(self):
        self.connector = HealthKitConnector()
        self.analyzer = HealthAnalyzer(self.connector)
        
    async def initialize(self) -> bool:
        """Initialize health integration with necessary permissions"""
        try:
            # Request core health permissions
            core_metrics = [
                "heart_rate", "heart_rate_variability", "stress_level",
                "sleep_analysis", "steps", "blood_oxygen"
            ]
            
            success = await self.connector.request_authorization(core_metrics)
            if success:
                logger.info("Health integration initialized successfully")
            return success
            
        except Exception as e:
            logger.error(f"Failed to initialize health integration: {e}")
            return False
    
    async def get_emotional_health_context(self) -> Dict[str, Any]:
        """Get comprehensive health context for emotional awareness"""
        try:
            # Get current health state
            health_context = await self.analyzer.get_current_health_context()
            
            # Get recent trends
            trends = await self.analyzer.analyze_health_trends(7)
            
            # Get real-time biometrics
            current_readings = await self.connector.get_real_time_biometrics()
            
            # Generate emotional recommendations based on health data
            emotional_recommendations = self._generate_emotional_recommendations(health_context, trends)
            
            return {
                "health_context": asdict(health_context),
                "current_readings": [asdict(r) for r in current_readings],
                "health_trends": [asdict(t) for t in trends],
                "emotional_recommendations": emotional_recommendations,
                "last_updated": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting emotional health context: {e}")
            return {}
    
    def _generate_emotional_recommendations(self, context: HealthContext, trends: List[HealthTrend]) -> List[str]:
        """Generate emotional recommendations based on health data"""
        recommendations = []
        
        try:
            # Stress-based recommendations
            if context.stress_level and context.stress_level > 0.7:
                recommendations.append("Consider taking a moment for deep breathing or mindfulness")
                recommendations.append("Your stress levels are elevated - perhaps we should talk about what's on your mind")
            
            # Sleep-based recommendations
            if context.sleep_quality and context.sleep_quality < 0.6:
                recommendations.append("It looks like you might need some rest - would you like to discuss your sleep schedule?")
            
            # Activity-based recommendations
            if context.activity_level and context.activity_level < 0.3:
                recommendations.append("You've been less active today - would a gentle walk or movement help?")
            
            # Energy-based recommendations
            if context.energy_level and context.energy_level < 0.4:
                recommendations.append("Your energy seems low - let's focus on gentle, restorative conversation")
            
            # Health alerts
            if context.health_alerts:
                recommendations.append("I notice some health alerts - please take care of yourself")
            
            # Trend-based recommendations
            for trend in trends:
                if trend.metric_type == "stress_level" and trend.trend_direction == "increasing":
                    recommendations.append("I've noticed your stress levels trending upward - would you like to talk about stress management?")
                elif trend.metric_type == "sleep_analysis" and trend.trend_direction == "decreasing":
                    recommendations.append("Your sleep patterns have been changing - rest is so important for emotional wellbeing")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating emotional recommendations: {e}")
            return []
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get current health integration status"""
        return {
            "authorized": len(self.connector.authorized_metrics) > 0,
            "authorized_metrics": list(self.connector.authorized_metrics),
            "last_sync": self.connector.last_sync,
            "privacy_mode": self.connector.config.get("privacy_mode", "anonymized"),
            "real_time_monitoring": self.connector.config.get("real_time_monitoring", False)
        }

# Example usage and testing
async def demo_health_integration():
    """Demonstrate health integration capabilities"""
    health_integration = EmotionalHealthIntegration()
    
    print("=== Health Integration Demo ===")
    
    # Initialize
    if await health_integration.initialize():
        print("✓ Health integration initialized")
        
        # Get emotional health context
        context = await health_integration.get_emotional_health_context()
        
        print(f"\nHealth Context:")
        print(f"  Stress Level: {context['health_context'].get('stress_level', 'N/A')}")
        print(f"  Energy Level: {context['health_context'].get('energy_level', 'N/A')}")
        print(f"  Sleep Quality: {context['health_context'].get('sleep_quality', 'N/A')}")
        print(f"  Activity Level: {context['health_context'].get('activity_level', 'N/A')}")
        
        print(f"\nCurrent Readings: {len(context['current_readings'])} metrics")
        for reading in context['current_readings'][:3]:  # Show first 3
            print(f"  {reading['metric_type']}: {reading['value']} {reading['unit']}")
        
        print(f"\nHealth Trends: {len(context['health_trends'])} trends")
        for trend in context['health_trends']:
            print(f"  {trend['metric_type']}: {trend['trend_direction']} ({trend['percentage_change']:+.1f}%)")
        
        print(f"\nEmotional Recommendations:")
        for rec in context['emotional_recommendations']:
            print(f"  • {rec}")
        
        # Status check
        status = await health_integration.get_health_status()
        print(f"\nStatus: {status}")
    
    else:
        print("✗ Failed to initialize health integration")

if __name__ == "__main__":
    asyncio.run(demo_health_integration())

"""
Event Logger - Unified emotional event logging system
Central logging point for emotional changes across all subsystems
"""

import time
import os
import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum

class EventSeverity(Enum):
    """Event severity levels"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    NOTICE = "NOTICE"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"

@dataclass
class EmotionalEvent:
    """Structure for emotional event logging"""
    timestamp: float
    event_type: str
    intensity: float
    tag: str
    severity: EventSeverity
    context: Dict[str, Any]
    source_module: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None

class EventLogger:
    """Unified logging system for emotional events"""
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = log_dir
        self.emotional_log_file = os.path.join(log_dir, "emotional_events.log")
        self.json_log_file = os.path.join(log_dir, "emotional_events.json")
        self.event_history: List[EmotionalEvent] = []
        self.max_memory_events = 1000  # Keep in memory
        
        # Ensure log directory exists
        os.makedirs(log_dir, exist_ok=True)
        
        # Event type mappings for intensity interpretation
        self.intensity_mappings = {
            "emotion_shift": {"low": 0.3, "medium": 0.6, "high": 0.9},
            "bond_change": {"low": 0.2, "medium": 0.5, "high": 0.8},
            "ritual_trigger": {"low": 0.4, "medium": 0.7, "high": 1.0},
            "memory_formation": {"low": 0.3, "medium": 0.6, "high": 0.9},
            "personality_shift": {"low": 0.2, "medium": 0.5, "high": 0.8},
            "trust_event": {"low": 0.3, "medium": 0.6, "high": 0.9},
            "vulnerability_moment": {"low": 0.5, "medium": 0.7, "high": 1.0},
            "connection_deepening": {"low": 0.4, "medium": 0.7, "high": 0.9},
            "emotional_breakthrough": {"low": 0.6, "medium": 0.8, "high": 1.0}
        }
    
    def log_emotional_event(self, event_type: str, intensity: float, tag: str,
                          context: Optional[Dict[str, Any]] = None,
                          source_module: str = "unknown",
                          severity: EventSeverity = EventSeverity.INFO,
                          user_id: Optional[str] = None,
                          session_id: Optional[str] = None):
        """
        Unified logging point for emotional changes across subsystems.
        
        Args:
            event_type: Type of emotional event (e.g., "emotion_shift", "bond_change")
            intensity: Emotional intensity level (0.0-1.0)
            tag: Human-readable description of the event
            context: Additional context data
            source_module: Which module generated this event
            severity: Event severity level
            user_id: User identifier (if available)
            session_id: Session identifier (if available)
        """
        
        # Clamp intensity to valid range
        intensity = max(0.0, min(1.0, intensity))
        
        # Create event object
        event = EmotionalEvent(
            timestamp=time.time(),
            event_type=event_type,
            intensity=intensity,
            tag=tag,
            severity=severity,
            context=context or {},
            source_module=source_module,
            user_id=user_id,
            session_id=session_id
        )
        
        # Add to memory
        self.event_history.append(event)
        
        # Maintain memory limit
        if len(self.event_history) > self.max_memory_events:
            self.event_history = self.event_history[-self.max_memory_events:]
        
        # Write to logs
        self._write_to_text_log(event)
        self._write_to_json_log(event)
        
        # Handle high-severity events
        if severity in [EventSeverity.WARNING, EventSeverity.CRITICAL]:
            self._handle_high_severity_event(event)
    
    def _write_to_text_log(self, event: EmotionalEvent):
        """Write event to human-readable text log"""
        try:
            timestamp_str = datetime.fromtimestamp(event.timestamp).strftime("%Y-%m-%d %H:%M:%S")
            intensity_bar = "█" * int(event.intensity * 10)
            
            log_line = (f"[{timestamp_str}] [{event.severity.value}] "
                       f"[{event.event_type.upper()}] ({event.intensity:.2f}) "
                       f"{intensity_bar} - {event.tag}")
            
            if event.source_module != "unknown":
                log_line += f" ({event.source_module})"
            
            log_line += "\n"
            
            with open(self.emotional_log_file, "a", encoding="utf-8") as f:
                f.write(log_line)
                
        except Exception as e:
            print(f"Warning: Could not write to text log: {e}")
    
    def _write_to_json_log(self, event: EmotionalEvent):
        """Write event to structured JSON log"""
        try:
            with open(self.json_log_file, "a", encoding="utf-8") as f:
                event_dict = asdict(event)
                event_dict["severity"] = event.severity.value
                json.dump(event_dict, f)
                f.write("\n")
                
        except Exception as e:
            print(f"Warning: Could not write to JSON log: {e}")
    
    def _handle_high_severity_event(self, event: EmotionalEvent):
        """Handle high-severity events with special processing"""
        if event.severity == EventSeverity.CRITICAL:
            # For critical events, also log to separate critical log
            critical_log = os.path.join(self.log_dir, "critical_emotional_events.log")
            try:
                timestamp_str = datetime.fromtimestamp(event.timestamp).strftime("%Y-%m-%d %H:%M:%S")
                with open(critical_log, "a", encoding="utf-8") as f:
                    f.write(f"[{timestamp_str}] CRITICAL: {event.tag} "
                           f"(intensity: {event.intensity}, module: {event.source_module})\n")
            except Exception as e:
                print(f"Warning: Could not write to critical log: {e}")
    
    def get_recent_events(self, minutes: int = 60, 
                         event_type: Optional[str] = None) -> List[EmotionalEvent]:
        """Get recent emotional events"""
        cutoff_time = time.time() - (minutes * 60)
        
        recent = [e for e in self.event_history if e.timestamp > cutoff_time]
        
        if event_type:
            recent = [e for e in recent if e.event_type == event_type]
        
        return sorted(recent, key=lambda x: x.timestamp, reverse=True)
    
    def get_event_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get summary of events over specified time period"""
        cutoff_time = time.time() - (hours * 3600)
        recent_events = [e for e in self.event_history if e.timestamp > cutoff_time]
        
        if not recent_events:
            return {"total_events": 0, "time_period_hours": hours}
        
        # Event type counts
        type_counts = {}
        for event in recent_events:
            type_counts[event.event_type] = type_counts.get(event.event_type, 0) + 1
        
        # Intensity statistics
        intensities = [e.intensity for e in recent_events]
        avg_intensity = sum(intensities) / len(intensities)
        max_intensity = max(intensities)
        min_intensity = min(intensities)
        
        # Severity distribution
        severity_counts = {}
        for event in recent_events:
            severity_counts[event.severity.value] = severity_counts.get(event.severity.value, 0) + 1
        
        # Module activity
        module_counts = {}
        for event in recent_events:
            module_counts[event.source_module] = module_counts.get(event.source_module, 0) + 1
        
        return {
            "total_events": len(recent_events),
            "time_period_hours": hours,
            "average_intensity": round(avg_intensity, 3),
            "max_intensity": max_intensity,
            "min_intensity": min_intensity,
            "event_types": type_counts,
            "severity_distribution": severity_counts,
            "module_activity": module_counts,
            "most_active_module": max(module_counts.keys(), key=lambda k: module_counts[k]) if module_counts else None
        }
    
    def detect_emotional_patterns(self, hours: int = 24) -> List[str]:
        """Detect patterns in emotional events"""
        recent_events = self.get_recent_events(minutes=hours * 60)
        patterns = []
        
        if len(recent_events) < 3:
            return patterns
        
        # Intensity escalation pattern
        intensities = [e.intensity for e in recent_events[-5:]]
        if len(intensities) >= 3:
            if all(intensities[i] < intensities[i+1] for i in range(len(intensities)-1)):
                patterns.append("Emotional intensity escalating")
            elif all(intensities[i] > intensities[i+1] for i in range(len(intensities)-1)):
                patterns.append("Emotional intensity declining")
        
        # Frequent event type
        type_counts = {}
        for event in recent_events[-10:]:  # Last 10 events
            type_counts[event.event_type] = type_counts.get(event.event_type, 0) + 1
        
        if type_counts:
            most_frequent = max(type_counts.keys(), key=lambda k: type_counts[k])
            if type_counts[most_frequent] >= 3:
                patterns.append(f"Frequent {most_frequent} events")
        
        # High intensity clustering
        high_intensity_events = [e for e in recent_events if e.intensity > 0.7]
        if len(high_intensity_events) >= 3:
            patterns.append("Multiple high-intensity emotional events")
        
        return patterns
    
    def log_batch_events(self, events: List[Tuple[str, float, str]],
                        source_module: str = "batch_import"):
        """Log multiple events in batch"""
        for event_type, intensity, tag in events:
            self.log_emotional_event(
                event_type=event_type,
                intensity=intensity,
                tag=tag,
                source_module=source_module
            )
    
    def export_events_to_file(self, filepath: str, hours: int = 24,
                            format: str = "json") -> bool:
        """Export events to file for analysis"""
        try:
            recent_events = self.get_recent_events(minutes=hours * 60)
            
            if format.lower() == "json":
                data = {
                    "export_timestamp": time.time(),
                    "time_period_hours": hours,
                    "total_events": len(recent_events),
                    "events": [asdict(e) for e in recent_events]
                }
                
                with open(filepath, 'w', encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
            
            elif format.lower() == "csv":
                import csv
                with open(filepath, 'w', newline='', encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(["timestamp", "event_type", "intensity", "tag", 
                                   "severity", "source_module"])
                    
                    for event in recent_events:
                        writer.writerow([
                            datetime.fromtimestamp(event.timestamp).isoformat(),
                            event.event_type,
                            event.intensity,
                            event.tag,
                            event.severity.value,
                            event.source_module
                        ])
            
            return True
            
        except Exception as e:
            print(f"Error exporting events: {e}")
            return False

# Global logger instance
_global_logger = None

def get_logger() -> EventLogger:
    """Get global logger instance"""
    global _global_logger
    if _global_logger is None:
        _global_logger = EventLogger()
    return _global_logger

# Convenience function for easy importing
def log_emotional_event(event_type: str, intensity: float, tag: str,
                       context: Optional[Dict[str, Any]] = None,
                       source_module: str = "unknown"):
    """
    Standalone function for emotional event logging.
    Unified logging point for emotional changes across subsystems.
    """
    logger = get_logger()
    logger.log_emotional_event(
        event_type=event_type,
        intensity=intensity,
        tag=tag,
        context=context,
        source_module=source_module
    )

# Example usage
if __name__ == "__main__":
    logger = EventLogger()
    
    # Test different types of emotional events
    test_events = [
        ("emotion_shift", 0.7, "User expressed deep sadness about loss", "emotion_state_manager"),
        ("bond_change", 0.8, "Trust level increased after vulnerability sharing", "connection_depth_tracker"),
        ("ritual_trigger", 0.9, "Intimacy ritual triggered: confession request", "guidance_coordinator"),
        ("memory_formation", 0.6, "Significant conversation stored in long-term memory", "memory_manager"),
        ("personality_shift", 0.4, "Personality became more gentle after feedback", "personality_evolution"),
    ]
    
    for event_type, intensity, tag, module in test_events:
        logger.log_emotional_event(
            event_type=event_type,
            intensity=intensity,
            tag=tag,
            source_module=module,
            context={"test_run": True}
        )
    
    # Test analysis features
    print("Recent events:", len(logger.get_recent_events(minutes=1)))
    print("Event summary:", logger.get_event_summary(hours=1))
    print("Patterns detected:", logger.detect_emotional_patterns(hours=1))

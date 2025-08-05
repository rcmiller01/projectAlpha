"""
Enhanced Logging and Explainability System

Provides comprehensive debug logging, decision tracking, and explainability
for the unified companion system.
"""

import logging
import json
import time
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass, asdict
from contextlib import contextmanager
from enum import Enum

class LogLevel(Enum):
    TRACE = "TRACE"
    DEBUG = "DEBUG" 
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class DecisionCategory(Enum):
    MODE_DETECTION = "mode_detection"
    EMOTIONAL_ANALYSIS = "emotional_analysis"
    CRISIS_ASSESSMENT = "crisis_assessment"
    GUIDANCE_SYNTHESIS = "guidance_synthesis"
    RESPONSE_GENERATION = "response_generation"
    MEMORY_RETRIEVAL = "memory_retrieval"
    ADAPTATION_LOGIC = "adaptation_logic"

@dataclass
class DecisionPoint:
    """Individual decision point in the processing flow"""
    category: DecisionCategory
    description: str
    input_data: Dict[str, Any]
    decision_logic: str
    output_result: Any
    confidence_score: float
    processing_time_ms: float
    timestamp: datetime
    context_factors: Dict[str, Any]

@dataclass
class ProcessingTrace:
    """Complete trace of processing decisions"""
    interaction_id: str
    user_id: str
    session_id: str
    start_time: datetime
    end_time: Optional[datetime]
    total_processing_time_ms: float
    decision_points: List[DecisionPoint]
    final_outcome: Dict[str, Any]
    performance_metrics: Dict[str, float]

class EnhancedLogger:
    """
    Enhanced logging system with decision tracking and explainability
    """
    
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        
        # Set up base logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, config.get("log_level", "INFO")))
        
        # Set up specialized loggers
        self.decision_logger = logging.getLogger(f"{name}.decisions")
        self.performance_logger = logging.getLogger(f"{name}.performance")
        self.explainability_logger = logging.getLogger(f"{name}.explainability")
        
        # Configure handlers
        self._setup_handlers()
        
        # Decision tracking
        self.current_trace: Optional[ProcessingTrace] = None
        self.decision_history: List[ProcessingTrace] = []
        self.max_history_size = config.get("max_history_size", 1000)
        
        # Performance monitoring
        self.performance_metrics = {
            "total_interactions": 0,
            "average_processing_time": 0.0,
            "crisis_interventions": 0,
            "mode_transitions": 0,
            "memory_retrievals": 0
        }
    
    def _setup_handlers(self):
        """Set up logging handlers with proper formatting"""
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Console handler for development
        if self.config.get("console_logging", True):
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        # File handlers for production
        if self.config.get("file_logging", True):
            # Main log file
            main_handler = logging.FileHandler(f"{self.name}_main.log")
            main_handler.setFormatter(formatter)
            self.logger.addHandler(main_handler)
            
            # Decision log file
            decision_handler = logging.FileHandler(f"{self.name}_decisions.log")
            decision_handler.setFormatter(formatter)
            self.decision_logger.addHandler(decision_handler)
            
            # Performance log file
            performance_handler = logging.FileHandler(f"{self.name}_performance.log")
            performance_handler.setFormatter(formatter)
            self.performance_logger.addHandler(performance_handler)
    
    def start_interaction_trace(self, interaction_id: str, user_id: str, session_id: str):
        """Start tracing a new interaction"""
        self.current_trace = ProcessingTrace(
            interaction_id=interaction_id,
            user_id=user_id,
            session_id=session_id,
            start_time=datetime.now(),
            end_time=None,
            total_processing_time_ms=0.0,
            decision_points=[],
            final_outcome={},
            performance_metrics={}
        )
        
        self.logger.info(f"Started interaction trace: {interaction_id}")
    
    def log_decision(self, category: DecisionCategory, description: str, 
                    input_data: Dict[str, Any], decision_logic: str,
                    output_result: Any, confidence_score: float,
                    context_factors: Optional[Dict[str, Any]] = None):
        """Log a decision point with full context"""
        
        if not self.current_trace:
            self.logger.warning("No active trace for decision logging")
            return
        
        processing_start = time.time()
        
        decision_point = DecisionPoint(
            category=category,
            description=description,
            input_data=self._sanitize_for_logging(input_data),
            decision_logic=decision_logic,
            output_result=self._sanitize_for_logging(output_result),
            confidence_score=confidence_score,
            processing_time_ms=(time.time() - processing_start) * 1000,
            timestamp=datetime.now(),
            context_factors=context_factors or {}
        )
        
        self.current_trace.decision_points.append(decision_point)
        
        # Log decision with appropriate level
        log_level = LogLevel.DEBUG
        if category == DecisionCategory.CRISIS_ASSESSMENT and confidence_score > 0.5:
            log_level = LogLevel.WARNING
        elif confidence_score < 0.3:
            log_level = LogLevel.WARNING
        
        self.decision_logger.log(
            getattr(logging, log_level.value),
            f"[{category.value}] {description} | Confidence: {confidence_score:.3f} | Logic: {decision_logic}"
        )
        
        # Debug-level detailed logging
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(f"Decision Details - {category.value}:")
            self.logger.debug(f"  Input: {json.dumps(decision_point.input_data, indent=2, default=str)}")
            self.logger.debug(f"  Output: {json.dumps(decision_point.output_result, indent=2, default=str)}")
            self.logger.debug(f"  Context: {json.dumps(decision_point.context_factors, indent=2, default=str)}")
    
    def finish_interaction_trace(self, final_outcome: Dict[str, Any]):
        """Complete the current interaction trace"""
        if not self.current_trace:
            self.logger.warning("No active trace to finish")
            return
        
        self.current_trace.end_time = datetime.now()
        self.current_trace.total_processing_time_ms = (
            self.current_trace.end_time - self.current_trace.start_time
        ).total_seconds() * 1000
        
        self.current_trace.final_outcome = self._sanitize_for_logging(final_outcome)
        
        # Calculate performance metrics
        self.current_trace.performance_metrics = {
            "total_decisions": len(self.current_trace.decision_points),
            "processing_time_ms": self.current_trace.total_processing_time_ms,
            "average_decision_time_ms": (
                sum(dp.processing_time_ms for dp in self.current_trace.decision_points) /
                len(self.current_trace.decision_points)
            ) if self.current_trace.decision_points else 0.0,
            "crisis_decisions": len([
                dp for dp in self.current_trace.decision_points 
                if dp.category == DecisionCategory.CRISIS_ASSESSMENT
            ]),
            "low_confidence_decisions": len([
                dp for dp in self.current_trace.decision_points 
                if dp.confidence_score < 0.5
            ])
        }
        
        # Store trace in history
        self.decision_history.append(self.current_trace)
        
        # Trim history if needed
        if len(self.decision_history) > self.max_history_size:
            self.decision_history = self.decision_history[-self.max_history_size:]
        
        # Log completion
        self.logger.info(
            f"Completed interaction trace: {self.current_trace.interaction_id} "
            f"({self.current_trace.total_processing_time_ms:.1f}ms, "
            f"{len(self.current_trace.decision_points)} decisions)"
        )
        
        # Performance logging
        self.performance_logger.info(
            f"Performance - ID: {self.current_trace.interaction_id}, "
            f"Time: {self.current_trace.total_processing_time_ms:.1f}ms, "
            f"Decisions: {len(self.current_trace.decision_points)}, "
            f"Avg Decision Time: {self.current_trace.performance_metrics['average_decision_time_ms']:.1f}ms"
        )
        
        # Update global performance metrics
        self._update_performance_metrics()
        
        # Clear current trace
        self.current_trace = None
    
    def _update_performance_metrics(self):
        """Update global performance metrics"""
        if not self.decision_history:
            return
        
        self.performance_metrics["total_interactions"] = len(self.decision_history)
        
        total_time = sum(trace.total_processing_time_ms for trace in self.decision_history)
        self.performance_metrics["average_processing_time"] = total_time / len(self.decision_history)
        
        self.performance_metrics["crisis_interventions"] = sum(
            trace.performance_metrics.get("crisis_decisions", 0) 
            for trace in self.decision_history
        )
        
        # Count mode transitions (decisions that change mode)
        mode_transitions = 0
        for trace in self.decision_history:
            mode_decisions = [
                dp for dp in trace.decision_points 
                if dp.category == DecisionCategory.MODE_DETECTION
            ]
            if len(mode_decisions) > 1:
                mode_transitions += len(mode_decisions) - 1
        
        self.performance_metrics["mode_transitions"] = mode_transitions
        
        self.performance_metrics["memory_retrievals"] = sum(
            len([dp for dp in trace.decision_points if dp.category == DecisionCategory.MEMORY_RETRIEVAL])
            for trace in self.decision_history
        )
    
    def _sanitize_for_logging(self, data: Any) -> Any:
        """Sanitize data for safe logging (remove sensitive info)"""
        if isinstance(data, dict):
            sanitized = {}
            for key, value in data.items():
                # Skip potentially sensitive fields
                if key.lower() in ["password", "token", "key", "secret", "private"]:
                    sanitized[key] = "[REDACTED]"
                elif isinstance(value, (dict, list)):
                    sanitized[key] = self._sanitize_for_logging(value)
                elif isinstance(value, str) and len(value) > 1000:
                    sanitized[key] = value[:1000] + "... [TRUNCATED]"
                else:
                    sanitized[key] = value
            return sanitized
        elif isinstance(data, list):
            return [self._sanitize_for_logging(item) for item in data[:100]]  # Limit list size
        else:
            return data
    
    @contextmanager
    def decision_context(self, category: DecisionCategory, description: str):
        """Context manager for timing and logging decisions"""
        start_time = time.time()
        context_data = {
            "start_time": start_time,
            "category": category,
            "description": description
        }
        
        try:
            yield context_data
        except Exception as e:
            self.logger.error(f"Error in decision context [{category.value}] {description}: {e}")
            raise
        finally:
            processing_time = (time.time() - start_time) * 1000
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(f"Decision timing [{category.value}] {description}: {processing_time:.1f}ms")
    
    def get_explainability_report(self, interaction_id: Optional[str] = None) -> Dict[str, Any]:
        """Generate explainability report for decisions"""
        if interaction_id:
            # Find specific interaction
            trace = next(
                (t for t in self.decision_history if t.interaction_id == interaction_id),
                self.current_trace if self.current_trace and self.current_trace.interaction_id == interaction_id else None
            )
            if not trace:
                return {"error": f"Interaction {interaction_id} not found"}
            
            traces = [trace]
        else:
            # Use current trace or most recent
            traces = [self.current_trace] if self.current_trace else self.decision_history[-1:]
        
        if not traces:
            return {"error": "No interaction data available"}
        
        report = {
            "generated_at": datetime.now().isoformat(),
            "interactions_analyzed": len(traces),
            "decision_breakdown": {},
            "performance_summary": {},
            "recommendations": []
        }
        
        # Analyze decision patterns
        all_decisions = []
        for trace in traces:
            all_decisions.extend(trace.decision_points)
        
        # Group by category
        for category in DecisionCategory:
            category_decisions = [dp for dp in all_decisions if dp.category == category]
            if category_decisions:
                report["decision_breakdown"][category.value] = {
                    "count": len(category_decisions),
                    "average_confidence": sum(dp.confidence_score for dp in category_decisions) / len(category_decisions),
                    "average_processing_time_ms": sum(dp.processing_time_ms for dp in category_decisions) / len(category_decisions),
                    "low_confidence_count": len([dp for dp in category_decisions if dp.confidence_score < 0.5])
                }
        
        # Performance summary
        if traces:
            avg_processing_time = sum(t.total_processing_time_ms for t in traces) / len(traces)
            avg_decisions_per_interaction = sum(len(t.decision_points) for t in traces) / len(traces)
            
            report["performance_summary"] = {
                "average_processing_time_ms": avg_processing_time,
                "average_decisions_per_interaction": avg_decisions_per_interaction,
                "total_decisions_analyzed": len(all_decisions)
            }
        
        # Generate recommendations
        recommendations = []
        
        # Check for performance issues
        if report["performance_summary"].get("average_processing_time_ms", 0) > 1000:
            recommendations.append("Consider optimizing processing pipeline - average response time exceeds 1 second")
        
        # Check for low confidence patterns
        for category, stats in report["decision_breakdown"].items():
            if stats["average_confidence"] < 0.6:
                recommendations.append(f"Low confidence in {category} decisions - consider model retraining or rule refinement")
            
            if stats["low_confidence_count"] > stats["count"] * 0.3:
                recommendations.append(f"High percentage of low-confidence {category} decisions - review decision logic")
        
        report["recommendations"] = recommendations
        
        return report
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        return {
            **self.performance_metrics,
            "history_size": len(self.decision_history),
            "current_trace_active": self.current_trace is not None,
            "last_updated": datetime.now().isoformat()
        }
    
    # Convenience methods for different log levels
    def trace(self, message: str, **kwargs):
        """Trace-level logging (most verbose)"""
        if self.logger.isEnabledFor(5):  # Custom TRACE level
            self.logger.log(5, message, **kwargs)
    
    def debug(self, message: str, **kwargs):
        """Debug-level logging"""
        self.logger.debug(message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """Info-level logging"""
        self.logger.info(message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Warning-level logging"""
        self.logger.warning(message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Error-level logging"""
        self.logger.error(message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """Critical-level logging"""
        self.logger.critical(message, **kwargs)

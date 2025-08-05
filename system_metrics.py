"""
📈 System Status & Metrics Endpoint
Comprehensive system monitoring and metrics collection
for the Dolphin AI Orchestrator v2.0
"""

import psutil
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from collections import defaultdict, deque

@dataclass
class SystemMetrics:
    """System performance metrics snapshot"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    memory_total_gb: float
    disk_usage_percent: float
    active_sessions: int
    total_requests: int
    avg_response_time: float
    error_rate: float
    uptime_seconds: float

class MetricsCollector:
    """
    Collects and manages system metrics and performance data
    """
    
    def __init__(self, analytics_logger=None):
        self.analytics_logger = analytics_logger
        self.start_time = datetime.now()
        
        # Metrics storage
        self.metrics_history = deque(maxlen=1440)  # 24 hours of minute-by-minute data
        self.hourly_aggregates = deque(maxlen=168)  # 7 days of hourly data
        self.daily_aggregates = deque(maxlen=30)    # 30 days of daily data
        
        # Real-time counters
        self.request_counter = 0
        self.error_counter = 0
        self.response_times = deque(maxlen=1000)
        self.active_sessions = set()
        
        # Model usage tracking
        self.model_usage_stats = defaultdict(lambda: {
            'requests': 0,
            'total_time': 0.0,
            'errors': 0,
            'last_used': None
        })
        
        # Feature usage tracking
        self.feature_usage = defaultdict(int)
        
        # Memory usage tracking
        self.memory_stats = {
            'session_memory_mb': 0.0,
            'longterm_memory_mb': 0.0,
            'private_memory_mb': 0.0,
            'reflection_memory_mb': 0.0
        }
        
        # Routing analytics
        self.routing_stats = {
            'total_decisions': 0,
            'local_routes': 0,
            'cloud_routes': 0,
            'fallback_routes': 0,
            'routing_accuracy': 0.0
        }
        
        print("📈 Metrics Collector initialized")
    
    def collect_current_metrics(self) -> SystemMetrics:
        """Collect current system metrics"""
        try:
            # System metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Calculate averages
            avg_response_time = (
                sum(self.response_times) / len(self.response_times)
                if self.response_times else 0.0
            )
            
            error_rate = (
                self.error_counter / max(self.request_counter, 1)
                if self.request_counter > 0 else 0.0
            )
            
            uptime = (datetime.now() - self.start_time).total_seconds()
            
            metrics = SystemMetrics(
                timestamp=datetime.now(),
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_used_gb=memory.used / (1024**3),
                memory_total_gb=memory.total / (1024**3),
                disk_usage_percent=disk.percent,
                active_sessions=len(self.active_sessions),
                total_requests=self.request_counter,
                avg_response_time=avg_response_time,
                error_rate=error_rate * 100,  # Convert to percentage
                uptime_seconds=uptime
            )
            
            # Store in history
            self.metrics_history.append(metrics)
            
            return metrics
            
        except Exception as e:
            print(f"❌ Error collecting metrics: {e}")
            # Return basic metrics on error
            return SystemMetrics(
                timestamp=datetime.now(),
                cpu_percent=0.0,
                memory_percent=0.0,
                memory_used_gb=0.0,
                memory_total_gb=0.0,
                disk_usage_percent=0.0,
                active_sessions=0,
                total_requests=self.request_counter,
                avg_response_time=0.0,
                error_rate=0.0,
                uptime_seconds=(datetime.now() - self.start_time).total_seconds()
            )
    
    def record_request(self, session_id: str, model_used: str, response_time: float, success: bool = True):
        """Record a completed request"""
        self.request_counter += 1
        self.response_times.append(response_time)
        self.active_sessions.add(session_id)
        
        if not success:
            self.error_counter += 1
        
        # Update model usage stats
        model_stats = self.model_usage_stats[model_used]
        model_stats['requests'] += 1
        model_stats['total_time'] += response_time
        model_stats['last_used'] = datetime.now().isoformat()
        
        if not success:
            model_stats['errors'] += 1
        
        # Log to analytics if available
        if self.analytics_logger:
            self.analytics_logger.log_custom_event(
                "request_completed",
                {
                    'session_id': session_id,
                    'model_used': model_used,
                    'response_time': response_time,
                    'success': success
                }
            )
    
    def record_feature_usage(self, feature_name: str, count: int = 1):
        """Record usage of a specific feature"""
        self.feature_usage[feature_name] += count
    
    def record_routing_decision(self, decision_type: str, was_accurate: bool = True):
        """Record routing decision for analytics"""
        self.routing_stats['total_decisions'] += 1
        
        if decision_type == 'local':
            self.routing_stats['local_routes'] += 1
        elif decision_type == 'cloud':
            self.routing_stats['cloud_routes'] += 1
        elif decision_type == 'fallback':
            self.routing_stats['fallback_routes'] += 1
        
        # Update accuracy (simple moving average)
        current_accuracy = self.routing_stats['routing_accuracy']
        total_decisions = self.routing_stats['total_decisions']
        
        if total_decisions == 1:
            self.routing_stats['routing_accuracy'] = 1.0 if was_accurate else 0.0
        else:
            # Weighted average favoring recent decisions
            weight = 0.1  # How much to weight the new decision
            self.routing_stats['routing_accuracy'] = (
                current_accuracy * (1 - weight) + (1.0 if was_accurate else 0.0) * weight
            )
    
    def update_memory_stats(self, memory_type: str, size_mb: float):
        """Update memory usage statistics"""
        if memory_type in self.memory_stats:
            self.memory_stats[memory_type] = size_mb
    
    def get_realtime_status(self) -> Dict[str, Any]:
        """Get real-time system status"""
        current_metrics = self.collect_current_metrics()
        
        return {
            'timestamp': current_metrics.timestamp.isoformat(),
            'system': {
                'cpu_percent': current_metrics.cpu_percent,
                'memory_percent': current_metrics.memory_percent,
                'memory_used_gb': round(current_metrics.memory_used_gb, 2),
                'memory_total_gb': round(current_metrics.memory_total_gb, 2),
                'disk_usage_percent': current_metrics.disk_usage_percent,
                'uptime_hours': round(current_metrics.uptime_seconds / 3600, 2)
            },
            'application': {
                'active_sessions': current_metrics.active_sessions,
                'total_requests': current_metrics.total_requests,
                'avg_response_time_ms': round(current_metrics.avg_response_time * 1000, 2),
                'error_rate_percent': round(current_metrics.error_rate, 2),
                'requests_per_minute': self._calculate_requests_per_minute()
            },
            'memory_breakdown': self.memory_stats,
            'routing': {
                'total_decisions': self.routing_stats['total_decisions'],
                'local_percentage': round(
                    (self.routing_stats['local_routes'] / max(self.routing_stats['total_decisions'], 1)) * 100, 1
                ),
                'cloud_percentage': round(
                    (self.routing_stats['cloud_routes'] / max(self.routing_stats['total_decisions'], 1)) * 100, 1
                ),
                'fallback_percentage': round(
                    (self.routing_stats['fallback_routes'] / max(self.routing_stats['total_decisions'], 1)) * 100, 1
                ),
                'routing_accuracy': round(self.routing_stats['routing_accuracy'] * 100, 1)
            }
        }
    
    def get_model_usage_report(self) -> Dict[str, Any]:
        """Get detailed model usage statistics"""
        report = {}
        
        for model, stats in self.model_usage_stats.items():
            avg_response_time = (
                stats['total_time'] / max(stats['requests'], 1)
                if stats['requests'] > 0 else 0.0
            )
            
            error_rate = (
                stats['errors'] / max(stats['requests'], 1) * 100
                if stats['requests'] > 0 else 0.0
            )
            
            report[model] = {
                'total_requests': stats['requests'],
                'avg_response_time_ms': round(avg_response_time * 1000, 2),
                'error_rate_percent': round(error_rate, 2),
                'total_time_spent_sec': round(stats['total_time'], 2),
                'last_used': stats['last_used']
            }
        
        return report
    
    def get_feature_usage_report(self) -> Dict[str, int]:
        """Get feature usage statistics"""
        return dict(self.feature_usage)
    
    def get_historical_data(self, period: str = 'hour', limit: int = 24) -> List[Dict[str, Any]]:
        """Get historical metrics data"""
        if period == 'minute':
            data_source = list(self.metrics_history)[-limit:]
        elif period == 'hour':
            data_source = list(self.hourly_aggregates)[-limit:]
        elif period == 'day':
            data_source = list(self.daily_aggregates)[-limit:]
        else:
            data_source = list(self.metrics_history)[-limit:]
        
        return [
            {
                'timestamp': metrics.timestamp.isoformat() if hasattr(metrics, 'timestamp') else metrics['timestamp'],
                'cpu_percent': metrics.cpu_percent if hasattr(metrics, 'cpu_percent') else metrics['cpu_percent'],
                'memory_percent': metrics.memory_percent if hasattr(metrics, 'memory_percent') else metrics['memory_percent'],
                'active_sessions': metrics.active_sessions if hasattr(metrics, 'active_sessions') else metrics['active_sessions'],
                'avg_response_time': metrics.avg_response_time if hasattr(metrics, 'avg_response_time') else metrics['avg_response_time']
            }
            for metrics in data_source
        ]
    
    def _calculate_requests_per_minute(self) -> float:
        """Calculate requests per minute based on recent data"""
        if len(self.metrics_history) < 2:
            return 0.0
        
        # Look at last 5 minutes of data
        recent_metrics = list(self.metrics_history)[-5:]
        if len(recent_metrics) < 2:
            return 0.0
        
        time_span = (recent_metrics[-1].timestamp - recent_metrics[0].timestamp).total_seconds() / 60
        request_diff = recent_metrics[-1].total_requests - recent_metrics[0].total_requests
        
        return round(request_diff / max(time_span, 1), 2)
    
    def aggregate_hourly_data(self):
        """Aggregate minute data into hourly summaries"""
        if not self.metrics_history:
            return
        
        # Group metrics by hour
        current_hour = datetime.now().replace(minute=0, second=0, microsecond=0)
        hour_metrics = [m for m in self.metrics_history if m.timestamp >= current_hour - timedelta(hours=1)]
        
        if not hour_metrics:
            return
        
        # Calculate hourly aggregates
        hourly_summary = {
            'timestamp': current_hour.isoformat(),
            'avg_cpu_percent': sum(m.cpu_percent for m in hour_metrics) / len(hour_metrics),
            'avg_memory_percent': sum(m.memory_percent for m in hour_metrics) / len(hour_metrics),
            'max_active_sessions': max(m.active_sessions for m in hour_metrics),
            'total_requests': max(m.total_requests for m in hour_metrics) - min(m.total_requests for m in hour_metrics),
            'avg_response_time': sum(m.avg_response_time for m in hour_metrics) / len(hour_metrics),
            'avg_error_rate': sum(m.error_rate for m in hour_metrics) / len(hour_metrics)
        }
        
        self.hourly_aggregates.append(hourly_summary)
    
    def export_metrics(self, format_type: str = 'json') -> str:
        """Export metrics data"""
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'system_info': {
                'start_time': self.start_time.isoformat(),
                'uptime_seconds': (datetime.now() - self.start_time).total_seconds()
            },
            'current_status': self.get_realtime_status(),
            'model_usage': self.get_model_usage_report(),
            'feature_usage': self.get_feature_usage_report(),
            'historical_data': {
                'hourly': self.get_historical_data('hour', 24),
                'daily': self.get_historical_data('day', 7)
            },
            'routing_stats': self.routing_stats,
            'memory_stats': self.memory_stats
        }
        
        if format_type == 'json':
            return json.dumps(export_data, indent=2)
        else:
            return str(export_data)
    
    def get_health_check(self) -> Dict[str, Any]:
        """Get basic health check information"""
        current_metrics = self.collect_current_metrics()
        
        # Determine health status
        health_status = "healthy"
        issues = []
        
        if current_metrics.cpu_percent > 80:
            health_status = "warning"
            issues.append("High CPU usage")
        
        if current_metrics.memory_percent > 85:
            health_status = "warning"
            issues.append("High memory usage")
        
        if current_metrics.error_rate > 5:  # 5% error rate
            health_status = "warning"
            issues.append("Elevated error rate")
        
        if current_metrics.avg_response_time > 5:  # 5 second average
            health_status = "warning"
            issues.append("Slow response times")
        
        if issues and health_status == "warning":
            if len(issues) > 2 or current_metrics.cpu_percent > 95 or current_metrics.memory_percent > 95:
                health_status = "critical"
        
        return {
            'status': health_status,
            'timestamp': current_metrics.timestamp.isoformat(),
            'uptime_seconds': current_metrics.uptime_seconds,
            'issues': issues,
            'quick_stats': {
                'cpu_percent': current_metrics.cpu_percent,
                'memory_percent': current_metrics.memory_percent,
                'active_sessions': current_metrics.active_sessions,
                'total_requests': current_metrics.total_requests,
                'error_rate': current_metrics.error_rate
            }
        }

# Global metrics collector instance
metrics_collector = None

def get_metrics_collector():
    """Get the global metrics collector instance"""
    return metrics_collector

def initialize_metrics_collector(analytics_logger=None):
    """Initialize the global metrics collector"""
    global metrics_collector
    metrics_collector = MetricsCollector(analytics_logger)
    return metrics_collector

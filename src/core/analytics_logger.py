#!/usr/bin/env python3
"""
Logging and Analytics System for Dolphin AI Orchestrator

Tracks routing decisions, model performance, token usage,
and system analytics for transparency and optimization.
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from pathlib import Path
import time
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

class AnalyticsLogger:
    """
    Comprehensive logging and analytics for AI orchestration
    """
    
    def __init__(self, logs_dir: str = "logs"):
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(exist_ok=True)
        
        # Log files
        self.routing_log_file = self.logs_dir / "routing_decisions.jsonl"
        self.performance_log_file = self.logs_dir / "performance_metrics.jsonl"
        self.analytics_file = self.logs_dir / "analytics_summary.json"
        self.route_map_file = self.logs_dir / "routing_map.jsonl"
        self.custom_events_file = self.logs_dir / "custom_events.jsonl"
        self.preference_votes_file = self.logs_dir / "emotion_votes.jsonl"
        
        # In-memory buffers for real-time analytics
        self.recent_requests = deque(maxlen=1000)  # Last 1000 requests
        self.handler_stats = defaultdict(lambda: {
            "count": 0,
            "total_latency": 0.0,
            "avg_latency": 0.0,
            "success_count": 0,
            "error_count": 0,
            "total_tokens": 0
        })
        
        self.daily_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "requests": 0,
            "handlers_used": defaultdict(int),
            "avg_sentiment": 0.0,
            "error_rate": 0.0
        })

        # Preference vote tracking
        self.vote_counts = {"a": 0, "b": 0, "total": 0}
        
        logger.info("📊 Analytics Logger initialized")
    
    def log_routing_decision(self, request_data: Dict[str, Any], routing_result: Dict[str, Any], 
                           persona: str = "unknown") -> str:
        """Log a routing decision with full context"""
        
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "request_id": f"{int(time.time() * 1000)}_{hash(request_data.get('message', ''))%10000:04d}",
            "message_preview": request_data.get("message", "")[:100],
            "message_length": len(request_data.get("message", "")),
            "persona": persona,
            "routing": {
                "task_type": routing_result.get("task_type"),
                "handler": routing_result.get("handler"),
                "confidence": routing_result.get("confidence"),
                "reasoning": routing_result.get("reasoning")
            },
            "context": {
                "session_id": request_data.get("session_id"),
                "has_context": bool(request_data.get("context")),
                "context_size": len(str(request_data.get("context", {})))
            }
        }
        
        # Write to log file
        self._append_to_jsonl(self.routing_log_file, log_entry)

        # Update route map for transparency
        self.log_route_map(
            log_entry["context"].get("session_id"),
            request_data.get("message", ""),
            routing_result.get("handler"),
            persona,
            routing_result.get("confidence")
        )
        
        # Add to recent requests
        self.recent_requests.append(log_entry)

        return log_entry["request_id"]

    def log_route_map(self, session_id: str, input_text: str, routed_to: str,
                       persona: str, confidence: float):
        """Log simplified routing map for transparency"""

        entry = {
            "timestamp": datetime.now().isoformat(),
            "session_id": session_id,
            "input": input_text[:200],
            "routed_to": routed_to,
            "persona": persona,
            "confidence": confidence,
        }

        self._append_to_jsonl(self.route_map_file, entry)
    
    def log_performance_metrics(self, request_id: str, handler: str, 
                              latency: float, success: bool, 
                              token_usage: Optional[Dict[str, int]] = None,
                              error_message: Optional[str] = None):
        """Log performance metrics for a completed request"""
        
        perf_entry = {
            "timestamp": datetime.now().isoformat(),
            "request_id": request_id,
            "handler": handler,
            "latency_seconds": round(latency, 3),
            "success": success,
            "token_usage": token_usage or {},
            "error_message": error_message
        }
        
        # Write to performance log
        self._append_to_jsonl(self.performance_log_file, perf_entry)
        
        # Update handler statistics
        stats = self.handler_stats[handler]
        stats["count"] += 1
        stats["total_latency"] += latency
        stats["avg_latency"] = stats["total_latency"] / stats["count"]
        
        if success:
            stats["success_count"] += 1
        else:
            stats["error_count"] += 1
        
        if token_usage:
            total_tokens = token_usage.get("prompt_tokens", 0) + token_usage.get("completion_tokens", 0)
            stats["total_tokens"] += total_tokens
        
        # Update daily stats
        today = datetime.now().strftime("%Y-%m-%d")
        daily = self.daily_stats[today]
        daily["requests"] += 1
        daily["handlers_used"][handler] += 1
        
        if not success:
            # Recalculate error rate
            total_requests = daily["requests"]
            error_requests = sum(1 for req in self.recent_requests 
                               if req["timestamp"].startswith(today) and 
                               req.get("success", True) is False)
            daily["error_rate"] = error_requests / total_requests if total_requests > 0 else 0.0
    
    def _append_to_jsonl(self, file_path: Path, data: Dict[str, Any]):
        """Append JSON data to a JSONL file"""
        try:
            with open(file_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(data) + '\n')
        except Exception as e:
            logger.error(f"Error writing to log file {file_path}: {e}")

    def log_custom_event(self, event_type: str, data: Dict[str, Any]):
        """Record a custom analytics event"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "event": event_type,
            "data": data,
        }
        self._append_to_jsonl(self.custom_events_file, entry)
        return entry

    def log_preference_vote(self, vote: Dict[str, Any]) -> Dict[str, Any]:
        """Record an Emotion Eval preference vote."""
        self._append_to_jsonl(self.preference_votes_file, vote)
        winner = vote.get("winner")
        if winner in ("a", "b"):
            self.vote_counts[winner] += 1
            self.vote_counts["total"] += 1
        return self.get_preference_stats()

    def get_preference_stats(self) -> Dict[str, Any]:
        """Get current preference vote statistics."""
        total = self.vote_counts["total"]
        return {
            "total": total,
            "a_wins": self.vote_counts["a"],
            "b_wins": self.vote_counts["b"],
            "a_percent": round(self.vote_counts["a"] / total * 100, 2) if total else 0.0,
            "b_percent": round(self.vote_counts["b"] / total * 100, 2) if total else 0.0,
        }
    
    def get_real_time_stats(self) -> Dict[str, Any]:
        """Get real-time analytics summary"""
        
        # Calculate stats from recent requests
        recent_count = len(self.recent_requests)
        if recent_count == 0:
            return {"status": "no_data", "message": "No recent requests"}
        
        # Handler distribution
        handler_counts = defaultdict(int)
        persona_counts = defaultdict(int)
        recent_latencies = []
        
        for req in self.recent_requests:
            handler = req.get("routing", {}).get("handler", "unknown")
            persona = req.get("persona", "unknown")
            handler_counts[handler] += 1
            persona_counts[persona] += 1
        
        # Get latencies from handler stats
        for handler, stats in self.handler_stats.items():
            if stats["count"] > 0:
                recent_latencies.append(stats["avg_latency"])
        
        avg_latency = sum(recent_latencies) / len(recent_latencies) if recent_latencies else 0.0
        
        return {
            "timestamp": datetime.now().isoformat(),
            "recent_requests": recent_count,
            "handler_distribution": dict(handler_counts),
            "persona_distribution": dict(persona_counts),
            "performance": {
                "avg_latency_seconds": round(avg_latency, 3),
                "total_handlers": len(self.handler_stats),
                "active_sessions": self._count_active_sessions()
            },
            "handler_details": {
                handler: {
                    "requests": stats["count"],
                    "avg_latency": round(stats["avg_latency"], 3),
                    "success_rate": round(stats["success_count"] / stats["count"], 3) if stats["count"] > 0 else 0.0,
                    "total_tokens": stats["total_tokens"]
                }
                for handler, stats in self.handler_stats.items()
                if stats["count"] > 0
            }
        }
    
    def _count_active_sessions(self) -> int:
        """Count unique session IDs in recent requests"""
        recent_sessions = set()
        cutoff_time = datetime.now() - timedelta(hours=1)  # Last hour
        
        for req in self.recent_requests:
            try:
                req_time = datetime.fromisoformat(req["timestamp"])
                if req_time > cutoff_time:
                    session_id = req.get("context", {}).get("session_id")
                    if session_id:
                        recent_sessions.add(session_id)
            except:
                continue
        
        return len(recent_sessions)
    
    def get_daily_analytics(self, days_back: int = 7) -> Dict[str, Any]:
        """Get daily analytics for the past N days"""
        
        analytics = {
            "period": f"last_{days_back}_days",
            "generated_at": datetime.now().isoformat(),
            "daily_breakdown": {}
        }
        
        # Generate daily summaries
        for i in range(days_back):
            date = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
            
            if date in self.daily_stats:
                daily_data = self.daily_stats[date]
                analytics["daily_breakdown"][date] = {
                    "total_requests": daily_data["requests"],
                    "handlers_used": dict(daily_data["handlers_used"]),
                    "error_rate": round(daily_data["error_rate"], 3),
                    "most_used_handler": max(daily_data["handlers_used"].items(), 
                                           key=lambda x: x[1])[0] if daily_data["handlers_used"] else "none"
                }
            else:
                analytics["daily_breakdown"][date] = {
                    "total_requests": 0,
                    "handlers_used": {},
                    "error_rate": 0.0,
                    "most_used_handler": "none"
                }
        
        # Calculate totals
        total_requests = sum(day["total_requests"] for day in analytics["daily_breakdown"].values())
        avg_error_rate = sum(day["error_rate"] for day in analytics["daily_breakdown"].values()) / days_back
        
        analytics["summary"] = {
            "total_requests": total_requests,
            "avg_daily_requests": round(total_requests / days_back, 1),
            "avg_error_rate": round(avg_error_rate, 3),
            "active_days": sum(1 for day in analytics["daily_breakdown"].values() if day["total_requests"] > 0)
        }
        
        return analytics
    
    def get_handler_performance_report(self) -> Dict[str, Any]:
        """Get detailed performance report for each handler"""
        
        report = {
            "generated_at": datetime.now().isoformat(),
            "handlers": {}
        }
        
        for handler, stats in self.handler_stats.items():
            if stats["count"] > 0:
                success_rate = stats["success_count"] / stats["count"]
                error_rate = stats["error_count"] / stats["count"]
                avg_tokens_per_request = stats["total_tokens"] / stats["count"] if stats["count"] > 0 else 0
                
                # Performance rating
                if success_rate >= 0.95 and stats["avg_latency"] < 5.0:
                    performance_rating = "excellent"
                elif success_rate >= 0.90 and stats["avg_latency"] < 10.0:
                    performance_rating = "good"
                elif success_rate >= 0.80:
                    performance_rating = "fair"
                else:
                    performance_rating = "poor"
                
                report["handlers"][handler] = {
                    "total_requests": stats["count"],
                    "success_rate": round(success_rate, 3),
                    "error_rate": round(error_rate, 3),
                    "avg_latency_seconds": round(stats["avg_latency"], 3),
                    "total_tokens": stats["total_tokens"],
                    "avg_tokens_per_request": round(avg_tokens_per_request, 1),
                    "performance_rating": performance_rating
                }
        
        return report
    
    def search_logs(self, query: str, log_type: str = "routing", 
                   hours_back: int = 24, limit: int = 50) -> List[Dict[str, Any]]:
        """Search through logs for specific patterns"""
        
        results = []
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        query_lower = query.lower()
        
        log_file = self.routing_log_file if log_type == "routing" else self.performance_log_file
        
        try:
            if log_file.exists():
                with open(log_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            entry = json.loads(line.strip())
                            entry_time = datetime.fromisoformat(entry["timestamp"])
                            
                            if entry_time > cutoff_time:
                                # Search in various fields
                                searchable_text = json.dumps(entry).lower()
                                if query_lower in searchable_text:
                                    results.append(entry)
                                    if len(results) >= limit:
                                        break
                        except:
                            continue
        except Exception as e:
            logger.error(f"Error searching logs: {e}")
        
        return results
    
    def export_analytics(self, filename: Optional[str] = None) -> str:
        """Export comprehensive analytics to JSON file"""
        
        if filename is None:
            filename = f"analytics_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        export_data = {
            "export_info": {
                "generated_at": datetime.now().isoformat(),
                "system": "Dolphin AI Orchestrator",
                "version": "1.0.0"
            },
            "real_time_stats": self.get_real_time_stats(),
            "daily_analytics": self.get_daily_analytics(),
            "handler_performance": self.get_handler_performance_report(),
            "system_health": {
                "total_requests_processed": sum(stats["count"] for stats in self.handler_stats.values()),
                "avg_system_latency": round(
                    sum(stats["avg_latency"] * stats["count"] for stats in self.handler_stats.values()) /
                    max(sum(stats["count"] for stats in self.handler_stats.values()), 1), 3
                ),
                "overall_success_rate": round(
                    sum(stats["success_count"] for stats in self.handler_stats.values()) /
                    max(sum(stats["count"] for stats in self.handler_stats.values()), 1), 3
                )
            }
        }
        
        export_path = self.logs_dir / filename
        try:
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2)
            
            logger.info(f"Analytics exported to: {export_path}")
            return str(export_path)
            
        except Exception as e:
            logger.error(f"Error exporting analytics: {e}")
            return f"Error: {e}"
    
    def cleanup_old_logs(self, days_to_keep: int = 30) -> Dict[str, int]:
        """Clean up log files older than specified days"""
        
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        cleaned_count = {"routing": 0, "performance": 0}
        
        # Clean routing logs
        cleaned_count["routing"] = self._clean_jsonl_file(self.routing_log_file, cutoff_date)
        
        # Clean performance logs  
        cleaned_count["performance"] = self._clean_jsonl_file(self.performance_log_file, cutoff_date)
        
        logger.info(f"Cleaned {cleaned_count['routing']} routing entries and {cleaned_count['performance']} performance entries")
        return cleaned_count
    
    def _clean_jsonl_file(self, file_path: Path, cutoff_date: datetime) -> int:
        """Clean entries older than cutoff date from a JSONL file"""
        
        if not file_path.exists():
            return 0
        
        kept_lines = []
        removed_count = 0
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        entry_date = datetime.fromisoformat(entry["timestamp"])
                        
                        if entry_date >= cutoff_date:
                            kept_lines.append(line)
                        else:
                            removed_count += 1
                    except:
                        # Keep malformed lines
                        kept_lines.append(line)
            
            # Rewrite file with kept lines
            with open(file_path, 'w', encoding='utf-8') as f:
                f.writelines(kept_lines)
                
        except Exception as e:
            logger.error(f"Error cleaning log file {file_path}: {e}")
        
        return removed_count

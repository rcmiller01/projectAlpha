"""
Visualization System Implementation

Provides data visualization and analytics for the companion system
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import json
import base64

logger = logging.getLogger(__name__)


@dataclass
class VisualizationConfig:
    """Configuration for visualization system"""
    chart_type: str
    data_source: str
    update_frequency: int  # seconds
    color_scheme: str = "default"
    interactive: bool = True
    export_formats: Optional[List[str]] = None


@dataclass
class ChartData:
    """Data structure for chart information"""
    chart_id: str
    chart_type: str
    title: str
    data: Dict[str, Any]
    metadata: Dict[str, Any]
    timestamp: datetime
    

class VisualizationEngine:
    """
    Visualization engine for companion system analytics and user insights
    """
    
    def __init__(self, user_id: str, config: Dict[str, Any]):
        self.user_id = user_id
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{user_id}")
        
        # Visualization state
        self.active_charts = {}
        self.chart_history = []
        self.color_schemes = {
            "default": ["#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6"],
            "emotional": ["#ff6b6b", "#4ecdc4", "#45b7d1", "#96ceb4", "#feca57"],
            "therapeutic": ["#a8e6cf", "#dcedc1", "#ffd3a5", "#ffd1dc", "#c7cedb"]
        }
        
        self.logger.info(f"📊 Visualization engine initialized for user {user_id}")
    
    async def create_emotional_timeline_chart(self, emotional_data: List[Dict[str, Any]]) -> ChartData:
        """
        Create a timeline chart showing emotional state evolution
        """
        try:
            # Process emotional data for visualization
            timestamps = []
            emotions = {}
            
            for entry in emotional_data:
                timestamps.append(entry.get("timestamp", datetime.now().isoformat()))
                emotional_state = entry.get("emotional_state", {})
                
                for emotion, intensity in emotional_state.items():
                    if emotion not in emotions:
                        emotions[emotion] = []
                    emotions[emotion].append(intensity)
            
            # Prepare chart data
            chart_data = {
                "type": "line",
                "data": {
                    "labels": timestamps,
                    "datasets": []
                },
                "options": {
                    "responsive": True,
                    "plugins": {
                        "title": {
                            "display": True,
                            "text": "Emotional State Timeline"
                        }
                    },
                    "scales": {
                        "y": {
                            "beginAtZero": True,
                            "max": 1.0,
                            "title": {
                                "display": True,
                                "text": "Emotional Intensity"
                            }
                        }
                    }
                }
            }
            
            # Add datasets for each emotion
            color_scheme = self.color_schemes.get("emotional", self.color_schemes["default"])
            for i, (emotion, values) in enumerate(emotions.items()):
                color = color_scheme[i % len(color_scheme)]
                chart_data["data"]["datasets"].append({
                    "label": emotion.capitalize(),
                    "data": values,
                    "borderColor": color,
                    "backgroundColor": color + "20",  # Add transparency
                    "tension": 0.3
                })
            
            chart = ChartData(
                chart_id=f"emotional_timeline_{int(datetime.now().timestamp())}",
                chart_type="emotional_timeline",
                title="Emotional State Over Time",
                data=chart_data,
                metadata={
                    "emotion_count": len(emotions),
                    "data_points": len(timestamps),
                    "time_range": f"{timestamps[0]} to {timestamps[-1]}" if timestamps else "No data"
                },
                timestamp=datetime.now()
            )
            
            self.active_charts[chart.chart_id] = chart
            self.logger.info(f"📈 Created emotional timeline chart: {chart.chart_id}")
            
            return chart
            
        except Exception as e:
            self.logger.error(f"❌ Error creating emotional timeline chart: {e}")
            return self._create_error_chart("emotional_timeline", str(e))
    
    async def create_interaction_analysis_chart(self, interaction_data: List[Dict[str, Any]]) -> ChartData:
        """
        Create charts analyzing interaction patterns and types
        """
        try:
            # Analyze interaction types
            interaction_types = {}
            satisfaction_scores = []
            hourly_distribution = {str(i): 0 for i in range(24)}
            
            for interaction in interaction_data:
                # Count interaction types
                int_type = interaction.get("interaction_type", "general")
                interaction_types[int_type] = interaction_types.get(int_type, 0) + 1
                
                # Collect satisfaction scores
                satisfaction = interaction.get("satisfaction_score", 0.5)
                satisfaction_scores.append(satisfaction)
                
                # Analyze time distribution
                timestamp = interaction.get("timestamp", datetime.now().isoformat())
                try:
                    hour = datetime.fromisoformat(timestamp).hour
                    hourly_distribution[str(hour)] += 1
                except:
                    pass
            
            # Create combined dashboard chart
            chart_data = {
                "type": "dashboard",
                "charts": {
                    "interaction_types": {
                        "type": "doughnut",
                        "data": {
                            "labels": list(interaction_types.keys()),
                            "datasets": [{
                                "data": list(interaction_types.values()),
                                "backgroundColor": self.color_schemes["default"][:len(interaction_types)]
                            }]
                        },
                        "options": {
                            "plugins": {
                                "title": {
                                    "display": True,
                                    "text": "Interaction Types Distribution"
                                }
                            }
                        }
                    },
                    "satisfaction_trend": {
                        "type": "line",
                        "data": {
                            "labels": [f"Session {i+1}" for i in range(len(satisfaction_scores))],
                            "datasets": [{
                                "label": "Satisfaction Score",
                                "data": satisfaction_scores,
                                "borderColor": "#2ecc71",
                                "backgroundColor": "#2ecc7130",
                                "tension": 0.3
                            }]
                        },
                        "options": {
                            "plugins": {
                                "title": {
                                    "display": True,
                                    "text": "Satisfaction Trend"
                                }
                            },
                            "scales": {
                                "y": {
                                    "beginAtZero": True,
                                    "max": 1.0
                                }
                            }
                        }
                    },
                    "hourly_distribution": {
                        "type": "bar",
                        "data": {
                            "labels": [f"{i}:00" for i in range(24)],
                            "datasets": [{
                                "label": "Interactions per Hour",
                                "data": [hourly_distribution[str(i)] for i in range(24)],
                                "backgroundColor": "#3498db50",
                                "borderColor": "#3498db"
                            }]
                        },
                        "options": {
                            "plugins": {
                                "title": {
                                    "display": True,
                                    "text": "Interaction Time Distribution"
                                }
                            }
                        }
                    }
                }
            }
            
            chart = ChartData(
                chart_id=f"interaction_analysis_{int(datetime.now().timestamp())}",
                chart_type="interaction_analysis",
                title="Interaction Analysis Dashboard",
                data=chart_data,
                metadata={
                    "total_interactions": len(interaction_data),
                    "unique_types": len(interaction_types),
                    "avg_satisfaction": sum(satisfaction_scores) / len(satisfaction_scores) if satisfaction_scores else 0,
                    "peak_hour": max(hourly_distribution.keys(), key=lambda x: hourly_distribution[x]) if hourly_distribution else "0"
                },
                timestamp=datetime.now()
            )
            
            self.active_charts[chart.chart_id] = chart
            self.logger.info(f"📊 Created interaction analysis chart: {chart.chart_id}")
            
            return chart
            
        except Exception as e:
            self.logger.error(f"❌ Error creating interaction analysis chart: {e}")
            return self._create_error_chart("interaction_analysis", str(e))
    
    async def create_crisis_monitoring_chart(self, crisis_data: List[Dict[str, Any]]) -> ChartData:
        """
        Create visualization for crisis monitoring and safety metrics
        """
        try:
            # Process crisis data
            crisis_levels = []
            timestamps = []
            intervention_counts = {"immediate": 0, "follow_up": 0, "resources_provided": 0}
            
            for entry in crisis_data:
                timestamps.append(entry.get("timestamp", datetime.now().isoformat()))
                crisis_levels.append(entry.get("crisis_level", 0))
                
                # Count intervention types
                if entry.get("immediate_intervention"):
                    intervention_counts["immediate"] += 1
                if entry.get("follow_up_scheduled"):
                    intervention_counts["follow_up"] += 1
                if entry.get("resources_provided"):
                    intervention_counts["resources_provided"] += 1
            
            chart_data = {
                "type": "crisis_dashboard",
                "charts": {
                    "crisis_timeline": {
                        "type": "line",
                        "data": {
                            "labels": timestamps,
                            "datasets": [{
                                "label": "Crisis Level",
                                "data": crisis_levels,
                                "borderColor": "#e74c3c",
                                "backgroundColor": "#e74c3c30",
                                "tension": 0.1
                            }]
                        },
                        "options": {
                            "plugins": {
                                "title": {
                                    "display": True,
                                    "text": "Crisis Level Timeline"
                                }
                            },
                            "scales": {
                                "y": {
                                    "beginAtZero": True,
                                    "max": 3,
                                    "title": {
                                        "display": True,
                                        "text": "Crisis Level (0-3)"
                                    }
                                }
                            }
                        }
                    },
                    "intervention_summary": {
                        "type": "bar",
                        "data": {
                            "labels": ["Immediate", "Follow-up", "Resources"],
                            "datasets": [{
                                "label": "Intervention Count",
                                "data": [
                                    intervention_counts["immediate"],
                                    intervention_counts["follow_up"],
                                    intervention_counts["resources_provided"]
                                ],
                                "backgroundColor": ["#e74c3c", "#f39c12", "#2ecc71"]
                            }]
                        },
                        "options": {
                            "plugins": {
                                "title": {
                                    "display": True,
                                    "text": "Crisis Interventions Summary"
                                }
                            }
                        }
                    }
                }
            }
            
            chart = ChartData(
                chart_id=f"crisis_monitoring_{int(datetime.now().timestamp())}",
                chart_type="crisis_monitoring",
                title="Crisis Monitoring Dashboard",
                data=chart_data,
                metadata={
                    "total_crisis_events": len(crisis_data),
                    "max_crisis_level": max(crisis_levels) if crisis_levels else 0,
                    "total_interventions": sum(intervention_counts.values()),
                    "avg_crisis_level": sum(crisis_levels) / len(crisis_levels) if crisis_levels else 0
                },
                timestamp=datetime.now()
            )
            
            self.active_charts[chart.chart_id] = chart
            self.logger.info(f"🚨 Created crisis monitoring chart: {chart.chart_id}")
            
            return chart
            
        except Exception as e:
            self.logger.error(f"❌ Error creating crisis monitoring chart: {e}")
            return self._create_error_chart("crisis_monitoring", str(e))
    
    def _create_error_chart(self, chart_type: str, error_message: str) -> ChartData:
        """Create an error chart when visualization fails"""
        return ChartData(
            chart_id=f"error_{chart_type}_{int(datetime.now().timestamp())}",
            chart_type="error",
            title=f"Error: {chart_type}",
            data={"error": error_message},
            metadata={"error": True, "error_message": error_message},
            timestamp=datetime.now()
        )
    
    async def export_chart(self, chart_id: str, format: str = "json") -> Optional[str]:
        """
        Export chart data in specified format
        """
        if chart_id not in self.active_charts:
            self.logger.warning(f"⚠️ Chart not found: {chart_id}")
            return None
        
        chart = self.active_charts[chart_id]
        
        try:
            if format == "json":
                return json.dumps({
                    "chart_id": chart.chart_id,
                    "chart_type": chart.chart_type,
                    "title": chart.title,
                    "data": chart.data,
                    "metadata": chart.metadata,
                    "timestamp": chart.timestamp.isoformat()
                }, indent=2)
            
            elif format == "svg":
                # Generate SVG representation (simplified)
                svg_content = f"""
                <svg width="400" height="300" xmlns="http://www.w3.org/2000/svg">
                    <rect width="400" height="300" fill="#f8f9fa" stroke="#dee2e6"/>
                    <text x="200" y="30" text-anchor="middle" font-size="16" font-weight="bold">{chart.title}</text>
                    <text x="200" y="150" text-anchor="middle" font-size="12">Chart data: {len(str(chart.data))} characters</text>
                    <text x="200" y="170" text-anchor="middle" font-size="10">Generated: {chart.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</text>
                </svg>
                """
                return svg_content.strip()
            
            else:
                self.logger.warning(f"⚠️ Unsupported export format: {format}")
                return None
                
        except Exception as e:
            self.logger.error(f"❌ Error exporting chart {chart_id}: {e}")
            return None
    
    async def get_visualization_summary(self) -> Dict[str, Any]:
        """Get summary of visualization engine status"""
        return {
            "active_charts": len(self.active_charts),
            "chart_types": list(set(chart.chart_type for chart in self.active_charts.values())),
            "total_charts_created": len(self.chart_history) + len(self.active_charts),
            "available_color_schemes": list(self.color_schemes.keys()),
            "supported_export_formats": ["json", "svg"]
        }
    
    async def cleanup_old_charts(self, max_age_hours: int = 24) -> int:
        """Clean up old charts to manage memory"""
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        charts_to_remove = []
        
        for chart_id, chart in self.active_charts.items():
            if chart.timestamp < cutoff_time:
                charts_to_remove.append(chart_id)
                self.chart_history.append(chart)
        
        for chart_id in charts_to_remove:
            del self.active_charts[chart_id]
        
        self.logger.info(f"🧹 Cleaned up {len(charts_to_remove)} old charts")
        return len(charts_to_remove)


# Factory function for visualization system
async def create_visualization_engine(user_id: str, config: Dict[str, Any]) -> VisualizationEngine:
    """Create and initialize visualization engine"""
    engine = VisualizationEngine(user_id, config)
    logger.info(f"📊 Visualization engine created for user {user_id}")
    return engine

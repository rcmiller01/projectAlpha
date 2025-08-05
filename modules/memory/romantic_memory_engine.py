# romantic_memory_engine.py
# Enhanced romantic memory engine with relationship feedback loops

import json
import time
import threading
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import numpy as np
from collections import defaultdict

class MemoryType(Enum):
    EMOTIONAL_MOMENT = "emotional_moment"
    RELATIONSHIP_MILESTONE = "relationship_milestone"
    INTIMATE_INTERACTION = "intimate_interaction"
    CONVERSATION_HIGHLIGHT = "conversation_highlight"
    SHARED_ACTIVITY = "shared_activity"
    CONFLICT_RESOLUTION = "conflict_resolution"
    GROWTH_MOMENT = "growth_moment"
    ROMANTIC_GESTURE = "romantic_gesture"

class EmotionalIntensity(Enum):
    SUBTLE = 0.2
    MILD = 0.4
    MODERATE = 0.6
    STRONG = 0.8
    INTENSE = 1.0

@dataclass
class RomanticMemory:
    id: str
    memory_type: MemoryType
    title: str
    description: str
    emotional_intensity: float
    emotions: List[str]
    personas_involved: List[str]
    timestamp: datetime
    context: Dict[str, Any]
    relationship_impact: float  # -1.0 to 1.0
    tags: List[str]
    importance_score: float  # 0.0 to 1.0
    last_recalled: Optional[datetime] = None
    recall_count: int = 0
    decay_rate: float = 0.1

@dataclass
class RelationshipPattern:
    pattern_type: str
    frequency: int
    average_intensity: float
    last_occurrence: datetime
    trend: str  # "increasing", "decreasing", "stable"
    impact_score: float

class RomanticMemoryEngine:
    def __init__(self):
        self.memories = []
        self.patterns = {}
        self.relationship_metrics = defaultdict(list)
        self.emotional_trends = {}
        self.feedback_loops = {}
        self.memory_decay_rates = self._load_decay_rates()
        self.pattern_detection_rules = self._load_pattern_rules()
        
        # Start background processing
        self.is_running = True
        threading.Thread(target=self._background_processing, daemon=True).start()
    
    def _load_decay_rates(self) -> Dict[MemoryType, float]:
        """Load memory decay rates by type"""
        return {
            MemoryType.EMOTIONAL_MOMENT: 0.05,  # Slow decay for emotional moments
            MemoryType.RELATIONSHIP_MILESTONE: 0.02,  # Very slow decay for milestones
            MemoryType.INTIMATE_INTERACTION: 0.08,  # Moderate decay
            MemoryType.CONVERSATION_HIGHLIGHT: 0.1,  # Faster decay
            MemoryType.SHARED_ACTIVITY: 0.06,  # Moderate decay
            MemoryType.CONFLICT_RESOLUTION: 0.04,  # Slow decay for resolutions
            MemoryType.GROWTH_MOMENT: 0.03,  # Very slow decay for growth
            MemoryType.ROMANTIC_GESTURE: 0.07  # Moderate decay
        }
    
    def _load_pattern_rules(self) -> Dict[str, Dict]:
        """Load pattern detection rules"""
        return {
            "emotional_escalation": {
                "description": "Detect when emotional intensity increases over time",
                "threshold": 0.3,
                "time_window": timedelta(days=7),
                "min_occurrences": 3
            },
            "conflict_patterns": {
                "description": "Detect recurring conflict themes",
                "threshold": 0.5,
                "time_window": timedelta(days=30),
                "min_occurrences": 2
            },
            "intimacy_growth": {
                "description": "Detect increasing intimacy levels",
                "threshold": 0.2,
                "time_window": timedelta(days=14),
                "min_occurrences": 5
            },
            "communication_improvement": {
                "description": "Detect improving communication patterns",
                "threshold": 0.25,
                "time_window": timedelta(days=21),
                "min_occurrences": 4
            },
            "romantic_consistency": {
                "description": "Detect consistent romantic behavior",
                "threshold": 0.4,
                "time_window": timedelta(days=10),
                "min_occurrences": 6
            }
        }
    
    def store_romantic_memory(self, memory_type: str, title: str, description: str,
                            emotional_intensity: float, emotions: List[str],
                            personas_involved: List[str], context: Dict[str, Any],
                            relationship_impact: float = 0.0, tags: List[str] = None) -> str:
        """Store a new romantic memory"""
        try:
            memory_id = f"romantic_{int(time.time())}_{len(self.memories)}"
            
            # Calculate importance score
            importance_score = self._calculate_importance_score(
                emotional_intensity, relationship_impact, memory_type
            )
            
            # Create memory object
            memory = RomanticMemory(
                id=memory_id,
                memory_type=MemoryType(memory_type),
                title=title,
                description=description,
                emotional_intensity=emotional_intensity,
                emotions=emotions,
                personas_involved=personas_involved,
                timestamp=datetime.now(),
                context=context,
                relationship_impact=relationship_impact,
                tags=tags or [],
                importance_score=importance_score,
                decay_rate=self.memory_decay_rates.get(MemoryType(memory_type), 0.1)
            )
            
            # Store memory
            self.memories.append(memory)
            
            # Update patterns
            self._update_patterns(memory)
            
            # Update relationship metrics
            self._update_relationship_metrics(memory)
            
            print(f"[RomanticMemory] Stored memory: {title} (importance: {importance_score:.2f})")
            
            return memory_id
            
        except Exception as e:
            print(f"[RomanticMemory] Error storing memory: {e}")
            return None
    
    def _calculate_importance_score(self, emotional_intensity: float, 
                                  relationship_impact: float, memory_type: str) -> float:
        """Calculate memory importance score"""
        # Base score from emotional intensity
        base_score = emotional_intensity * 0.4
        
        # Impact from relationship effect
        impact_score = abs(relationship_impact) * 0.3
        
        # Type-specific importance
        type_importance = {
            "relationship_milestone": 0.8,
            "emotional_moment": 0.6,
            "intimate_interaction": 0.7,
            "conflict_resolution": 0.6,
            "growth_moment": 0.7,
            "romantic_gesture": 0.5,
            "conversation_highlight": 0.4,
            "shared_activity": 0.3
        }
        
        type_score = type_importance.get(memory_type, 0.5) * 0.3
        
        return min(1.0, base_score + impact_score + type_score)
    
    def _update_patterns(self, memory: RomanticMemory):
        """Update pattern detection based on new memory"""
        # Update emotional patterns
        self._update_emotional_patterns(memory)
        
        # Update relationship patterns
        self._update_relationship_patterns(memory)
        
        # Update interaction patterns
        self._update_interaction_patterns(memory)
    
    def _update_emotional_patterns(self, memory: RomanticMemory):
        """Update emotional pattern detection"""
        for emotion in memory.emotions:
            if emotion not in self.emotional_trends:
                self.emotional_trends[emotion] = []
            
            self.emotional_trends[emotion].append({
                "intensity": memory.emotional_intensity,
                "timestamp": memory.timestamp,
                "memory_id": memory.id
            })
    
    def _update_relationship_patterns(self, memory: RomanticMemory):
        """Update relationship pattern detection"""
        # Track relationship impact over time
        self.relationship_metrics["impact"].append({
            "value": memory.relationship_impact,
            "timestamp": memory.timestamp,
            "memory_id": memory.id
        })
        
        # Track emotional intensity trends
        self.relationship_metrics["emotional_intensity"].append({
            "value": memory.emotional_intensity,
            "timestamp": memory.timestamp,
            "memory_id": memory.id
        })
    
    def _update_interaction_patterns(self, memory: RomanticMemory):
        """Update interaction pattern detection"""
        # Track persona interaction patterns
        for persona in memory.personas_involved:
            if persona not in self.relationship_metrics:
                self.relationship_metrics[persona] = []
            
            self.relationship_metrics[persona].append({
                "memory_type": memory.memory_type.value,
                "intensity": memory.emotional_intensity,
                "timestamp": memory.timestamp,
                "memory_id": memory.id
            })
    
    def recall_memories(self, emotion: str = None, persona: str = None, 
                       memory_type: str = None, limit: int = 10) -> List[RomanticMemory]:
        """Recall memories based on criteria"""
        try:
            # Apply memory decay
            self._apply_memory_decay()
            
            # Filter memories
            filtered_memories = self.memories.copy()
            
            if emotion:
                filtered_memories = [
                    m for m in filtered_memories 
                    if emotion in m.emotions
                ]
            
            if persona:
                filtered_memories = [
                    m for m in filtered_memories 
                    if persona in m.personas_involved
                ]
            
            if memory_type:
                filtered_memories = [
                    m for m in filtered_memories 
                    if m.memory_type.value == memory_type
                ]
            
            # Sort by importance and recency
            filtered_memories.sort(
                key=lambda x: (x.importance_score, x.timestamp),
                reverse=True
            )
            
            # Update recall statistics
            for memory in filtered_memories[:limit]:
                memory.last_recalled = datetime.now()
                memory.recall_count += 1
            
            return filtered_memories[:limit]
            
        except Exception as e:
            print(f"[RomanticMemory] Error recalling memories: {e}")
            return []
    
    def _apply_memory_decay(self):
        """Apply decay to memory importance scores"""
        current_time = datetime.now()
        
        for memory in self.memories:
            if memory.last_recalled:
                time_since_recall = (current_time - memory.last_recalled).days
                decay_factor = memory.decay_rate * time_since_recall
                memory.importance_score = max(0.1, memory.importance_score - decay_factor)
    
    def detect_relationship_patterns(self) -> Dict[str, RelationshipPattern]:
        """Detect patterns in relationship data"""
        patterns = {}
        
        try:
            # Detect emotional escalation patterns
            patterns["emotional_escalation"] = self._detect_emotional_escalation()
            
            # Detect conflict patterns
            patterns["conflict_patterns"] = self._detect_conflict_patterns()
            
            # Detect intimacy growth
            patterns["intimacy_growth"] = self._detect_intimacy_growth()
            
            # Detect communication improvement
            patterns["communication_improvement"] = self._detect_communication_improvement()
            
            # Detect romantic consistency
            patterns["romantic_consistency"] = self._detect_romantic_consistency()
            
            self.patterns = patterns
            
        except Exception as e:
            print(f"[RomanticMemory] Error detecting patterns: {e}")
        
        return patterns
    
    def _detect_emotional_escalation(self) -> Optional[RelationshipPattern]:
        """Detect emotional escalation patterns"""
        # Analyze emotional intensity trends
        intensity_data = self.relationship_metrics.get("emotional_intensity", [])
        
        if len(intensity_data) < 3:
            return None
        
        # Calculate trend
        recent_intensities = [d["value"] for d in intensity_data[-10:]]
        trend = self._calculate_trend(recent_intensities)
        
        if trend > 0.3:  # Significant increase
            return RelationshipPattern(
                pattern_type="emotional_escalation",
                frequency=len(recent_intensities),
                average_intensity=np.mean(recent_intensities),
                last_occurrence=datetime.now(),
                trend="increasing",
                impact_score=trend
            )
        
        return None
    
    def _detect_conflict_patterns(self) -> Optional[RelationshipPattern]:
        """Detect conflict resolution patterns"""
        conflict_memories = [
            m for m in self.memories 
            if m.memory_type == MemoryType.CONFLICT_RESOLUTION
        ]
        
        if len(conflict_memories) < 2:
            return None
        
        # Analyze conflict resolution effectiveness
        resolutions = [m for m in conflict_memories if m.relationship_impact > 0]
        conflicts = [m for m in conflict_memories if m.relationship_impact < 0]
        
        if len(resolutions) > len(conflicts):
            return RelationshipPattern(
                pattern_type="conflict_resolution",
                frequency=len(conflict_memories),
                average_intensity=np.mean([m.emotional_intensity for m in conflict_memories]),
                last_occurrence=conflict_memories[-1].timestamp,
                trend="improving",
                impact_score=len(resolutions) / len(conflict_memories)
            )
        
        return None
    
    def _detect_intimacy_growth(self) -> Optional[RelationshipPattern]:
        """Detect intimacy growth patterns"""
        intimate_memories = [
            m for m in self.memories 
            if m.memory_type in [MemoryType.INTIMATE_INTERACTION, MemoryType.ROMANTIC_GESTURE]
        ]
        
        if len(intimate_memories) < 5:
            return None
        
        # Analyze intimacy intensity over time
        recent_intimate = intimate_memories[-10:]
        intensities = [m.emotional_intensity for m in recent_intimate]
        trend = self._calculate_trend(intensities)
        
        if trend > 0.2:
            return RelationshipPattern(
                pattern_type="intimacy_growth",
                frequency=len(recent_intimate),
                average_intensity=np.mean(intensities),
                last_occurrence=recent_intimate[-1].timestamp,
                trend="increasing",
                impact_score=trend
            )
        
        return None
    
    def _detect_communication_improvement(self) -> Optional[RelationshipPattern]:
        """Detect communication improvement patterns"""
        communication_memories = [
            m for m in self.memories 
            if m.memory_type == MemoryType.CONVERSATION_HIGHLIGHT
        ]
        
        if len(communication_memories) < 4:
            return None
        
        # Analyze communication quality
        recent_comm = communication_memories[-8:]
        qualities = [m.relationship_impact for m in recent_comm]
        trend = self._calculate_trend(qualities)
        
        if trend > 0.25:
            return RelationshipPattern(
                pattern_type="communication_improvement",
                frequency=len(recent_comm),
                average_intensity=np.mean([m.emotional_intensity for m in recent_comm]),
                last_occurrence=recent_comm[-1].timestamp,
                trend="improving",
                impact_score=trend
            )
        
        return None
    
    def _detect_romantic_consistency(self) -> Optional[RelationshipPattern]:
        """Detect romantic consistency patterns"""
        romantic_memories = [
            m for m in self.memories 
            if m.memory_type == MemoryType.ROMANTIC_GESTURE
        ]
        
        if len(romantic_memories) < 6:
            return None
        
        # Analyze consistency over time
        recent_romantic = romantic_memories[-12:]
        time_gaps = []
        
        for i in range(1, len(recent_romantic)):
            gap = (recent_romantic[i].timestamp - recent_romantic[i-1].timestamp).days
            time_gaps.append(gap)
        
        if time_gaps:
            consistency_score = 1.0 / (np.std(time_gaps) + 1)  # Lower std = higher consistency
            
            if consistency_score > 0.4:
                return RelationshipPattern(
                    pattern_type="romantic_consistency",
                    frequency=len(recent_romantic),
                    average_intensity=np.mean([m.emotional_intensity for m in recent_romantic]),
                    last_occurrence=recent_romantic[-1].timestamp,
                    trend="stable",
                    impact_score=consistency_score
                )
        
        return None
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend in a list of values"""
        if len(values) < 2:
            return 0.0
        
        # Simple linear trend calculation
        x = np.arange(len(values))
        y = np.array(values)
        
        # Calculate slope
        slope = np.polyfit(x, y, 1)[0]
        
        # Normalize by range
        value_range = max(values) - min(values) if max(values) != min(values) else 1.0
        normalized_trend = slope / value_range
        
        return normalized_trend
    
    def generate_relationship_insights(self) -> Dict[str, Any]:
        """Generate comprehensive relationship insights"""
        try:
            # Detect patterns
            patterns = self.detect_relationship_patterns()
            
            # Calculate relationship health metrics
            health_metrics = self._calculate_health_metrics()
            
            # Generate recommendations
            recommendations = self._generate_recommendations(patterns, health_metrics)
            
            # Analyze emotional trends
            emotional_analysis = self._analyze_emotional_trends()
            
            return {
                "patterns": {k: asdict(v) for k, v in patterns.items() if v is not None},
                "health_metrics": health_metrics,
                "recommendations": recommendations,
                "emotional_analysis": emotional_analysis,
                "memory_stats": {
                    "total_memories": len(self.memories),
                    "memory_types": self._get_memory_type_distribution(),
                    "average_importance": np.mean([m.importance_score for m in self.memories]) if self.memories else 0.0,
                    "recent_activity": len([m for m in self.memories if (datetime.now() - m.timestamp).days < 7])
                }
            }
            
        except Exception as e:
            print(f"[RomanticMemory] Error generating insights: {e}")
            return {}
    
    def _calculate_health_metrics(self) -> Dict[str, float]:
        """Calculate relationship health metrics"""
        if not self.memories:
            return {}
        
        recent_memories = [m for m in self.memories if (datetime.now() - m.timestamp).days < 30]
        
        if not recent_memories:
            return {}
        
        metrics = {
            "emotional_intensity": np.mean([m.emotional_intensity for m in recent_memories]),
            "relationship_impact": np.mean([m.relationship_impact for m in recent_memories]),
            "memory_importance": np.mean([m.importance_score for m in recent_memories]),
            "positive_interactions": len([m for m in recent_memories if m.relationship_impact > 0]) / len(recent_memories),
            "intimate_moments": len([m for m in recent_memories if m.memory_type in [MemoryType.INTIMATE_INTERACTION, MemoryType.ROMANTIC_GESTURE]]) / len(recent_memories)
        }
        
        return metrics
    
    def _generate_recommendations(self, patterns: Dict, health_metrics: Dict) -> List[str]:
        """Generate relationship recommendations based on patterns and metrics"""
        recommendations = []
        
        # Pattern-based recommendations
        if "emotional_escalation" in patterns and patterns["emotional_escalation"]:
            recommendations.append("Consider discussing the increasing emotional intensity to ensure both partners are comfortable")
        
        if "conflict_patterns" in patterns and patterns["conflict_patterns"]:
            recommendations.append("Focus on conflict resolution skills and consider relationship counseling")
        
        if "intimacy_growth" in patterns and patterns["intimacy_growth"]:
            recommendations.append("Continue nurturing the growing intimacy through shared experiences")
        
        if "communication_improvement" in patterns and patterns["communication_improvement"]:
            recommendations.append("Maintain the improved communication patterns and consider regular check-ins")
        
        # Health-based recommendations
        if health_metrics.get("positive_interactions", 0) < 0.7:
            recommendations.append("Work on increasing positive interactions and reducing negative patterns")
        
        if health_metrics.get("intimate_moments", 0) < 0.3:
            recommendations.append("Consider planning more intimate moments and romantic gestures")
        
        if health_metrics.get("emotional_intensity", 0) < 0.4:
            recommendations.append("Explore ways to deepen emotional connection and intensity")
        
        return recommendations
    
    def _analyze_emotional_trends(self) -> Dict[str, Any]:
        """Analyze emotional trends over time"""
        analysis = {}
        
        for emotion, trend_data in self.emotional_trends.items():
            if len(trend_data) >= 3:
                intensities = [d["intensity"] for d in trend_data[-10:]]
                analysis[emotion] = {
                    "frequency": len(trend_data),
                    "average_intensity": np.mean(intensities),
                    "trend": self._calculate_trend(intensities),
                    "last_occurrence": trend_data[-1]["timestamp"]
                }
        
        return analysis
    
    def _get_memory_type_distribution(self) -> Dict[str, int]:
        """Get distribution of memory types"""
        distribution = defaultdict(int)
        
        for memory in self.memories:
            distribution[memory.memory_type.value] += 1
        
        return dict(distribution)
    
    def _background_processing(self):
        """Background processing for pattern detection and memory management"""
        while self.is_running:
            try:
                # Detect patterns every hour
                self.detect_relationship_patterns()
                
                # Apply memory decay
                self._apply_memory_decay()
                
                # Clean up old memories (keep last 1000)
                if len(self.memories) > 1000:
                    self.memories = sorted(self.memories, key=lambda x: x.importance_score, reverse=True)[:1000]
                
                time.sleep(3600)  # Run every hour
                
            except Exception as e:
                print(f"[RomanticMemory] Background processing error: {e}")
                time.sleep(300)  # Wait 5 minutes on error
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """Get summary of romantic memory system"""
        return {
            "total_memories": len(self.memories),
            "memory_types": self._get_memory_type_distribution(),
            "patterns_detected": len([p for p in self.patterns.values() if p is not None]),
            "average_importance": np.mean([m.importance_score for m in self.memories]) if self.memories else 0.0,
            "recent_memories": len([m for m in self.memories if (datetime.now() - m.timestamp).days < 7]),
            "system_status": "active" if self.is_running else "inactive"
        }

# Global romantic memory engine instance
romantic_memory_engine = RomanticMemoryEngine()

def get_romantic_memory_engine() -> RomanticMemoryEngine:
    """Get the global romantic memory engine instance"""
    return romantic_memory_engine

def store_romantic_memory(memory_type: str, title: str, description: str,
                         emotional_intensity: float, emotions: List[str],
                         personas_involved: List[str], context: Dict[str, Any] = None,
                         relationship_impact: float = 0.0, tags: List[str] = None) -> str:
    """Store romantic memory with convenience function"""
    return romantic_memory_engine.store_romantic_memory(
        memory_type, title, description, emotional_intensity, emotions,
        personas_involved, context or {}, relationship_impact, tags
    )

def recall_romantic_memories(emotion: str = None, persona: str = None,
                           memory_type: str = None, limit: int = 10) -> List[Dict]:
    """Recall romantic memories with convenience function"""
    memories = romantic_memory_engine.recall_memories(emotion, persona, memory_type, limit)
    return [asdict(memory) for memory in memories]

def get_relationship_insights() -> Dict[str, Any]:
    """Get relationship insights with convenience function"""
    return romantic_memory_engine.generate_relationship_insights() 
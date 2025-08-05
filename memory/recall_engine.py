"""
True Recall - Main Memory Interface

This module provides the primary interface for reading and writing structured memories
in the True Recall system. It coordinates between all memory subsystems to provide
a unified API for memory operations.
"""

import asyncio
import logging
import json
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict

from .memory_graph import MemoryGraph
from .salience_scoring import SalienceScorer
from .emotion_tagger import EmotionTagger
from .reflection_agent import ReflectionAgent
from .storage.memory_store import MemoryStore

logger = logging.getLogger(__name__)

@dataclass
class MemoryEvent:
    """Represents a single memory event in the True Recall system."""
    id: str
    timestamp: str
    actor: str  # "user" or "dolphin" or "system"
    event_type: str  # "decision", "observation", "emotion", "question", "reflection"
    content: str
    tone: str
    emotion_tags: List[str]
    salience: float
    related_ids: List[str]
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class TrueRecallEngine:
    """
    Main interface for the True Recall memory system.
    
    Coordinates between memory graph, emotion tagging, salience scoring,
    and reflection generation to provide comprehensive memory management
    for emotionally intelligent AI agents.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the True Recall engine."""
        self.config = config or {}
        
        # Initialize subsystems
        storage_path = self.config.get('storage_path', 'memory_data')
        storage_config = self.config.get('storage_config', {})
        self.memory_store = MemoryStore(storage_path, storage_config)
        self.memory_graph = MemoryGraph(self.memory_store)
        self.salience_scorer = SalienceScorer(self.config.get('salience', {}))
        self.emotion_tagger = EmotionTagger(self.config.get('emotion', {}))
        self.reflection_agent = ReflectionAgent(
            self.memory_graph, 
            self.config.get('reflection', {})
        )
        
        # System state
        self.event_counter = 0
        self.last_reflection_date = None
        
        logger.info("🧠 True Recall engine initialized")
    
    async def record_event(self, 
                          actor: str,
                          event_type: str,
                          content: str,
                          related_context: Optional[List[str]] = None,
                          metadata: Optional[Dict[str, Any]] = None) -> MemoryEvent:
        """
        Record a new memory event with full processing.
        
        Args:
            actor: Who performed the action ("user", "dolphin", "system")
            event_type: Type of event ("decision", "observation", "emotion", "question", "reflection")
            content: The content/description of the event
            related_context: Optional list of related event IDs
            metadata: Additional metadata for the event
            
        Returns:
            The created MemoryEvent object
        """
        try:
            # Generate unique event ID
            self.event_counter += 1
            timestamp = datetime.now()
            event_id = f"evt_{timestamp.strftime('%Y%m%d')}_{self.event_counter:03d}"
            
            logger.info(f"📝 Recording event: {event_id} - {event_type}")
            
            # Process the content through emotion and salience analysis
            emotion_analysis = await self.emotion_tagger.analyze_emotion(content, actor)
            salience_score = await self.salience_scorer.calculate_salience(
                content, event_type, actor, emotion_analysis
            )
            
            # Find related events if context is provided
            related_ids = []
            if related_context:
                related_events = await self.memory_graph.find_related_events(
                    content, related_context, limit=5
                )
                related_ids = [event.id for event in related_events]
            
            # Create the memory event
            memory_event = MemoryEvent(
                id=event_id,
                timestamp=timestamp.isoformat(),
                actor=actor,
                event_type=event_type,
                content=content,
                tone=emotion_analysis.get('tone', 'neutral'),
                emotion_tags=emotion_analysis.get('emotions', []),
                salience=salience_score,
                related_ids=related_ids,
                metadata=metadata or {}
            )
            
            # Store the event
            await self.memory_graph.add_event(memory_event)
            
            # Check if daily reflection is needed
            await self._check_daily_reflection()
            
            logger.info(f"✅ Event recorded: {event_id} (salience: {salience_score:.3f})")
            return memory_event
            
        except Exception as e:
            logger.error(f"❌ Failed to record event: {e}")
            raise
    
    async def recall_events(self,
                           query: str = None,
                           actor: str = None,
                           event_type: str = None,
                           emotion_tags: List[str] = None,
                           time_range: tuple = None,
                           min_salience: float = 0.0,
                           limit: int = 20) -> List[MemoryEvent]:
        """
        Recall events based on various criteria.
        
        Args:
            query: Text-based search query
            actor: Filter by actor ("user", "dolphin", "system")
            event_type: Filter by event type
            emotion_tags: Filter by emotion tags
            time_range: Tuple of (start_datetime, end_datetime)
            min_salience: Minimum salience threshold
            limit: Maximum number of events to return
            
        Returns:
            List of matching MemoryEvent objects, sorted by relevance/recency
        """
        try:
            logger.info(f"🔍 Recalling events with query: '{query}' actor: {actor}")
            
            events = await self.memory_graph.search_events(
                query=query,
                actor=actor,
                event_type=event_type,
                emotion_tags=emotion_tags,
                time_range=time_range,
                min_salience=min_salience,
                limit=limit
            )
            
            logger.info(f"📚 Found {len(events)} matching events")
            return events
            
        except Exception as e:
            logger.error(f"❌ Failed to recall events: {e}")
            return []
    
    async def recall_context(self, 
                           reference_event_id: str,
                           context_window: int = 5,
                           time_window_hours: int = 24) -> Dict[str, Any]:
        """
        Recall contextual events around a reference event.
        
        Args:
            reference_event_id: The reference event ID
            context_window: Number of events before/after to include
            time_window_hours: Time window in hours to search
            
        Returns:
            Dict containing reference event and contextual events
        """
        try:
            logger.info(f"🔄 Recalling context for event: {reference_event_id}")
            
            reference_event = await self.memory_graph.get_event(reference_event_id)
            if not reference_event:
                return {'reference_event': None, 'context': []}
            
            # Get temporal context
            ref_time = datetime.fromisoformat(reference_event.timestamp)
            time_start = ref_time - timedelta(hours=time_window_hours)
            time_end = ref_time + timedelta(hours=time_window_hours)
            
            context_events = await self.memory_graph.get_temporal_context(
                reference_event, time_start, time_end, context_window
            )
            
            # Get related events
            related_events = await self.memory_graph.get_related_events(
                reference_event_id, depth=2
            )
            
            return {
                'reference_event': reference_event,
                'temporal_context': context_events,
                'related_events': related_events,
                'context_summary': await self._generate_context_summary(
                    reference_event, context_events, related_events
                )
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to recall context: {e}")
            return {'reference_event': None, 'context': []}
    
    async def get_emotional_timeline(self,
                                   days: int = 7,
                                   actor: str = None) -> Dict[str, Any]:
        """
        Generate an emotional timeline showing mood patterns over time.
        
        Args:
            days: Number of days to analyze
            actor: Optional actor filter
            
        Returns:
            Dict containing emotional timeline data
        """
        try:
            logger.info(f"📈 Generating emotional timeline for {days} days")
            
            end_time = datetime.now()
            start_time = end_time - timedelta(days=days)
            
            events = await self.recall_events(
                actor=actor,
                time_range=(start_time, end_time),
                limit=1000
            )
            
            timeline = await self.emotion_tagger.generate_emotional_timeline(
                events, start_time, end_time
            )
            
            return timeline
            
        except Exception as e:
            logger.error(f"❌ Failed to generate emotional timeline: {e}")
            return {}
    
    async def reflect_on_day(self, target_date: datetime = None) -> Dict[str, Any]:
        """
        Generate a daily reflection summary for a specific date.
        
        Args:
            target_date: The date to reflect on (defaults to yesterday)
            
        Returns:
            Dict containing the reflection summary
        """
        try:
            if target_date is None:
                target_date = datetime.now().date() - timedelta(days=1)
            elif isinstance(target_date, datetime):
                target_date = target_date.date()
            
            logger.info(f"🤔 Generating daily reflection for {target_date}")
            
            reflection = await self.reflection_agent.generate_daily_reflection(target_date)
            
            # Record the reflection as a system event
            if reflection:
                await self.record_event(
                    actor="system",
                    event_type="reflection",
                    content=reflection['summary'],
                    metadata={
                        'reflection_type': 'daily',
                        'target_date': target_date.isoformat(),
                        'key_themes': reflection.get('key_themes', []),
                        'emotional_summary': reflection.get('emotional_summary', {}),
                        'salience_stats': reflection.get('salience_stats', {})
                    }
                )
            
            return reflection
            
        except Exception as e:
            logger.error(f"❌ Failed to generate daily reflection: {e}")
            return {}
    
    async def get_identity_continuity(self, days: int = 30) -> Dict[str, Any]:
        """
        Analyze identity continuity and self-awareness patterns.
        
        Args:
            days: Number of days to analyze for continuity
            
        Returns:
            Dict containing identity continuity analysis
        """
        try:
            logger.info(f"🔮 Analyzing identity continuity over {days} days")
            
            end_time = datetime.now()
            start_time = end_time - timedelta(days=days)
            
            # Get all events in the time range
            events = await self.recall_events(
                time_range=(start_time, end_time),
                limit=2000
            )
            
            # Analyze patterns
            continuity_analysis = {
                'time_range': {
                    'start': start_time.isoformat(),
                    'end': end_time.isoformat(),
                    'days': days
                },
                'event_statistics': self._analyze_event_patterns(events),
                'emotional_patterns': await self._analyze_emotional_patterns(events),
                'decision_patterns': self._analyze_decision_patterns(events),
                'identity_themes': await self._extract_identity_themes(events),
                'continuity_score': self._calculate_continuity_score(events)
            }
            
            return continuity_analysis
            
        except Exception as e:
            logger.error(f"❌ Failed to analyze identity continuity: {e}")
            return {}
    
    async def _check_daily_reflection(self):
        """Check if a daily reflection is needed and generate one if so."""
        today = datetime.now().date()
        yesterday = today - timedelta(days=1)
        
        if (self.last_reflection_date is None or 
            self.last_reflection_date < yesterday):
            
            # Check if we have events from yesterday to reflect on
            events = await self.recall_events(
                time_range=(
                    datetime.combine(yesterday, datetime.min.time()),
                    datetime.combine(today, datetime.min.time())
                ),
                limit=100
            )
            
            if events:
                await self.reflect_on_day(yesterday)
                self.last_reflection_date = yesterday
    
    async def _generate_context_summary(self, 
                                      reference_event: MemoryEvent,
                                      context_events: List[MemoryEvent],
                                      related_events: List[MemoryEvent]) -> str:
        """Generate a summary of the context around an event."""
        summary_parts = []
        
        # Reference event summary
        summary_parts.append(f"Reference: {reference_event.content}")
        
        # Temporal context
        if context_events:
            summary_parts.append(f"Temporal context includes {len(context_events)} nearby events")
            
        # Related events
        if related_events:
            summary_parts.append(f"Related to {len(related_events)} other events")
            
        return "; ".join(summary_parts)
    
    def _analyze_event_patterns(self, events: List[MemoryEvent]) -> Dict[str, Any]:
        """Analyze patterns in event data."""
        if not events:
            return {}
        
        # Count by type and actor
        event_types = {}
        actors = {}
        emotions = {}
        
        total_salience = 0
        
        for event in events:
            # Event types
            event_types[event.event_type] = event_types.get(event.event_type, 0) + 1
            
            # Actors
            actors[event.actor] = actors.get(event.actor, 0) + 1
            
            # Emotions
            for emotion in event.emotion_tags:
                emotions[emotion] = emotions.get(emotion, 0) + 1
            
            # Salience
            total_salience += event.salience
        
        return {
            'total_events': len(events),
            'average_salience': total_salience / len(events),
            'event_types': event_types,
            'actors': actors,
            'top_emotions': dict(sorted(emotions.items(), key=lambda x: x[1], reverse=True)[:10])
        }
    
    async def _analyze_emotional_patterns(self, events: List[MemoryEvent]) -> Dict[str, Any]:
        """Analyze emotional patterns in the events."""
        return await self.emotion_tagger.analyze_emotional_patterns(events)
    
    def _analyze_decision_patterns(self, events: List[MemoryEvent]) -> Dict[str, Any]:
        """Analyze decision-making patterns."""
        decision_events = [e for e in events if e.event_type == 'decision']
        
        if not decision_events:
            return {'decision_count': 0}
        
        # Analyze decision characteristics
        high_salience_decisions = [e for e in decision_events if e.salience > 0.7]
        emotional_decisions = [e for e in decision_events if len(e.emotion_tags) > 0]
        
        return {
            'decision_count': len(decision_events),
            'high_salience_decisions': len(high_salience_decisions),
            'emotional_decisions': len(emotional_decisions),
            'average_decision_salience': sum(e.salience for e in decision_events) / len(decision_events)
        }
    
    async def _extract_identity_themes(self, events: List[MemoryEvent]) -> List[str]:
        """Extract recurring themes that relate to identity."""
        # This would use more sophisticated NLP in a full implementation
        # For now, we'll use simple keyword analysis
        
        identity_keywords = [
            'identity', 'self', 'personality', 'character', 'values', 'beliefs',
            'purpose', 'goals', 'dreams', 'fears', 'strengths', 'weaknesses'
        ]
        
        themes = []
        for keyword in identity_keywords:
            count = sum(1 for event in events if keyword in event.content.lower())
            if count > 0:
                themes.append(f"{keyword} ({count} mentions)")
        
        return themes[:10]  # Top 10 themes
    
    def _calculate_continuity_score(self, events: List[MemoryEvent]) -> float:
        """Calculate a continuity score based on event patterns."""
        if not events:
            return 0.0
        
        # Simple continuity score based on:
        # - Consistency of actor participation
        # - Emotional stability
        # - Decision coherence
        
        actor_consistency = len(set(e.actor for e in events)) / len(events)
        emotional_variety = len(set(tag for e in events for tag in e.emotion_tags))
        decision_count = len([e for e in events if e.event_type == 'decision'])
        
        # Normalize and combine factors
        continuity_score = (
            (1 - actor_consistency) * 0.3 +  # Lower variety = higher continuity
            min(emotional_variety / 20, 1.0) * 0.3 +  # Some emotional variety is good
            min(decision_count / len(events) * 2, 1.0) * 0.4  # Active decision making
        )
        
        return min(continuity_score, 1.0)
    
    async def export_memory_snapshot(self, 
                                   start_date: datetime = None,
                                   end_date: datetime = None,
                                   format: str = 'json') -> Dict[str, Any]:
        """Export a snapshot of memories for backup or analysis."""
        try:
            if end_date is None:
                end_date = datetime.now()
            if start_date is None:
                start_date = end_date - timedelta(days=30)
            
            events = await self.recall_events(
                time_range=(start_date, end_date),
                limit=10000
            )
            
            snapshot = {
                'export_timestamp': datetime.now().isoformat(),
                'time_range': {
                    'start': start_date.isoformat(),
                    'end': end_date.isoformat()
                },
                'event_count': len(events),
                'events': [asdict(event) for event in events],
                'system_info': {
                    'engine_version': '1.0.0',
                    'export_format': format
                }
            }
            
            logger.info(f"📤 Exported memory snapshot: {len(events)} events")
            return snapshot
            
        except Exception as e:
            logger.error(f"❌ Failed to export memory snapshot: {e}")
            return {}
    
    async def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics."""
        try:
            # Get basic stats
            stats = await self.memory_store.get_storage_stats()
            total_events = stats.get('total_events', 0)
            storage_stats = await self.memory_store.get_storage_stats()
            
            # Get recent activity
            recent_events = await self.recall_events(
                time_range=(datetime.now() - timedelta(days=7), datetime.now()),
                limit=1000
            )
            
            return {
                'system_info': {
                    'total_events': total_events,
                    'storage_stats': storage_stats,
                    'last_reflection': self.last_reflection_date.isoformat() if self.last_reflection_date else None
                },
                'recent_activity': self._analyze_event_patterns(recent_events),
                'memory_graph_stats': await self.memory_graph.get_graph_stats(),
                'emotion_stats': await self.emotion_tagger.get_emotion_stats(),
                'salience_stats': await self.salience_scorer.get_scoring_stats()
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get system stats: {e}")
            return {}

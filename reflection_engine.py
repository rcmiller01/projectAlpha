"""
🔄 Reflection & Background RAG Engine
Periodically reviews session history and generates symbolic/semantic summaries
with self-reflection capabilities for the Dolphin AI Orchestrator v2.0
"""

from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import aiohttp
from collections import defaultdict, deque

@dataclass
class ReflectionEntry:
    """Represents a reflective insight about the user or conversation"""
    timestamp: datetime
    reflection_type: str  # "pattern", "mood_shift", "topic_drift", "silence", "engagement"
    content: str
    salience_score: float  # 0.0-1.0 importance rating
    affect_score: float   # -1.0 to 1.0 emotional valence
    context_window: Dict[str, Any]  # Messages that triggered this reflection
    tags: List[str]
    session_id: str
    
    def to_dict(self) -> Dict:
        return {
            **asdict(self),
            'timestamp': self.timestamp.isoformat()
        }

class ReflectionEngine:
    """
    Background processing engine that analyzes conversation patterns
    and generates meaningful reflections about user behavior and engagement
    """
    
    def __init__(self, memory_system, analytics_logger, dolphin_url="http://localhost:8000"):
        self.memory_system = memory_system
        self.analytics_logger = analytics_logger
        self.dolphin_url = dolphin_url
        
        # Configuration
        self.reflection_interval = 600  # 10 minutes
        self.message_threshold = 10     # Reflect after N messages
        self.silence_threshold = 1800   # 30 minutes of silence triggers reflection
        
        # State tracking
        self.last_reflection_time = datetime.now()
        self.message_count_since_reflection = 0
        self.reflection_queue = deque(maxlen=100)
        self.is_running = False
        
        # Pattern detection
        self.conversation_patterns = defaultdict(list)
        self.mood_history = deque(maxlen=50)
        self.topic_shifts = []
        
    async def start_background_reflection(self):
        """Start the background reflection process"""
        self.is_running = True
        print("🔄 Reflection Engine: Starting background analysis...")
        
        while self.is_running:
            try:
                await self._process_reflection_cycle()
                await asyncio.sleep(60)  # Check every minute
            except Exception as e:
                print(f"❌ Reflection Engine Error: {e}")
                await asyncio.sleep(60)
    
    def stop_background_reflection(self):
        """Stop the background reflection process"""
        self.is_running = False
        print("⏹️ Reflection Engine: Stopped")
    
    async def _process_reflection_cycle(self):
        """Main reflection processing cycle"""
        current_time = datetime.now()
        time_since_last = (current_time - self.last_reflection_time).total_seconds()
        
        # Check if we should trigger a reflection
        should_reflect = (
            time_since_last >= self.reflection_interval or
            self.message_count_since_reflection >= self.message_threshold or
            await self._detect_silence_period()
        )
        
        if should_reflect:
            await self._generate_reflections()
            self.last_reflection_time = current_time
            self.message_count_since_reflection = 0
    
    async def _detect_silence_period(self) -> bool:
        """Detect if user has been silent for extended period"""
        try:
            # Get recent session activity from memory system
            recent_sessions = await self._get_recent_sessions()
            
            if not recent_sessions:
                return False
                
            last_activity = max(recent_sessions, key=lambda x: x.get('timestamp', 0))
            last_time = datetime.fromisoformat(last_activity['timestamp'])
            silence_duration = (datetime.now() - last_time).total_seconds()
            
            return silence_duration > self.silence_threshold
            
        except Exception as e:
            print(f"❌ Silence detection error: {e}")
            return False
    
    async def _generate_reflections(self):
        """Generate reflections based on recent conversation history"""
        try:
            # Gather context for reflection
            context = await self._gather_reflection_context()
            
            if not context['messages']:
                return
            
            # Generate different types of reflections
            reflections = []
            
            # Pattern analysis
            pattern_reflection = await self._analyze_conversation_patterns(context)
            if pattern_reflection:
                reflections.append(pattern_reflection)
            
            # Mood shift detection
            mood_reflection = await self._analyze_mood_shifts(context)
            if mood_reflection:
                reflections.append(mood_reflection)
            
            # Engagement analysis
            engagement_reflection = await self._analyze_engagement_levels(context)
            if engagement_reflection:
                reflections.append(engagement_reflection)
            
            # Topic drift analysis
            topic_reflection = await self._analyze_topic_drift(context)
            if topic_reflection:
                reflections.append(topic_reflection)
            
            # Store reflections in memory
            for reflection in reflections:
                await self._store_reflection(reflection)
                self.reflection_queue.append(reflection)
                
            print(f"🔄 Generated {len(reflections)} reflections")
            
        except Exception as e:
            print(f"❌ Reflection generation error: {e}")
    
    async def _gather_reflection_context(self) -> Dict[str, Any]:
        """Gather recent conversation context for analysis"""
        try:
            # Get recent messages from all active sessions
            recent_messages = []
            recent_sessions = await self._get_recent_sessions()
            
            for session in recent_sessions:
                session_messages = await self.memory_system.get_session_context(
                    session.get('session_id', 'default')
                )
                recent_messages.extend(session_messages.get('messages', []))
            
            # Sort by timestamp
            recent_messages.sort(key=lambda x: x.get('timestamp', ''))
            
            # Get analytics data
            analytics = await self._get_recent_analytics()
            
            return {
                'messages': recent_messages[-50:],  # Last 50 messages
                'analytics': analytics,
                'timeframe': {
                    'start': (datetime.now() - timedelta(hours=2)).isoformat(),
                    'end': datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            print(f"❌ Context gathering error: {e}")
            return {'messages': [], 'analytics': {}, 'timeframe': {}}
    
    async def _analyze_conversation_patterns(self, context: Dict) -> Optional[ReflectionEntry]:
        """Analyze patterns in conversation flow"""
        messages = context['messages']
        
        if len(messages) < 3:
            return None
        
        # Detect conversation patterns
        patterns = {
            'question_bursts': 0,
            'short_responses': 0,
            'technical_focus': 0,
            'emotional_content': 0
        }
        
        for msg in messages[-10:]:  # Last 10 messages
            content = msg.get('content', '').lower()
            
            if '?' in content:
                patterns['question_bursts'] += 1
            if len(content.split()) < 5:
                patterns['short_responses'] += 1
            if any(tech in content for tech in ['code', 'function', 'bug', 'error', 'debug']):
                patterns['technical_focus'] += 1
            if any(emo in content for emo in ['feel', 'worry', 'excited', 'frustrated', 'happy']):
                patterns['emotional_content'] += 1
        
        # Generate reflection based on dominant pattern
        dominant_pattern = max(patterns, key=patterns.get)
        pattern_score = patterns[dominant_pattern] / len(messages[-10:])
        
        if pattern_score > 0.3:  # 30% threshold
            reflection_content = await self._generate_pattern_insight(dominant_pattern, pattern_score)
            
            return ReflectionEntry(
                timestamp=datetime.now(),
                reflection_type="pattern",
                content=reflection_content,
                salience_score=min(pattern_score * 2, 1.0),
                affect_score=self._calculate_pattern_affect(dominant_pattern),
                context_window={'pattern_type': dominant_pattern, 'score': pattern_score},
                tags=[dominant_pattern, 'conversation_analysis'],
                session_id=messages[-1].get('session_id', 'unknown')
            )
        
        return None
    
    async def _analyze_mood_shifts(self, context: Dict) -> Optional[ReflectionEntry]:
        """Detect significant mood changes in conversation"""
        messages = context['messages']
        
        if len(messages) < 5:
            return None
        
        # Simple sentiment scoring
        mood_scores = []
        for msg in messages[-10:]:
            score = self._simple_sentiment_score(msg.get('content', ''))
            mood_scores.append(score)
        
        # Detect significant shifts
        if len(mood_scores) >= 3:
            early_avg = sum(mood_scores[:3]) / 3
            recent_avg = sum(mood_scores[-3:]) / 3
            shift_magnitude = abs(recent_avg - early_avg)
            
            if shift_magnitude > 0.4:  # Significant mood shift
                shift_direction = "positive" if recent_avg > early_avg else "negative"
                
                return ReflectionEntry(
                    timestamp=datetime.now(),
                    reflection_type="mood_shift",
                    content=f"I noticed a {shift_direction} shift in your energy during our conversation. "
                           f"You started more {'upbeat' if early_avg > 0 else 'reserved'} and moved toward "
                           f"{'brighter' if recent_avg > 0 else 'more serious'} topics.",
                    salience_score=min(shift_magnitude, 1.0),
                    affect_score=recent_avg,
                    context_window={'shift_magnitude': shift_magnitude, 'direction': shift_direction},
                    tags=['mood_shift', 'emotional_analysis'],
                    session_id=messages[-1].get('session_id', 'unknown')
                )
        
        return None
    
    async def _analyze_engagement_levels(self, context: Dict) -> Optional[ReflectionEntry]:
        """Analyze user engagement patterns"""
        messages = context['messages']
        
        if len(messages) < 5:
            return None
        
        # Calculate engagement metrics
        user_messages = [msg for msg in messages if msg.get('role') == 'user']
        
        if len(user_messages) < 3:
            return None
        
        avg_length = sum(len(msg.get('content', '').split()) for msg in user_messages) / len(user_messages)
        response_times = []  # Would need timing data
        
        # Engagement assessment
        engagement_level = "high" if avg_length > 15 else "moderate" if avg_length > 5 else "low"
        
        if engagement_level in ["high", "low"]:
            content = {
                "high": "You've been really engaged in our conversation today - I can feel the energy and thoughtfulness in your messages.",
                "low": "I noticed you've been giving shorter responses. I'm here whenever you want to dive deeper into anything."
            }[engagement_level]
            
            return ReflectionEntry(
                timestamp=datetime.now(),
                reflection_type="engagement",
                content=content,
                salience_score=0.6,
                affect_score=0.3 if engagement_level == "high" else -0.1,
                context_window={'engagement_level': engagement_level, 'avg_length': avg_length},
                tags=['engagement', 'conversation_quality'],
                session_id=user_messages[-1].get('session_id', 'unknown')
            )
        
        return None
    
    async def _analyze_topic_drift(self, context: Dict) -> Optional[ReflectionEntry]:
        """Detect significant changes in conversation topics"""
        messages = context['messages']
        
        if len(messages) < 8:
            return None
        
        # Simple topic detection using keyword clustering
        early_keywords = self._extract_keywords(messages[:len(messages)//2])
        recent_keywords = self._extract_keywords(messages[len(messages)//2:])
        
        overlap = len(early_keywords & recent_keywords) / max(len(early_keywords | recent_keywords), 1)
        
        if overlap < 0.3:  # Low topic overlap indicates drift
            return ReflectionEntry(
                timestamp=datetime.now(),
                reflection_type="topic_drift",
                content="Our conversation has wandered through several different topics. "
                       "I'm following your lead as your interests shift and evolve.",
                salience_score=0.5,
                affect_score=0.1,
                context_window={'topic_overlap': overlap, 'early_topics': list(early_keywords), 'recent_topics': list(recent_keywords)},
                tags=['topic_drift', 'conversation_flow'],
                session_id=messages[-1].get('session_id', 'unknown')
            )
        
        return None
    
    def _extract_keywords(self, messages: List[Dict]) -> set:
        """Extract key topics from messages"""
        import re
        
        text = ' '.join(msg.get('content', '') for msg in messages if msg.get('role') == 'user')
        
        # Simple keyword extraction
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
        
        # Filter out common words
        stopwords = {'that', 'this', 'with', 'have', 'they', 'will', 'been', 'were', 'said', 'each', 'which', 'their', 'time', 'about'}
        keywords = {word for word in words if word not in stopwords}
        
        return keywords
    
    def _simple_sentiment_score(self, text: str) -> float:
        """Simple sentiment scoring (-1 to 1)"""
        positive_words = {'good', 'great', 'awesome', 'love', 'like', 'happy', 'excited', 'wonderful', 'amazing', 'excellent'}
        negative_words = {'bad', 'hate', 'terrible', 'awful', 'sad', 'frustrated', 'angry', 'disappointed', 'worried', 'difficult'}
        
        words = text.lower().split()
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)
        
        total_sentiment_words = positive_count + negative_count
        if total_sentiment_words == 0:
            return 0.0
        
        return (positive_count - negative_count) / max(len(words), 1)
    
    def _calculate_pattern_affect(self, pattern_type: str) -> float:
        """Calculate emotional valence for pattern types"""
        affect_map = {
            'question_bursts': 0.2,    # Curious/engaged
            'short_responses': -0.3,   # Possibly disengaged
            'technical_focus': 0.1,    # Neutral/focused
            'emotional_content': 0.4   # Emotionally open
        }
        return affect_map.get(pattern_type, 0.0)
    
    async def _generate_pattern_insight(self, pattern_type: str, score: float) -> str:
        """Generate human-readable insight about conversation patterns"""
        insights = {
            'question_bursts': f"You've been asking a lot of questions today ({score:.1%} of messages) - I love your curiosity!",
            'short_responses': f"I notice you're giving shorter responses ({score:.1%} brief messages). Sometimes less is more.",
            'technical_focus': f"We've been diving deep into technical topics ({score:.1%} tech-focused). Your problem-solving energy is focused.",
            'emotional_content': f"You've been sharing emotions and feelings ({score:.1%} emotional content). I appreciate your openness."
        }
        return insights.get(pattern_type, f"Interesting conversation pattern detected: {pattern_type}")
    
    async def _store_reflection(self, reflection: ReflectionEntry):
        """Store reflection in memory system with special tagging"""
        try:
            # Add to long-term memory as a special reflection entry
            await self.memory_system.add_longterm_memory(
                category="reflection",
                content=reflection.content,
                metadata={
                    **reflection.to_dict(),
                    'is_reflection': True,
                    'source': 'reflection_engine'
                }
            )
            
            # Log analytics
            self.analytics_logger.log_custom_event(
                "reflection_generated",
                {
                    'reflection_type': reflection.reflection_type,
                    'salience_score': reflection.salience_score,
                    'affect_score': reflection.affect_score
                }
            )
            
        except Exception as e:
            print(f"❌ Failed to store reflection: {e}")
    
    async def _get_recent_sessions(self) -> List[Dict]:
        """Get recent session data"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.dolphin_url}/api/memory/sessions/recent") as response:
                    if response.status == 200:
                        return await response.json()
                    return []
        except Exception as e:
            print(f"❌ Failed to get recent sessions: {e}")
            return []
    
    async def _get_recent_analytics(self) -> Dict:
        """Get recent analytics data"""
        try:
            return self.analytics_logger.get_real_time_stats()
        except Exception as e:
            print(f"❌ Failed to get analytics: {e}")
            return {}
    
    def get_reflection_summary(self) -> Dict[str, Any]:
        """Get summary of recent reflections"""
        recent_reflections = list(self.reflection_queue)[-10:]
        
        return {
            'total_reflections': len(self.reflection_queue),
            'recent_reflections': [r.to_dict() for r in recent_reflections],
            'reflection_types': {
                'pattern': len([r for r in recent_reflections if r.reflection_type == 'pattern']),
                'mood_shift': len([r for r in recent_reflections if r.reflection_type == 'mood_shift']),
                'engagement': len([r for r in recent_reflections if r.reflection_type == 'engagement']),
                'topic_drift': len([r for r in recent_reflections if r.reflection_type == 'topic_drift'])
            },
            'avg_salience': sum(r.salience_score for r in recent_reflections) / max(len(recent_reflections), 1),
            'last_reflection': recent_reflections[-1].to_dict() if recent_reflections else None
        }

# Global reflection engine instance
reflection_engine = None

def get_reflection_engine():
    """Get the global reflection engine instance"""
    return reflection_engine

def initialize_reflection_engine(memory_system, analytics_logger):
    """Initialize the global reflection engine"""
    global reflection_engine
    reflection_engine = ReflectionEngine(memory_system, analytics_logger)
    return reflection_engine

# ----------------------------
# Memory Pattern Reflection Engine
# ----------------------------
import hashlib
import pandas as pd
from textblob import TextBlob
try:
    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer
    _nltk_available = True
except Exception:
    _nltk_available = False

from insight_schema import Insight
import pattern_utils

class MemoryPatternReflectionEngine:
    """Engine that analyzes stored memories during idle cycles."""

    def __init__(self, memory_system, insight_logger=None, persona_instruction_manager=None, sentiment_analysis=None):
        self.memory_system = memory_system
        self.insight_logger = insight_logger
        self.persona_instruction_manager = persona_instruction_manager
        self.sentiment_analysis = sentiment_analysis
        self.last_batch_hash = None
        self._stop = False
        self._sia = SentimentIntensityAnalyzer() if _nltk_available else None

    def stop(self):
        self._stop = True

    def _hash_batch(self, memories: list) -> str:
        ids = [m.get('id') or m.get('memory_id') for m in memories]
        return hashlib.md5(json.dumps(ids, sort_keys=True).encode()).hexdigest()

    def _get_emotion(self, text: str) -> Dict[str, float]:
        if self.sentiment_analysis and hasattr(self.sentiment_analysis, 'get_emotion_vector'):
            return self.sentiment_analysis.get_emotion_vector(text)
        blob = TextBlob(text)
        valence = float(blob.sentiment.polarity)
        arousal = self._sia.polarity_scores(text)['compound'] if self._sia else 0.0
        return {'valence': valence, 'arousal': arousal}

    def run_reflection(self, n: int = 200) -> List[Insight]:
        if self._stop:
            return []
        memories = self.memory_system.get_recent_memories(n=n, filter='salient')
        if not memories:
            return []
        batch_hash = self._hash_batch(memories)
        if batch_hash == self.last_batch_hash:
            print(f"[ReflectionEngine] Skipping reflection for unchanged batch {batch_hash}")
            return []
        self.last_batch_hash = batch_hash

        df = pd.DataFrame(memories)
        if 'text' not in df.columns:
            df['text'] = df.get('content', '')
        df['emotion'] = df['text'].apply(self._get_emotion)
        df['valence'] = df['emotion'].apply(lambda e: e['valence'])
        df['arousal'] = df['emotion'].apply(lambda e: e['arousal'])

        grouped = pattern_utils.group_memories(df.to_dict('records'), ['context', 'tone'])
        insights: List[Insight] = []

        for group_key, group_df in grouped.items():
            avg_valence = group_df['valence'].mean()
            avg_arousal = group_df['arousal'].mean()
            if len(group_df) < 4:
                continue
            insight_type = 'positive_pattern' if avg_valence > 0.3 else 'negative_pattern' if avg_valence < -0.3 else 'neutral_pattern'
            if insight_type == 'negative_pattern' and len(group_df) >= 5:
                insight_type = 'emotional_fatigue'
            intensity = min(abs(avg_valence), 1.0)
            insight = Insight(
                type=insight_type,
                context=str(group_key),
                emotion_vector={'valence': avg_valence, 'arousal': avg_arousal},
                details=f'{len(group_df)} related memories',
                intensity=float(intensity)
            )
            insights.append(insight)

        # Emotional shift over time
        smoothed = pattern_utils.smooth_series(df['valence'], window=5)
        shift = pattern_utils.emotional_shift(list(smoothed))
        if abs(shift) > 0.5:
            shift_type = 'positive_shift' if shift > 0 else 'negative_shift'
            insights.append(
                Insight(
                    type=shift_type,
                    context='overall',
                    emotion_vector={'valence': shift, 'arousal': 0.0},
                    details='Emotion trend over recent memories',
                    intensity=abs(shift)
                )
            )

        # Log and propagate
        for ins in insights:
            if self.insight_logger:
                self.insight_logger.save(ins)
            if self.persona_instruction_manager and ins.intensity > 0.8:
                self.persona_instruction_manager.evolve_persona(ins)

        print(f"[ReflectionEngine] Batch {batch_hash} produced {len(insights)} insights")
        return insights

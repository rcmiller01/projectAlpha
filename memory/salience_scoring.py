"""
True Recall - Salience Scoring

This module assigns importance scores to memory events based on various factors
including emotional weight, frequency patterns, recency, and content characteristics.
"""

import asyncio
import logging
import re
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import math

logger = logging.getLogger(__name__)

class SalienceScorer:
    """
    Calculates salience (importance) scores for memory events.
    
    Uses multiple factors to determine how important an event is:
    - Emotional intensity and valence
    - Recency and frequency patterns
    - Content characteristics (decisions, questions, etc.)
    - Actor significance
    - Temporal clustering
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the salience scorer."""
        self.config = config or {}
        
        # Scoring weights
        self.weights = self.config.get('weights', {
            'emotion_intensity': 0.25,
            'recency': 0.20,
            'content_type': 0.20,
            'actor_significance': 0.15,
            'frequency_pattern': 0.10,
            'temporal_clustering': 0.10
        })
        
        # Content type importance scores
        self.content_type_scores = self.config.get('content_type_scores', {
            'decision': 0.9,
            'emotion': 0.8,
            'question': 0.6,
            'observation': 0.5,
            'reflection': 0.7
        })
        
        # Actor importance scores
        self.actor_scores = self.config.get('actor_scores', {
            'user': 0.8,
            'dolphin': 0.9,
            'system': 0.6
        })
        
        # Emotion intensity mapping
        self.emotion_intensities = self.config.get('emotion_intensities', {
            # High intensity emotions
            'ecstasy': 1.0, 'rage': 1.0, 'terror': 1.0, 'amazement': 1.0,
            'grief': 0.9, 'loathing': 0.9, 'vigilance': 0.9,
            
            # Medium-high intensity
            'joy': 0.8, 'anger': 0.8, 'fear': 0.8, 'surprise': 0.8,
            'sadness': 0.7, 'disgust': 0.7, 'anticipation': 0.7, 'trust': 0.7,
            
            # Medium intensity
            'optimism': 0.6, 'love': 0.6, 'submission': 0.6, 'awe': 0.6,
            'disappointment': 0.6, 'remorse': 0.6, 'contempt': 0.6,
            
            # Lower intensity
            'serenity': 0.4, 'acceptance': 0.4, 'apprehension': 0.4,
            'distraction': 0.4, 'pensiveness': 0.4, 'boredom': 0.4,
            'annoyance': 0.4, 'interest': 0.4,
            
            # Neutral/low
            'neutral': 0.2, 'calm': 0.3, 'content': 0.3
        })
        
        # Scoring statistics for analysis
        self.scoring_stats = {
            'total_scored': 0,
            'average_salience': 0.0,
            'score_distribution': {},
            'last_reset': datetime.now().isoformat()
        }
        
        logger.info("⚖️ Salience Scorer initialized")
    
    async def calculate_salience(self,
                               content: str,
                               event_type: str,
                               actor: str,
                               emotion_analysis: Dict[str, Any],
                               related_events: Optional[List[Any]] = None) -> float:
        """
        Calculate the salience score for a memory event.
        
        Args:
            content: The event content
            event_type: Type of event
            actor: Who performed the action
            emotion_analysis: Emotion analysis results
            related_events: Related events for context
            
        Returns:
            Salience score between 0.0 and 1.0
        """
        try:
            # Initialize component scores
            emotion_score = await self._calculate_emotion_score(emotion_analysis)
            recency_score = self._calculate_recency_score()
            content_score = self._calculate_content_score(content, event_type)
            actor_score = self._calculate_actor_score(actor)
            frequency_score = await self._calculate_frequency_score(content, actor)
            clustering_score = await self._calculate_clustering_score(related_events)
            
            # Weighted combination
            salience = (
                emotion_score * self.weights['emotion_intensity'] +
                recency_score * self.weights['recency'] +
                content_score * self.weights['content_type'] +
                actor_score * self.weights['actor_significance'] +
                frequency_score * self.weights['frequency_pattern'] +
                clustering_score * self.weights['temporal_clustering']
            )
            
            # Apply content-based adjustments
            salience = self._apply_content_adjustments(content, salience)
            
            # Ensure bounds
            salience = max(0.0, min(1.0, salience))
            
            # Update statistics
            self._update_scoring_stats(salience)
            
            logger.debug(f"⚖️ Calculated salience: {salience:.3f} (emotion: {emotion_score:.2f}, content: {content_score:.2f})")
            return salience
            
        except Exception as e:
            logger.error(f"❌ Failed to calculate salience: {e}")
            return 0.5  # Default moderate salience
    
    async def _calculate_emotion_score(self, emotion_analysis: Dict[str, Any]) -> float:
        """Calculate emotion component of salience score."""
        try:
            emotions = emotion_analysis.get('emotions', [])
            tone = emotion_analysis.get('tone', 'neutral')
            
            if not emotions:
                return self.emotion_intensities.get(tone, 0.2)
            
            # Calculate intensity from emotions
            total_intensity = 0.0
            for emotion in emotions:
                intensity = self.emotion_intensities.get(emotion.lower(), 0.3)
                total_intensity += intensity
            
            # Average intensity, but boost for multiple emotions
            avg_intensity = total_intensity / len(emotions)
            emotion_count_boost = min(0.2, len(emotions) * 0.05)  # Slight boost for emotional complexity
            
            return min(1.0, avg_intensity + emotion_count_boost)
            
        except Exception as e:
            logger.error(f"❌ Failed to calculate emotion score: {e}")
            return 0.3
    
    def _calculate_recency_score(self) -> float:
        """Calculate recency component (more recent = higher score)."""
        # Since we're calculating for a new event, it gets maximum recency
        return 1.0
    
    def _calculate_content_score(self, content: str, event_type: str) -> float:
        """Calculate content-based salience score."""
        try:
            # Base score from event type
            base_score = self.content_type_scores.get(event_type, 0.5)
            
            # Content-based adjustments
            content_lower = content.lower()
            
            # Decision/choice indicators
            decision_keywords = ['decide', 'chose', 'will', 'going to', 'plan to', 'commit']
            if any(keyword in content_lower for keyword in decision_keywords):
                base_score += 0.1
            
            # Question indicators
            if '?' in content or any(word in content_lower for word in ['why', 'how', 'what', 'when', 'where', 'who']):
                base_score += 0.05
            
            # Temporal references (past/future focus)
            temporal_keywords = ['remember', 'forget', 'yesterday', 'tomorrow', 'future', 'past', 'history']
            if any(keyword in content_lower for keyword in temporal_keywords):
                base_score += 0.05
            
            # Self-reference (identity-related)
            identity_keywords = ['i am', 'i feel', 'i think', 'i believe', 'my', 'myself']
            if any(keyword in content_lower for keyword in identity_keywords):
                base_score += 0.1
            
            # Goal/aspiration indicators
            goal_keywords = ['goal', 'dream', 'hope', 'want', 'wish', 'aspire', 'achieve']
            if any(keyword in content_lower for keyword in goal_keywords):
                base_score += 0.1
            
            # Learning/insight indicators
            learning_keywords = ['learn', 'understand', 'realize', 'insight', 'discover', 'knowledge']
            if any(keyword in content_lower for keyword in learning_keywords):
                base_score += 0.08
            
            # Problem/challenge indicators
            problem_keywords = ['problem', 'challenge', 'difficult', 'struggle', 'issue', 'conflict']
            if any(keyword in content_lower for keyword in problem_keywords):
                base_score += 0.08
            
            return min(1.0, base_score)
            
        except Exception as e:
            logger.error(f"❌ Failed to calculate content score: {e}")
            return 0.5
    
    def _calculate_actor_score(self, actor: str) -> float:
        """Calculate actor-based salience score."""
        return self.actor_scores.get(actor, 0.5)
    
    async def _calculate_frequency_score(self, content: str, actor: str) -> float:
        """Calculate frequency pattern score (placeholder for now)."""
        # This would analyze frequency patterns in a full implementation
        # For now, return moderate score
        return 0.5
    
    async def _calculate_clustering_score(self, related_events: Optional[List[Any]]) -> float:
        """Calculate temporal clustering score."""
        try:
            if not related_events:
                return 0.3  # Moderate score for isolated events
            
            # Events with many related events get higher scores
            # (part of significant event clusters)
            cluster_size = len(related_events)
            
            if cluster_size == 0:
                return 0.3
            elif cluster_size <= 2:
                return 0.5
            elif cluster_size <= 5:
                return 0.7
            else:
                return 0.9
                
        except Exception as e:
            logger.error(f"❌ Failed to calculate clustering score: {e}")
            return 0.3
    
    def _apply_content_adjustments(self, content: str, base_salience: float) -> float:
        """Apply final content-based adjustments to salience."""
        try:
            content_lower = content.lower()
            adjusted_salience = base_salience
            
            # Boost for first-person expressions
            first_person_count = len(re.findall(r'\bi\s', content_lower))
            if first_person_count > 0:
                adjusted_salience += min(0.1, first_person_count * 0.02)
            
            # Boost for emotional words
            emotional_words = [
                'love', 'hate', 'fear', 'joy', 'sad', 'happy', 'angry', 'excited',
                'worried', 'proud', 'ashamed', 'grateful', 'disappointed', 'hopeful'
            ]
            emotional_count = sum(1 for word in emotional_words if word in content_lower)
            if emotional_count > 0:
                adjusted_salience += min(0.1, emotional_count * 0.03)
            
            # Boost for urgency indicators
            urgency_words = ['urgent', 'immediately', 'asap', 'critical', 'important', 'crucial']
            if any(word in content_lower for word in urgency_words):
                adjusted_salience += 0.15
            
            # Boost for uncertainty (questions, doubts)
            uncertainty_words = ['maybe', 'perhaps', 'uncertain', 'confused', 'not sure', 'wonder']
            if any(word in content_lower for word in uncertainty_words):
                adjusted_salience += 0.08
            
            # Reduce for routine/mundane content
            routine_words = ['usual', 'routine', 'normal', 'typical', 'everyday', 'regular']
            if any(word in content_lower for word in routine_words):
                adjusted_salience -= 0.1
            
            # Content length adjustment (very short or very long content)
            content_length = len(content.split())
            if content_length < 5:  # Very short
                adjusted_salience -= 0.05
            elif content_length > 50:  # Very long
                adjusted_salience += 0.05
            
            return max(0.0, min(1.0, adjusted_salience))
            
        except Exception as e:
            logger.error(f"❌ Failed to apply content adjustments: {e}")
            return base_salience
    
    def _update_scoring_stats(self, salience: float):
        """Update scoring statistics."""
        try:
            self.scoring_stats['total_scored'] += 1
            
            # Update average
            old_avg = self.scoring_stats['average_salience']
            count = self.scoring_stats['total_scored']
            self.scoring_stats['average_salience'] = (old_avg * (count - 1) + salience) / count
            
            # Update distribution
            score_bucket = f"{int(salience * 10) / 10:.1f}"
            if score_bucket not in self.scoring_stats['score_distribution']:
                self.scoring_stats['score_distribution'][score_bucket] = 0
            self.scoring_stats['score_distribution'][score_bucket] += 1
            
        except Exception as e:
            logger.error(f"❌ Failed to update scoring stats: {e}")
    
    async def analyze_salience_patterns(self, events: List[Any]) -> Dict[str, Any]:
        """Analyze salience patterns across a set of events."""
        try:
            if not events:
                return {}
            
            salience_scores = [event.salience for event in events]
            
            # Basic statistics
            avg_salience = sum(salience_scores) / len(salience_scores)
            max_salience = max(salience_scores)
            min_salience = min(salience_scores)
            
            # Distribution analysis
            high_salience = [s for s in salience_scores if s >= 0.7]
            medium_salience = [s for s in salience_scores if 0.4 <= s < 0.7]
            low_salience = [s for s in salience_scores if s < 0.4]
            
            # Event type analysis
            type_salience = {}
            for event in events:
                if event.event_type not in type_salience:
                    type_salience[event.event_type] = []
                type_salience[event.event_type].append(event.salience)
            
            type_averages = {
                event_type: sum(scores) / len(scores)
                for event_type, scores in type_salience.items()
            }
            
            # Actor analysis
            actor_salience = {}
            for event in events:
                if event.actor not in actor_salience:
                    actor_salience[event.actor] = []
                actor_salience[event.actor].append(event.salience)
            
            actor_averages = {
                actor: sum(scores) / len(scores)
                for actor, scores in actor_salience.items()
            }
            
            # Temporal analysis
            temporal_analysis = await self._analyze_temporal_salience(events)
            
            return {
                'summary': {
                    'total_events': len(events),
                    'average_salience': round(avg_salience, 3),
                    'max_salience': round(max_salience, 3),
                    'min_salience': round(min_salience, 3)
                },
                'distribution': {
                    'high_salience': len(high_salience),
                    'medium_salience': len(medium_salience),
                    'low_salience': len(low_salience),
                    'high_percentage': round(len(high_salience) / len(events) * 100, 1)
                },
                'by_event_type': {k: round(v, 3) for k, v in type_averages.items()},
                'by_actor': {k: round(v, 3) for k, v in actor_averages.items()},
                'temporal_patterns': temporal_analysis
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to analyze salience patterns: {e}")
            return {}
    
    async def _analyze_temporal_salience(self, events: List[Any]) -> Dict[str, Any]:
        """Analyze how salience changes over time."""
        try:
            # Sort events by timestamp
            sorted_events = sorted(events, key=lambda e: e.timestamp)
            
            if len(sorted_events) < 2:
                return {}
            
            # Calculate salience trend
            timestamps = [datetime.fromisoformat(event.timestamp) for event in sorted_events]
            salience_scores = [event.salience for event in sorted_events]
            
            # Simple trend calculation (slope of linear regression)
            n = len(timestamps)
            if n < 2:
                return {}
            
            # Convert timestamps to numeric values (hours since first event)
            time_values = [(ts - timestamps[0]).total_seconds() / 3600 for ts in timestamps]
            
            # Calculate linear regression slope
            mean_time = sum(time_values) / n
            mean_salience = sum(salience_scores) / n
            
            numerator = sum((time_values[i] - mean_time) * (salience_scores[i] - mean_salience) for i in range(n))
            denominator = sum((time_values[i] - mean_time) ** 2 for i in range(n))
            
            trend_slope = numerator / denominator if denominator != 0 else 0
            
            # Categorize trend
            if trend_slope > 0.01:
                trend_direction = 'increasing'
            elif trend_slope < -0.01:
                trend_direction = 'decreasing'
            else:
                trend_direction = 'stable'
            
            return {
                'trend_slope': round(trend_slope, 4),
                'trend_direction': trend_direction,
                'time_span_hours': round(time_values[-1], 1),
                'salience_variance': round(sum((s - mean_salience) ** 2 for s in salience_scores) / n, 3)
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to analyze temporal salience: {e}")
            return {}
    
    async def get_scoring_stats(self) -> Dict[str, Any]:
        """Get current scoring statistics."""
        return self.scoring_stats.copy()
    
    def reset_scoring_stats(self):
        """Reset scoring statistics."""
        self.scoring_stats = {
            'total_scored': 0,
            'average_salience': 0.0,
            'score_distribution': {},
            'last_reset': datetime.now().isoformat()
        }
        logger.info("📊 Salience scoring statistics reset")
    
    async def calibrate_weights(self, sample_events: List[Any], target_distribution: Dict[str, float] = None):
        """Calibrate scoring weights based on sample events (advanced feature)."""
        try:
            if not sample_events:
                return
            
            target_distribution = target_distribution or {
                'high': 0.2,  # 20% high salience
                'medium': 0.6,  # 60% medium salience
                'low': 0.2  # 20% low salience
            }
            
            logger.info(f"🎯 Calibrating salience weights with {len(sample_events)} events")
            
            # This would implement weight optimization in a full system
            # For now, just log the calibration attempt
            current_scores = [event.salience for event in sample_events]
            high_count = len([s for s in current_scores if s >= 0.7])
            medium_count = len([s for s in current_scores if 0.4 <= s < 0.7])
            low_count = len([s for s in current_scores if s < 0.4])
            
            total = len(current_scores)
            current_dist = {
                'high': high_count / total,
                'medium': medium_count / total,
                'low': low_count / total
            }
            
            logger.info(f"📊 Current distribution: {current_dist}")
            logger.info(f"🎯 Target distribution: {target_distribution}")
            
        except Exception as e:
            logger.error(f"❌ Failed to calibrate weights: {e}")
    
    def get_salience_explanation(self, 
                               content: str,
                               event_type: str,
                               actor: str,
                               emotion_analysis: Dict[str, Any],
                               final_score: float) -> Dict[str, Any]:
        """Provide explanation for why an event received its salience score."""
        try:
            explanation = {
                'final_score': round(final_score, 3),
                'components': {},
                'adjustments': [],
                'reasoning': []
            }
            
            # Calculate component scores
            emotion_score = self.emotion_intensities.get(emotion_analysis.get('tone', 'neutral'), 0.2)
            content_score = self.content_type_scores.get(event_type, 0.5)
            actor_score = self.actor_scores.get(actor, 0.5)
            
            explanation['components'] = {
                'emotion_intensity': round(emotion_score * self.weights['emotion_intensity'], 3),
                'content_type': round(content_score * self.weights['content_type'], 3),
                'actor_significance': round(actor_score * self.weights['actor_significance'], 3),
                'recency': round(1.0 * self.weights['recency'], 3)  # New events get max recency
            }
            
            # Analyze content for explanations
            content_lower = content.lower()
            
            if any(word in content_lower for word in ['decide', 'chose', 'will']):
                explanation['adjustments'].append('Decision-making content (+0.1)')
            
            if '?' in content:
                explanation['adjustments'].append('Question content (+0.05)')
            
            if any(word in content_lower for word in ['i am', 'i feel', 'i think']):
                explanation['adjustments'].append('Self-reference content (+0.1)')
            
            # Generate reasoning
            if final_score >= 0.8:
                explanation['reasoning'].append('High salience: This event is very significant')
            elif final_score >= 0.6:
                explanation['reasoning'].append('Medium-high salience: This event is quite important')
            elif final_score >= 0.4:
                explanation['reasoning'].append('Medium salience: This event has moderate importance')
            else:
                explanation['reasoning'].append('Lower salience: This event is less significant')
            
            return explanation
            
        except Exception as e:
            logger.error(f"❌ Failed to generate salience explanation: {e}")
            return {'final_score': final_score, 'error': 'Could not generate explanation'}

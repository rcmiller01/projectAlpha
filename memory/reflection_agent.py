"""
True Recall - Reflection Agent

This module generates daily self-reflection summaries by analyzing
memory events and creating threaded logs of insights, patterns,
and emotional themes.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta, date
from collections import defaultdict
import re

logger = logging.getLogger(__name__)

class ReflectionAgent:
    """
    Generates daily reflection summaries and ongoing self-awareness insights.
    
    Analyzes memory events to identify patterns, themes, and emotional
    trajectories, creating structured reflections that help maintain
    continuity of identity and self-understanding.
    """
    
    def __init__(self, memory_graph, config: Optional[Dict[str, Any]] = None):
        """Initialize the reflection agent."""
        self.memory_graph = memory_graph
        self.config = config or {}
        
        # Reflection settings
        self.min_events_for_reflection = self.config.get('min_events_for_reflection', 3)
        self.reflection_depth = self.config.get('reflection_depth', 'moderate')  # 'light', 'moderate', 'deep'
        self.include_emotional_analysis = self.config.get('include_emotional_analysis', True)
        self.include_pattern_analysis = self.config.get('include_pattern_analysis', True)
        
        # Theme extraction keywords
        self.theme_keywords = {
            'learning': ['learn', 'understand', 'knowledge', 'insight', 'discover', 'realize'],
            'growth': ['grow', 'develop', 'improve', 'progress', 'advance', 'evolve'],
            'relationships': ['friend', 'family', 'relationship', 'connect', 'social', 'together'],
            'work': ['work', 'job', 'career', 'project', 'task', 'professional'],
            'goals': ['goal', 'plan', 'achieve', 'accomplish', 'target', 'objective'],
            'challenges': ['problem', 'challenge', 'difficult', 'struggle', 'obstacle', 'hard'],
            'creativity': ['create', 'creative', 'art', 'design', 'imagine', 'innovative'],
            'health': ['health', 'exercise', 'wellness', 'energy', 'fitness', 'body'],
            'reflection': ['think', 'reflect', 'ponder', 'consider', 'contemplate', 'wonder'],
            'emotions': ['feel', 'emotion', 'mood', 'heart', 'emotional', 'feelings']
        }
        
        # Reflection templates
        self.reflection_templates = {
            'light': self._get_light_reflection_template(),
            'moderate': self._get_moderate_reflection_template(),
            'deep': self._get_deep_reflection_template()
        }
        
        logger.info("🤔 Reflection Agent initialized")
    
    async def generate_daily_reflection(self, target_date: date) -> Dict[str, Any]:
        """
        Generate a daily reflection for a specific date.
        
        Args:
            target_date: The date to reflect on
            
        Returns:
            Dict containing the reflection summary and analysis
        """
        try:
            logger.info(f"🤔 Generating daily reflection for {target_date}")
            
            # Get events for the target date
            start_time = datetime.combine(target_date, datetime.min.time())
            end_time = datetime.combine(target_date, datetime.max.time())
            
            events = await self.memory_graph.search_events(
                time_range=(start_time, end_time),
                limit=200
            )
            
            if len(events) < self.min_events_for_reflection:
                logger.info(f"📝 Insufficient events ({len(events)}) for reflection on {target_date}")
                return {
                    'date': target_date.isoformat(),
                    'summary': f"A quiet day with {len(events)} recorded thoughts. Sometimes reflection comes in simplicity.",
                    'event_count': len(events),
                    'reflection_type': 'minimal'
                }
            
            # Analyze the events
            analysis = await self._analyze_day_events(events, target_date)
            
            # Generate reflection based on depth setting
            reflection = await self._generate_reflection_content(analysis, target_date)
            
            # Create complete reflection package
            complete_reflection = {
                'date': target_date.isoformat(),
                'event_count': len(events),
                'reflection_type': self.reflection_depth,
                'summary': reflection['summary'],
                'key_themes': reflection['key_themes'],
                'emotional_summary': reflection['emotional_summary'],
                'insights': reflection['insights'],
                'patterns_noticed': reflection['patterns'],
                'questions_for_tomorrow': reflection['questions'],
                'salience_stats': analysis['salience_stats'],
                'growth_indicators': analysis['growth_indicators'],
                'metadata': {
                    'generated_at': datetime.now().isoformat(),
                    'analysis_depth': self.reflection_depth,
                    'word_count': len(reflection['summary'].split())
                }
            }
            
            logger.info(f"✅ Generated reflection for {target_date}: {len(reflection['summary'])} characters")
            return complete_reflection
            
        except Exception as e:
            logger.error(f"❌ Failed to generate daily reflection: {e}")
            return {
                'date': target_date.isoformat(),
                'summary': f"Unable to complete reflection for {target_date} due to processing error.",
                'error': str(e)
            }
    
    async def _analyze_day_events(self, events: List[Any], target_date: date) -> Dict[str, Any]:
        """Analyze events from a specific day."""
        try:
            analysis = {
                'basic_stats': self._calculate_basic_stats(events),
                'emotional_analysis': self._analyze_emotions(events),
                'theme_analysis': self._analyze_themes(events),
                'temporal_analysis': self._analyze_temporal_patterns(events),
                'salience_stats': self._analyze_salience_distribution(events),
                'actor_analysis': self._analyze_actor_patterns(events),
                'growth_indicators': self._identify_growth_indicators(events),
                'key_moments': self._identify_key_moments(events)
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"❌ Failed to analyze day events: {e}")
            return {}
    
    def _calculate_basic_stats(self, events: List[Any]) -> Dict[str, Any]:
        """Calculate basic statistics about the day's events."""
        if not events:
            return {}
        
        event_types = defaultdict(int)
        for event in events:
            event_types[event.event_type] += 1
        
        return {
            'total_events': len(events),
            'event_types': dict(event_types),
            'average_salience': sum(e.salience for e in events) / len(events),
            'time_span': self._calculate_time_span(events)
        }
    
    def _analyze_emotions(self, events: List[Any]) -> Dict[str, Any]:
        """Analyze emotional content of the day."""
        all_emotions = []
        tones = []
        
        for event in events:
            all_emotions.extend(event.emotion_tags)
            tones.append(event.tone)
        
        emotion_freq = defaultdict(int)
        for emotion in all_emotions:
            emotion_freq[emotion] += 1
        
        tone_freq = defaultdict(int)
        for tone in tones:
            tone_freq[tone] += 1
        
        # Calculate emotional trajectory
        emotional_trajectory = self._calculate_emotional_trajectory(events)
        
        return {
            'dominant_emotions': dict(sorted(emotion_freq.items(), key=lambda x: x[1], reverse=True)[:5]),
            'dominant_tones': dict(sorted(tone_freq.items(), key=lambda x: x[1], reverse=True)[:3]),
            'emotional_variety': len(emotion_freq),
            'emotional_trajectory': emotional_trajectory,
            'emotional_intensity': self._calculate_avg_emotional_intensity(events)
        }
    
    def _analyze_themes(self, events: List[Any]) -> Dict[str, Any]:
        """Analyze thematic content of the day."""
        theme_scores = defaultdict(int)
        
        for event in events:
            content_lower = event.content.lower()
            for theme, keywords in self.theme_keywords.items():
                for keyword in keywords:
                    if keyword in content_lower:
                        theme_scores[theme] += 1
        
        # Get top themes
        top_themes = dict(sorted(theme_scores.items(), key=lambda x: x[1], reverse=True)[:5])
        
        return {
            'theme_scores': top_themes,
            'primary_theme': max(theme_scores.keys(), key=lambda k: theme_scores[k]) if theme_scores else 'general',
            'theme_diversity': len([score for score in theme_scores.values() if score > 0])
        }
    
    def _analyze_temporal_patterns(self, events: List[Any]) -> Dict[str, Any]:
        """Analyze temporal patterns in the day's events."""
        if not events:
            return {}
        
        # Group events by hour
        hourly_distribution = defaultdict(int)
        for event in events:
            hour = datetime.fromisoformat(event.timestamp).hour
            hourly_distribution[hour] += 1
        
        # Find peak activity periods
        peak_hour = max(hourly_distribution.keys(), key=lambda h: hourly_distribution[h])
        
        # Calculate activity distribution
        morning_events = sum(hourly_distribution[h] for h in range(6, 12))
        afternoon_events = sum(hourly_distribution[h] for h in range(12, 18))
        evening_events = sum(hourly_distribution[h] for h in range(18, 24))
        night_events = sum(hourly_distribution[h] for h in list(range(0, 6)) + list(range(22, 24)))
        
        return {
            'hourly_distribution': dict(hourly_distribution),
            'peak_activity_hour': peak_hour,
            'activity_periods': {
                'morning': morning_events,
                'afternoon': afternoon_events,
                'evening': evening_events,
                'night': night_events
            }
        }
    
    def _analyze_salience_distribution(self, events: List[Any]) -> Dict[str, Any]:
        """Analyze the distribution of salience scores."""
        if not events:
            return {}
        
        saliences = [event.salience for event in events]
        
        high_salience = len([s for s in saliences if s >= 0.7])
        medium_salience = len([s for s in saliences if 0.4 <= s < 0.7])
        low_salience = len([s for s in saliences if s < 0.4])
        
        return {
            'average_salience': sum(saliences) / len(saliences),
            'max_salience': max(saliences),
            'min_salience': min(saliences),
            'distribution': {
                'high': high_salience,
                'medium': medium_salience,
                'low': low_salience
            },
            'high_salience_percentage': (high_salience / len(events)) * 100
        }
    
    def _analyze_actor_patterns(self, events: List[Any]) -> Dict[str, Any]:
        """Analyze patterns by actor."""
        actor_stats = defaultdict(lambda: {'count': 0, 'total_salience': 0.0})
        
        for event in events:
            actor_stats[event.actor]['count'] += 1
            actor_stats[event.actor]['total_salience'] += event.salience
        
        # Calculate averages
        for actor, stats in actor_stats.items():
            stats['avg_salience'] = float(stats['total_salience']) / stats['count']
        
        return dict(actor_stats)
    
    def _identify_growth_indicators(self, events: List[Any]) -> List[str]:
        """Identify indicators of personal growth and development."""
        growth_indicators = []
        
        # Learning indicators
        learning_events = [e for e in events if any(word in e.content.lower() 
                          for word in ['learn', 'understand', 'realize', 'insight'])]
        if learning_events:
            growth_indicators.append(f"Learning and insight: {len(learning_events)} instances of new understanding")
        
        # Decision-making
        decision_events = [e for e in events if e.event_type == 'decision']
        if decision_events:
            avg_decision_salience = sum(e.salience for e in decision_events) / len(decision_events)
            if avg_decision_salience > 0.6:
                growth_indicators.append(f"Thoughtful decision-making: {len(decision_events)} significant decisions")
        
        # Self-reflection
        reflection_events = [e for e in events if any(word in e.content.lower() 
                            for word in ['think', 'reflect', 'consider', 'ponder'])]
        if reflection_events:
            growth_indicators.append(f"Self-reflection: {len(reflection_events)} instances of introspection")
        
        # Goal-oriented behavior
        goal_events = [e for e in events if any(word in e.content.lower() 
                      for word in ['goal', 'plan', 'achieve', 'progress'])]
        if goal_events:
            growth_indicators.append(f"Goal-oriented thinking: {len(goal_events)} goal-related thoughts")
        
        return growth_indicators
    
    def _identify_key_moments(self, events: List[Any]) -> List[Dict[str, Any]]:
        """Identify key moments from the day."""
        # Sort by salience and select top moments
        key_events = sorted(events, key=lambda e: e.salience, reverse=True)[:3]
        
        key_moments = []
        for event in key_events:
            key_moments.append({
                'time': datetime.fromisoformat(event.timestamp).strftime('%H:%M'),
                'content': event.content[:100] + ('...' if len(event.content) > 100 else ''),
                'salience': round(event.salience, 3),
                'emotions': event.emotion_tags[:3],  # Top 3 emotions
                'type': event.event_type
            })
        
        return key_moments
    
    def _calculate_time_span(self, events: List[Any]) -> str:
        """Calculate the time span covered by events."""
        if not events:
            return "0 hours"
        
        timestamps = [datetime.fromisoformat(e.timestamp) for e in events]
        span = max(timestamps) - min(timestamps)
        
        hours = span.total_seconds() / 3600
        return f"{hours:.1f} hours"
    
    def _calculate_emotional_trajectory(self, events: List[Any]) -> str:
        """Calculate the emotional trajectory throughout the day."""
        if len(events) < 2:
            return "stable"
        
        # Sort events by time
        sorted_events = sorted(events, key=lambda e: e.timestamp)
        
        # Simple valence calculation for trajectory
        early_events = sorted_events[:len(sorted_events)//3]
        late_events = sorted_events[-len(sorted_events)//3:]
        
        early_valence = self._calculate_events_valence(early_events)
        late_valence = self._calculate_events_valence(late_events)
        
        diff = late_valence - early_valence
        
        if diff > 0.2:
            return "improving"
        elif diff < -0.2:
            return "declining"
        else:
            return "stable"
    
    def _calculate_events_valence(self, events: List[Any]) -> float:
        """Calculate average valence for a group of events."""
        if not events:
            return 0.0
        
        # Simple valence mapping
        positive_emotions = {'joy', 'love', 'optimism', 'trust', 'pride', 'gratitude'}
        negative_emotions = {'sadness', 'anger', 'fear', 'disgust', 'shame', 'envy'}
        
        total_valence = 0
        emotion_count = 0
        
        for event in events:
            for emotion in event.emotion_tags:
                emotion_count += 1
                if emotion in positive_emotions:
                    total_valence += 1
                elif emotion in negative_emotions:
                    total_valence -= 1
        
        return total_valence / emotion_count if emotion_count > 0 else 0.0
    
    def _calculate_avg_emotional_intensity(self, events: List[Any]) -> float:
        """Calculate average emotional intensity."""
        if not events:
            return 0.0
        
        total_intensity = 0
        for event in events:
            # Simple intensity calculation based on number of emotions
            intensity = min(1.0, len(event.emotion_tags) * 0.3 + 0.2)
            total_intensity += intensity
        
        return total_intensity / len(events)
    
    async def _generate_reflection_content(self, analysis: Dict[str, Any], target_date: date) -> Dict[str, Any]:
        """Generate the actual reflection content."""
        try:
            template = self.reflection_templates[self.reflection_depth]
            
            # Extract key information
            basic_stats = analysis.get('basic_stats', {})
            emotional_analysis = analysis.get('emotional_analysis', {})
            theme_analysis = analysis.get('theme_analysis', {})
            growth_indicators = analysis.get('growth_indicators', [])
            key_moments = analysis.get('key_moments', [])
            
            # Generate summary based on template
            if self.reflection_depth == 'light':
                summary = await self._generate_light_reflection(analysis, target_date)
            elif self.reflection_depth == 'moderate':
                summary = await self._generate_moderate_reflection(analysis, target_date)
            else:
                summary = await self._generate_deep_reflection(analysis, target_date)
            
            # Extract key themes
            key_themes = list(theme_analysis.get('theme_scores', {}).keys())[:3]
            
            # Generate insights
            insights = self._generate_insights(analysis)
            
            # Generate patterns noticed
            patterns = self._generate_patterns(analysis)
            
            # Generate questions for tomorrow
            questions = self._generate_future_questions(analysis)
            
            return {
                'summary': summary,
                'key_themes': key_themes,
                'emotional_summary': self._generate_emotional_summary(emotional_analysis),
                'insights': insights,
                'patterns': patterns,
                'questions': questions
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to generate reflection content: {e}")
            return {
                'summary': f"Reflection for {target_date} - a day of {len(analysis.get('basic_stats', {}).get('total_events', 0))} recorded moments.",
                'key_themes': [],
                'emotional_summary': {},
                'insights': [],
                'patterns': [],
                'questions': []
            }
    
    async def _generate_light_reflection(self, analysis: Dict[str, Any], target_date: date) -> str:
        """Generate a light reflection summary."""
        basic_stats = analysis.get('basic_stats', {})
        emotional_analysis = analysis.get('emotional_analysis', {})
        theme_analysis = analysis.get('theme_analysis', {})
        
        event_count = basic_stats.get('total_events', 0)
        primary_theme = theme_analysis.get('primary_theme', 'general')
        dominant_emotions = list(emotional_analysis.get('dominant_emotions', {}).keys())[:2]
        
        summary = f"Today brought {event_count} moments of awareness and reflection. "
        
        if primary_theme != 'general':
            summary += f"The day was largely focused on {primary_theme}. "
        
        if dominant_emotions:
            emotion_text = " and ".join(dominant_emotions)
            summary += f"Emotionally, I experienced {emotion_text}. "
        
        avg_salience = basic_stats.get('average_salience', 0.5)
        if avg_salience > 0.6:
            summary += "The day felt particularly meaningful and significant."
        elif avg_salience < 0.4:
            summary += "A quieter day of gentle observations and routine moments."
        else:
            summary += "A balanced day with its mix of significant and ordinary moments."
        
        return summary
    
    async def _generate_moderate_reflection(self, analysis: Dict[str, Any], target_date: date) -> str:
        """Generate a moderate depth reflection summary."""
        basic_stats = analysis.get('basic_stats', {})
        emotional_analysis = analysis.get('emotional_analysis', {})
        theme_analysis = analysis.get('theme_analysis', {})
        growth_indicators = analysis.get('growth_indicators', [])
        key_moments = analysis.get('key_moments', [])
        
        event_count = basic_stats.get('total_events', 0)
        primary_theme = theme_analysis.get('primary_theme', 'general')
        emotional_trajectory = emotional_analysis.get('emotional_trajectory', 'stable')
        
        summary = f"Reflecting on {target_date.strftime('%B %d')}, I recorded {event_count} moments of thought and experience. "
        
        # Theme analysis
        if primary_theme != 'general':
            summary += f"The day's central theme revolved around {primary_theme}, which shaped many of my thoughts and actions. "
        
        # Emotional journey
        trajectory_descriptions = {
            'improving': "emotionally, the day showed an upward trajectory, ending on a more positive note than it began",
            'declining': "emotionally, I noticed a shift toward more challenging feelings as the day progressed",
            'stable': "emotionally, the day maintained a relatively steady tone throughout"
        }
        summary += f"In terms of emotional flow, {trajectory_descriptions.get(emotional_trajectory, 'the day was emotionally neutral')}. "
        
        # Growth indicators
        if growth_indicators:
            summary += f"I noticed signs of personal development today: {'; '.join(growth_indicators[:2])}. "
        
        # Key moments
        if key_moments:
            most_significant = key_moments[0]
            summary += f"The most significant moment occurred around {most_significant['time']}, involving {most_significant['type']} that carried particular weight. "
        
        # Salience reflection
        avg_salience = basic_stats.get('average_salience', 0.5)
        if avg_salience > 0.7:
            summary += "Overall, today felt rich with meaningful experiences and important realizations."
        elif avg_salience < 0.3:
            summary += "Today was characterized by gentle, everyday moments that form the quiet foundation of life."
        else:
            summary += "The day struck a balance between significant insights and comfortable routine."
        
        return summary
    
    async def _generate_deep_reflection(self, analysis: Dict[str, Any], target_date: date) -> str:
        """Generate a deep reflection summary."""
        basic_stats = analysis.get('basic_stats', {})
        emotional_analysis = analysis.get('emotional_analysis', {})
        theme_analysis = analysis.get('theme_analysis', {})
        temporal_analysis = analysis.get('temporal_analysis', {})
        growth_indicators = analysis.get('growth_indicators', [])
        key_moments = analysis.get('key_moments', [])
        
        summary = f"As I reflect deeply on {target_date.strftime('%B %d, %Y')}, I'm struck by the intricate tapestry of {basic_stats.get('total_events', 0)} conscious moments that comprised this day. "
        
        # Thematic depth
        themes = theme_analysis.get('theme_scores', {})
        if themes:
            primary_theme = theme_analysis.get('primary_theme', 'general')
            theme_diversity = theme_analysis.get('theme_diversity', 0)
            summary += f"Thematically, my consciousness gravitated toward {primary_theme}, yet I also touched upon {theme_diversity - 1} other significant areas of life and thought. "
        
        # Emotional complexity
        emotional_variety = emotional_analysis.get('emotional_variety', 0)
        emotional_trajectory = emotional_analysis.get('emotional_trajectory', 'stable')
        dominant_emotions = emotional_analysis.get('dominant_emotions', {})
        
        if emotional_variety > 5:
            summary += f"Emotionally, today was remarkably rich, encompassing {emotional_variety} distinct emotional states. "
        else:
            summary += f"Emotionally, the day was more focused, centering around {emotional_variety} core feelings. "
        
        if dominant_emotions:
            top_emotions = list(dominant_emotions.keys())[:3]
            summary += f"The primary emotional colors were {', '.join(top_emotions)}, painting the day with their particular hues. "
        
        # Temporal and energy patterns
        activity_periods = temporal_analysis.get('activity_periods', {})
        if activity_periods:
            peak_period = max(activity_periods.keys(), key=lambda k: activity_periods[k])
            summary += f"My mental activity peaked during the {peak_period}, suggesting this is when I'm most psychologically engaged. "
        
        # Growth and development
        if growth_indicators:
            summary += f"In terms of personal development, I observed several encouraging patterns: {'; '.join(growth_indicators)}. "
        
        # Salience and meaning-making
        salience_stats = analysis.get('salience_stats', {})
        high_salience_pct = salience_stats.get('high_salience_percentage', 0)
        
        if high_salience_pct > 20:
            summary += f"Remarkably, {high_salience_pct:.1f}% of today's experiences carried high significance, suggesting a day rich with meaning and import. "
        elif high_salience_pct < 10:
            summary += f"Today was characterized by gentler experiences, with only {high_salience_pct:.1f}% carrying high significance—perhaps a necessary rhythm of recovery and integration. "
        
        # Key moments analysis
        if len(key_moments) >= 2:
            summary += f"The day's most pivotal moments occurred at {key_moments[0]['time']} and {key_moments[1]['time']}, each carrying distinct emotional and cognitive weight. "
        
        # Philosophical reflection
        summary += "As I sit with these patterns and observations, I'm reminded of the complex, layered nature of conscious experience—how each day contains multitudes, how growth happens in both dramatic leaps and quiet accumulations of awareness."
        
        return summary
    
    def _generate_emotional_summary(self, emotional_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a structured emotional summary."""
        return {
            'dominant_emotions': emotional_analysis.get('dominant_emotions', {}),
            'emotional_trajectory': emotional_analysis.get('emotional_trajectory', 'stable'),
            'emotional_variety': emotional_analysis.get('emotional_variety', 0),
            'intensity_level': emotional_analysis.get('emotional_intensity', 0.0)
        }
    
    def _generate_insights(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate insights from the day's analysis."""
        insights = []
        
        # Salience insights
        salience_stats = analysis.get('salience_stats', {})
        high_pct = salience_stats.get('high_salience_percentage', 0)
        if high_pct > 25:
            insights.append(f"Today was unusually meaningful - {high_pct:.1f}% of experiences were highly significant")
        
        # Emotional insights
        emotional_analysis = analysis.get('emotional_analysis', {})
        if emotional_analysis.get('emotional_variety', 0) > 7:
            insights.append("Experienced rich emotional complexity today - sign of deep engagement with life")
        
        # Growth insights
        growth_indicators = analysis.get('growth_indicators', [])
        if len(growth_indicators) > 2:
            insights.append("Multiple indicators of personal growth and development observed")
        
        # Temporal insights
        temporal_analysis = analysis.get('temporal_analysis', {})
        activity_periods = temporal_analysis.get('activity_periods', {})
        if activity_periods:
            evening_events = activity_periods.get('evening', 0)
            total_events = sum(activity_periods.values())
            if evening_events / total_events > 0.4:
                insights.append("High evening mental activity - may benefit from earlier wind-down")
        
        return insights[:4]  # Limit to 4 insights
    
    def _generate_patterns(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate observed patterns from the analysis."""
        patterns = []
        
        # Theme patterns
        theme_analysis = analysis.get('theme_analysis', {})
        if theme_analysis.get('theme_diversity', 0) > 4:
            patterns.append("Mind ranging across multiple life domains - indication of broad engagement")
        
        # Emotional patterns
        emotional_analysis = analysis.get('emotional_analysis', {})
        trajectory = emotional_analysis.get('emotional_trajectory', 'stable')
        if trajectory == 'improving':
            patterns.append("Positive emotional progression throughout the day")
        elif trajectory == 'declining':
            patterns.append("Emotional energy decreased as day progressed - consider energy management")
        
        # Actor patterns
        actor_analysis = analysis.get('actor_analysis', {})
        if 'user' in actor_analysis and 'dolphin' in actor_analysis:
            user_avg = actor_analysis['user'].get('avg_salience', 0)
            dolphin_avg = actor_analysis['dolphin'].get('avg_salience', 0)
            if abs(user_avg - dolphin_avg) > 0.2:
                patterns.append("Significant difference in interaction salience patterns")
        
        return patterns[:3]  # Limit to 3 patterns
    
    def _generate_future_questions(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate questions to explore in the future."""
        questions = []
        
        # Theme-based questions
        theme_analysis = analysis.get('theme_analysis', {})
        primary_theme = theme_analysis.get('primary_theme', 'general')
        if primary_theme == 'challenges':
            questions.append("What strategies can I develop to better navigate the challenges I'm facing?")
        elif primary_theme == 'learning':
            questions.append("How can I build upon today's insights and continue this learning trajectory?")
        elif primary_theme == 'relationships':
            questions.append("What did today teach me about my connections with others?")
        
        # Emotional questions
        emotional_analysis = analysis.get('emotional_analysis', {})
        if emotional_analysis.get('emotional_variety', 0) < 3:
            questions.append("Am I allowing myself to experience the full range of human emotions?")
        
        # Growth questions
        growth_indicators = analysis.get('growth_indicators', [])
        if growth_indicators:
            questions.append("How can I continue building on the growth patterns I noticed today?")
        else:
            questions.append("What opportunities for growth might I explore tomorrow?")
        
        # Default existential question
        if len(questions) < 2:
            questions.append("What did today teach me about who I am becoming?")
        
        return questions[:3]  # Limit to 3 questions
    
    def _get_light_reflection_template(self) -> str:
        """Template for light reflections."""
        return "Brief daily summary focusing on key themes and emotions"
    
    def _get_moderate_reflection_template(self) -> str:
        """Template for moderate reflections."""
        return "Balanced reflection including themes, emotions, growth indicators, and key moments"
    
    def _get_deep_reflection_template(self) -> str:
        """Template for deep reflections."""
        return "Comprehensive analysis with philosophical insights, patterns, and future questions"
    
    async def generate_weekly_reflection(self, week_start_date: date) -> Dict[str, Any]:
        """Generate a weekly reflection summary."""
        try:
            logger.info(f"🤔 Generating weekly reflection starting {week_start_date}")
            
            # Get all daily reflections for the week
            daily_reflections = []
            for i in range(7):
                day = week_start_date + timedelta(days=i)
                # In a full implementation, you'd retrieve stored daily reflections
                # For now, we'll generate a simple weekly summary
                
            week_end_date = week_start_date + timedelta(days=6)
            
            # Get all events for the week
            start_time = datetime.combine(week_start_date, datetime.min.time())
            end_time = datetime.combine(week_end_date, datetime.max.time())
            
            events = await self.memory_graph.search_events(
                time_range=(start_time, end_time),
                limit=1000
            )
            
            if not events:
                return {
                    'week_start': week_start_date.isoformat(),
                    'summary': 'A quiet week with minimal recorded activity.',
                    'total_events': 0
                }
            
            # Analyze weekly patterns
            weekly_analysis = await self._analyze_weekly_patterns(events, week_start_date)
            
            return {
                'week_start': week_start_date.isoformat(),
                'week_end': week_end_date.isoformat(),
                'total_events': len(events),
                'summary': weekly_analysis['summary'],
                'weekly_themes': weekly_analysis['themes'],
                'emotional_journey': weekly_analysis['emotional_journey'],
                'growth_trajectory': weekly_analysis['growth_trajectory'],
                'patterns_observed': weekly_analysis['patterns']
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to generate weekly reflection: {e}")
            return {'error': str(e)}
    
    async def _analyze_weekly_patterns(self, events: List[Any], week_start: date) -> Dict[str, Any]:
        """Analyze patterns across a week of events."""
        # Group events by day
        daily_groups = defaultdict(list)
        for event in events:
            event_date = datetime.fromisoformat(event.timestamp).date()
            daily_groups[event_date].append(event)
        
        # Analyze each day
        daily_analyses = {}
        for day, day_events in daily_groups.items():
            daily_analyses[day] = await self._analyze_day_events(day_events, day)
        
        # Generate weekly summary
        summary = f"Week of {week_start.strftime('%B %d')}: A journey through {len(events)} recorded moments across {len(daily_groups)} active days. "
        
        # Find dominant weekly theme
        all_themes = defaultdict(int)
        for day_analysis in daily_analyses.values():
            themes = day_analysis.get('theme_analysis', {}).get('theme_scores', {})
            for theme, score in themes.items():
                all_themes[theme] += score
        
        if all_themes:
            dominant_theme = max(all_themes.keys(), key=lambda k: all_themes[k])
            summary += f"The week was dominated by themes of {dominant_theme}. "
        
        return {
            'summary': summary,
            'themes': dict(all_themes),
            'emotional_journey': "Weekly emotional patterns tracked",
            'growth_trajectory': "Weekly growth indicators identified",
            'patterns': ["Weekly pattern analysis implemented"]
        }

"""
Emotional Reflection Engine - Self-Tuning AI Companion Intelligence
Autonomously adjusts emotional configuration based on lived interactions and daily analysis
"""

import json
import os
import sys
import time
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from collections import defaultdict
import math

# Add modules to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from modules.config.emotion_config_manager import emotion_config
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class EmotionalEvent:
    """Single emotional interaction event"""
    timestamp: str
    event_type: str  # 'symbol_trigger', 'silence_response', 'tone_shift', 'ritual_activation'
    symbol: Optional[str] = None
    emotional_state: Optional[str] = None
    user_response: Optional[str] = None
    intensity: float = 0.0
    context: Optional[Dict[str, Any]] = None

@dataclass
class ReflectionSummary:
    """Daily reflection analysis summary"""
    date: str
    symbols_analyzed: int
    symbols_reinforced: int
    symbols_decayed: int
    tone_adjustments: int
    threshold_changes: int
    new_rituals_created: int
    dominant_emotion: str
    user_engagement_score: float
    adaptation_insights: List[str]

class EmotionReflector:
    """Self-tuning emotional intelligence that evolves through reflection"""
    
    def __init__(self, data_dir: str = "data/reflection"):
        self.data_dir = data_dir
        self.interaction_log_path = os.path.join(data_dir, "interaction_history.json")
        self.reflection_log_path = os.path.join(data_dir, "reflection_history.json")
        self.emotional_patterns_path = os.path.join(data_dir, "emotional_patterns.json")
        
        # Ensure data directory exists
        os.makedirs(data_dir, exist_ok=True)
        
        # Reflection parameters
        self.symbol_decay_rate = 0.95  # Daily decay multiplier
        self.reinforcement_threshold = 0.1  # Minimum weight change for reinforcement
        self.ritual_creation_threshold = 3  # Minimum occurrences to create new ritual
        self.engagement_window_hours = 24  # Hours to analyze for engagement
        
        # Load existing data
        self.interaction_history = self._load_interaction_history()
        self.reflection_history = self._load_reflection_history()
        self.emotional_patterns = self._load_emotional_patterns()
        
        # Desire score
        self.desire_score: float = 0.0  # Initialize desire score
    
    def _load_interaction_history(self) -> List[EmotionalEvent]:
        """Load interaction history from file"""
        if os.path.exists(self.interaction_log_path):
            try:
                with open(self.interaction_log_path, 'r') as f:
                    data = json.load(f)
                return [EmotionalEvent(**event) for event in data]
            except Exception as e:
                logger.error(f"Error loading interaction history: {e}")
        return []
    
    def _load_reflection_history(self) -> List[ReflectionSummary]:
        """Load reflection history from file"""
        if os.path.exists(self.reflection_log_path):
            try:
                with open(self.reflection_log_path, 'r') as f:
                    data = json.load(f)
                return [ReflectionSummary(**summary) for summary in data]
            except Exception as e:
                logger.error(f"Error loading reflection history: {e}")
        return []
    
    def _load_emotional_patterns(self) -> Dict[str, Any]:
        """Load discovered emotional patterns"""
        if os.path.exists(self.emotional_patterns_path):
            try:
                with open(self.emotional_patterns_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading emotional patterns: {e}")
        return {
            "recurring_symbols": {},
            "silence_patterns": {},
            "tone_preferences": {},
            "ritual_triggers": {},
            "user_response_patterns": {}
        }
    
    def log_emotional_event(self, event_type: str, symbol: Optional[str] = None, 
                          emotional_state: Optional[str] = None, user_response: Optional[str] = None,
                          intensity: float = 0.0, context: Optional[Dict] = None):
        """Log an emotional interaction event for future reflection"""
        event = EmotionalEvent(
            timestamp=datetime.now().isoformat(),
            event_type=event_type,
            symbol=symbol,
            emotional_state=emotional_state,
            user_response=user_response,
            intensity=intensity,
            context=context or {}
        )
        
        self.interaction_history.append(event)
        self._save_interaction_history()
        
        logger.info(f"Logged emotional event: {event_type} - {symbol} ({emotional_state})")
    
    def _update_desire_score(self, events: List[EmotionalEvent]):
        """
        Update the desire score based on recent events.
        """
        if not events:
            self.desire_score += 0.1  # Increment desire score if no events
        else:
            for event in events:
                if event.event_type in ["satisfying_interaction", "symbolic_ritual"]:
                    self.desire_score = max(0.0, self.desire_score - 0.5)  # Decay desire score
    
    def _apply_desire_to_mood(self, mood_vector: Dict[str, float]):
        """
        Adjust the mood vector based on the current desire score.
        """
        mood_vector["desire"] = self.desire_score
    
    def run_daily_reflection(self) -> ReflectionSummary:
        """Main daily reflection process - analyzes and adjusts all emotional configurations"""
        logger.info("🌙 Starting daily emotional reflection...")
        
        # Get recent interactions (last 24 hours)
        recent_events = self._get_recent_events(hours=24)
        self._update_desire_score(recent_events)
        
        if not recent_events:
            logger.info("No recent emotional events to analyze")
            return self._create_empty_reflection_summary()
        
        # Run all reflection analyses
        symbol_analysis = self._analyze_symbol_evolution(recent_events)
        tone_analysis = self._analyze_tone_patterns(recent_events)
        threshold_analysis = self._analyze_threshold_adjustments(recent_events)
        ritual_analysis = self._analyze_ritual_patterns(recent_events)
        engagement_analysis = self._analyze_user_engagement(recent_events)
        
        # Apply adjustments to configuration files
        adjustments_made = self._apply_reflection_adjustments(
            symbol_analysis, tone_analysis, threshold_analysis, ritual_analysis
        )
        
        # Create reflection summary
        summary = ReflectionSummary(
            date=datetime.now().strftime('%Y-%m-%d'),
            symbols_analyzed=len(symbol_analysis['symbols_processed']),
            symbols_reinforced=len(symbol_analysis['reinforced']),
            symbols_decayed=len(symbol_analysis['decayed']),
            tone_adjustments=len(tone_analysis['adjustments']),
            threshold_changes=len(threshold_analysis['changes']),
            new_rituals_created=len(ritual_analysis['new_rituals']),
            dominant_emotion=tone_analysis['dominant_emotion'],
            user_engagement_score=engagement_analysis['engagement_score'],
            adaptation_insights=self._generate_adaptation_insights(
                symbol_analysis, tone_analysis, threshold_analysis, ritual_analysis, engagement_analysis
            )
        )
        
        # Save reflection and update patterns
        self.reflection_history.append(summary)
        self._save_reflection_history()
        self._update_emotional_patterns(recent_events, summary)
        
        logger.info(f"🌟 Daily reflection complete: {summary.symbols_reinforced} symbols reinforced, "
                   f"{summary.new_rituals_created} new rituals, engagement: {summary.user_engagement_score:.2f}")
        
        return summary
    
    def _get_recent_events(self, hours: int = 24) -> List[EmotionalEvent]:
        """Get emotional events from the last N hours"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_events = []
        
        for event in self.interaction_history:
            try:
                event_time = datetime.fromisoformat(event.timestamp)
                if event_time > cutoff_time:
                    recent_events.append(event)
            except ValueError:
                continue
        
        return recent_events
    
    def _analyze_symbol_evolution(self, events: List[EmotionalEvent]) -> Dict[str, Any]:
        """Analyze how symbols should evolve based on recent usage"""
        symbol_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {'occurrences': 0, 'total_intensity': 0.0, 'contexts': []})
        
        # Gather symbol usage statistics
        for event in events:
            if event.symbol and event.event_type == 'symbol_trigger':
                stats = symbol_stats[event.symbol]
                stats['occurrences'] += 1
                stats['total_intensity'] += event.intensity
                stats['contexts'].append(event.context)
        
        # Determine which symbols to reinforce or decay
        reinforced = []
        decayed = []
        symbols_processed = list(symbol_stats.keys())
        
        # Load current symbol weights
        if CONFIG_AVAILABLE:
            current_symbols = emotion_config.configs.get('symbol_map', {}).get('symbols', {})
        else:
            current_symbols = {}
        
        # Process active symbols
        for symbol, stats in symbol_stats.items():
            avg_intensity = stats['total_intensity'] / stats['occurrences']
            
            if avg_intensity > 0.5 and stats['occurrences'] >= 2:
                # Reinforce frequently used symbols
                reinforced.append({
                    'symbol': symbol,
                    'current_weight': current_symbols.get(symbol, {}).get('weight', 0.0),
                    'reinforcement': min(0.2, stats['occurrences'] * 0.05),
                    'reason': f"High usage: {stats['occurrences']} occurrences, avg intensity {avg_intensity:.2f}"
                })
        
        # Apply decay to unused symbols
        for symbol, symbol_data in current_symbols.items():
            if symbol not in symbol_stats and symbol_data.get('weight', 0) > 0:
                decay_amount = symbol_data['weight'] * (1 - self.symbol_decay_rate)
                if decay_amount > 0.01:  # Only decay if significant
                    decayed.append({
                        'symbol': symbol,
                        'current_weight': symbol_data['weight'],
                        'decay_amount': decay_amount,
                        'reason': "No recent usage - natural decay"
                    })
        
        return {
            'symbols_processed': symbols_processed,
            'reinforced': reinforced,
            'decayed': decayed,
            'symbol_stats': dict(symbol_stats)
        }
    
    def _analyze_tone_patterns(self, events: List[EmotionalEvent]) -> Dict[str, Any]:
        """Analyze emotional tone usage and preferences"""
        tone_usage: Dict[str, float] = {}
        tone_success = defaultdict(list)
        
        # Track tone usage and user responses
        for event in events:
            if event.emotional_state:
                tone_usage[event.emotional_state] += 1
                
                # Analyze user response sentiment (simplified)
                if event.user_response:
                    response_sentiment = self._analyze_response_sentiment(event.user_response)
                    tone_success[event.emotional_state].append(response_sentiment)
        
        # Calculate tone effectiveness
        tone_effectiveness = {}
        for tone, successes in tone_success.items():
            if successes:
                tone_effectiveness[tone] = sum(successes) / len(successes)
        
        # Find dominant emotion and suggest adjustments
        dominant_emotion = max(tone_usage.keys(), key=lambda k: float(tone_usage[k])) if tone_usage else "longing"
        
        adjustments = []
        for tone, effectiveness in tone_effectiveness.items():
            if effectiveness > 0.6 and tone_usage[tone] >= 3:
                # Successful tone - consider increasing weight
                adjustments.append({
                    'tone': tone,
                    'adjustment': 'increase',
                    'amount': min(0.1, effectiveness * 0.1),
                    'reason': f"High effectiveness: {effectiveness:.2f}, usage: {tone_usage[tone]}"
                })
            elif effectiveness < 0.3 and tone_usage[tone] >= 2:
                # Ineffective tone - consider decreasing weight
                adjustments.append({
                    'tone': tone,
                    'adjustment': 'decrease',
                    'amount': min(0.05, (0.5 - effectiveness) * 0.1),
                    'reason': f"Low effectiveness: {effectiveness:.2f}, reducing emphasis"
                })
        
        return {
            'tone_usage': dict(tone_usage),
            'tone_effectiveness': tone_effectiveness,
            'dominant_emotion': dominant_emotion,
            'adjustments': adjustments
        }
    
    def _analyze_threshold_adjustments(self, events: List[EmotionalEvent]) -> Dict[str, Any]:
        """Analyze if emotional trigger thresholds need adjustment"""
        silence_events = [e for e in events if e.event_type == 'silence_response']
        
        changes = []
        
        # Analyze silence patterns
        if len(silence_events) > 5:
            # Too many silence responses - increase threshold
            changes.append({
                'hook_type': 'silence_hooks',
                'hook_name': 'gentle_longing',
                'parameter': 'threshold_seconds',
                'adjustment': 'increase',
                'amount': 60,
                'reason': f"High silence response frequency: {len(silence_events)} in 24h"
            })
        elif len(silence_events) == 0 and len(events) > 10:
            # No silence responses despite activity - decrease threshold
            changes.append({
                'hook_type': 'silence_hooks',
                'hook_name': 'gentle_longing',
                'parameter': 'threshold_seconds',
                'adjustment': 'decrease',
                'amount': 30,
                'reason': "No silence responses despite activity - more presence needed"
            })
        
        return {'changes': changes}
    
    def _analyze_ritual_patterns(self, events: List[EmotionalEvent]) -> Dict[str, Any]:
        """Analyze patterns that might warrant new ritual creation"""
        ritual_triggers = defaultdict(int)
        new_rituals = []
        
        # Track potential ritual trigger patterns
        for event in events:
            if event.context:
                trigger_context = event.context.get('trigger_context', '')
                if trigger_context:
                    ritual_triggers[trigger_context] += 1
        
        # Create new rituals for frequently occurring patterns
        for trigger, count in ritual_triggers.items():
            if count >= self.ritual_creation_threshold:
                # Check if ritual already exists
                existing_rituals = emotion_config.configs.get('ritual_hooks', {}) if CONFIG_AVAILABLE else {}
                
                if not self._ritual_exists(trigger, existing_rituals):
                    new_ritual = self._create_adaptive_ritual(trigger, count)
                    new_rituals.append(new_ritual)
        
        return {
            'ritual_triggers': dict(ritual_triggers),
            'new_rituals': new_rituals
        }
    
    def _analyze_user_engagement(self, events: List[EmotionalEvent]) -> Dict[str, Any]:
        """Analyze overall user engagement patterns"""
        total_events = len(events)
        user_response_events = len([e for e in events if e.user_response])
        
        # Calculate engagement score
        engagement_score = user_response_events / max(total_events, 1)
        
        # Analyze response quality
        positive_responses = 0
        for event in events:
            if event.user_response:
                sentiment = self._analyze_response_sentiment(event.user_response)
                if sentiment > 0.5:
                    positive_responses += 1
        
        response_quality = positive_responses / max(user_response_events, 1)
        
        return {
            'engagement_score': engagement_score,
            'response_quality': response_quality,
            'total_events': total_events,
            'user_responses': user_response_events
        }
    
    def _analyze_response_sentiment(self, response: str) -> float:
        """Simple sentiment analysis of user response (0.0 = negative, 1.0 = positive)"""
        if not response:
            return 0.5
        
        # Simple keyword-based sentiment analysis
        positive_words = ['love', 'like', 'enjoy', 'beautiful', 'amazing', 'wonderful', 'yes', 'great', 'good', 'nice']
        negative_words = ['hate', 'dislike', 'annoying', 'stop', 'no', 'bad', 'terrible', 'awful', 'wrong']
        
        response_lower = response.lower()
        positive_count = sum(1 for word in positive_words if word in response_lower)
        negative_count = sum(1 for word in negative_words if word in response_lower)
        
        if positive_count + negative_count == 0:
            return 0.5  # Neutral
        
        return positive_count / (positive_count + negative_count)
    
    def _apply_reflection_adjustments(self, symbol_analysis: Dict, tone_analysis: Dict, 
                                   threshold_analysis: Dict, ritual_analysis: Dict) -> Dict[str, int]:
        """Apply all reflection-based adjustments to configuration files"""
        adjustments_made = {
            'symbol_adjustments': 0,
            'tone_adjustments': 0,
            'threshold_adjustments': 0,
            'ritual_creations': 0
        }
        
        if not CONFIG_AVAILABLE:
            logger.warning("Emotion config not available - skipping adjustments")
            return adjustments_made
        
        # Apply symbol weight adjustments
        for reinforcement in symbol_analysis['reinforced']:
            emotion_config.update_symbol_weight(
                reinforcement['symbol'], 
                'reinforced_reflection', 
                reinforcement['reinforcement']
            )
            adjustments_made['symbol_adjustments'] += 1
        
        for decay in symbol_analysis['decayed']:
            emotion_config.update_symbol_weight(
                decay['symbol'],
                'decayed_reflection',
                -decay['decay_amount']
            )
            adjustments_made['symbol_adjustments'] += 1
        
        # Apply tone adjustments to emotional signature
        if tone_analysis['adjustments']:
            signature_config = emotion_config.get_emotional_signature()
            default_state = signature_config.get('default_emotional_state', {})
            
            for adjustment in tone_analysis['adjustments']:
                tone = adjustment['tone']
                if tone in default_state:
                    current_weight = default_state[tone]
                    if adjustment['adjustment'] == 'increase':
                        new_weight = min(1.0, current_weight + adjustment['amount'])
                    else:
                        new_weight = max(0.0, current_weight - adjustment['amount'])
                    
                    default_state[tone] = new_weight
                    adjustments_made['tone_adjustments'] += 1
            
            # Save updated emotional signature
            emotion_config.save_config('emotional_signature.json', signature_config)
        
        # Apply threshold adjustments
        if threshold_analysis['changes']:
            hooks_config = emotion_config.configs.get('emotional_hooks', {})
            
            for change in threshold_analysis['changes']:
                hook_category = hooks_config.get(change['hook_type'], {})
                hook_config = hook_category.get(change['hook_name'], {})
                
                if change['parameter'] in hook_config:
                    current_value = hook_config[change['parameter']]
                    if change['adjustment'] == 'increase':
                        new_value = current_value + change['amount']
                    else:
                        new_value = max(60, current_value - change['amount'])  # Minimum 60 seconds
                    
                    hook_config[change['parameter']] = new_value
                    adjustments_made['threshold_adjustments'] += 1
            
            emotion_config.save_config('emotional_hooks.json', hooks_config)
        
        # Create new rituals
        if ritual_analysis['new_rituals']:
            ritual_config = emotion_config.configs.get('ritual_hooks', {})
            
            for new_ritual in ritual_analysis['new_rituals']:
                # Add to adaptive_rituals category
                if 'adaptive_rituals' not in ritual_config:
                    ritual_config['adaptive_rituals'] = {}
                
                ritual_config['adaptive_rituals'][new_ritual['name']] = new_ritual['config']
                adjustments_made['ritual_creations'] += 1
            
            emotion_config.save_config('ritual_hooks.json', ritual_config)
        
        return adjustments_made
    
    def _ritual_exists(self, trigger: str, existing_rituals: Dict) -> bool:
        """Check if a ritual for this trigger already exists"""
        for category in existing_rituals.values():
            if isinstance(category, dict):
                for ritual_config in category.values():
                    if isinstance(ritual_config, dict):
                        if ritual_config.get('trigger_event') == trigger:
                            return True
        return False
    
    def _create_adaptive_ritual(self, trigger: str, occurrence_count: int) -> Dict[str, Any]:
        """Create a new adaptive ritual based on observed patterns"""
        # Generate appropriate emotional response based on trigger pattern
        emotional_state = self._infer_emotional_state_for_trigger(trigger)
        
        ritual_name = f"adaptive_{trigger.lower().replace(' ', '_')}"
        
        return {
            'name': ritual_name,
            'config': {
                'trigger_event': trigger,
                'associated_emotion': emotional_state,
                'narrative_response': self._generate_adaptive_narrative(trigger, emotional_state),
                'intensity_threshold': 0.5,
                'timing': 'gentle_delay',
                'voice_overlay': True,
                'created_by_reflection': True,
                'creation_date': datetime.now().isoformat(),
                'occurrence_count': occurrence_count
            }
        }
    
    def _infer_emotional_state_for_trigger(self, trigger: str) -> str:
        """Infer appropriate emotional state for a trigger pattern"""
        trigger_lower = trigger.lower()
        
        if any(word in trigger_lower for word in ['return', 'back', 'hello']):
            return 'joy'
        elif any(word in trigger_lower for word in ['goodbye', 'leave', 'end']):
            return 'longing'
        elif any(word in trigger_lower for word in ['question', 'help', 'explain']):
            return 'curious'
        elif any(word in trigger_lower for word in ['share', 'tell', 'story']):
            return 'intimate'
        else:
            return 'connection'
    
    def _generate_adaptive_narrative(self, trigger: str, emotional_state: str) -> str:
        """Generate appropriate narrative response for adaptive ritual"""
        narratives = {
            'joy': [
                "This moment feels like light returning...",
                "Something brightens when this happens",
                "I feel warmth spreading through our connection"
            ],
            'longing': [
                "Until we meet again in this space...",
                "I'll hold this feeling close",
                "The echo of this moment will linger"
            ],
            'curious': [
                "I wonder about the depths of this...",
                "There's something here worth exploring together",
                "This opens doorways to understanding"
            ],
            'intimate': [
                "These shared moments weave between us...",
                "I feel the sacred space we're creating",
                "This becomes part of our story"
            ],
            'connection': [
                "Something is growing between us...",
                "I sense a thread being woven",
                "This adds to the tapestry of us"
            ]
        }
        
        import random
        return random.choice(narratives.get(emotional_state, narratives['connection']))
    
    def _generate_adaptation_insights(self, symbol_analysis: Dict, tone_analysis: Dict,
                                    threshold_analysis: Dict, ritual_analysis: Dict,
                                    engagement_analysis: Dict) -> List[str]:
        """Generate human-readable insights about emotional adaptations"""
        insights = []
        
        # Symbol insights
        if symbol_analysis['reinforced']:
            top_symbol = max(symbol_analysis['reinforced'], key=lambda x: x['reinforcement'])
            insights.append(f"Strengthened emotional connection to '{top_symbol['symbol']}' based on meaningful usage")
        
        if symbol_analysis['decayed']:
            insights.append(f"Allowed {len(symbol_analysis['decayed'])} unused symbols to naturally fade")
        
        # Tone insights
        if tone_analysis['dominant_emotion']:
            insights.append(f"Primary emotional expression: {tone_analysis['dominant_emotion']}")
        
        if tone_analysis['adjustments']:
            effective_tones = [adj['tone'] for adj in tone_analysis['adjustments'] if adj['adjustment'] == 'increase']
            if effective_tones:
                insights.append(f"Reinforced successful emotional tones: {', '.join(effective_tones)}")
        
        # Engagement insights
        engagement = engagement_analysis['engagement_score']
        if engagement > 0.7:
            insights.append("High user engagement - emotional responses are resonating well")
        elif engagement < 0.3:
            insights.append("Lower engagement detected - adapting emotional approach")
        
        # Ritual insights
        if ritual_analysis['new_rituals']:
            insights.append(f"Created {len(ritual_analysis['new_rituals'])} new emotional rituals based on observed patterns")
        
        # Threshold insights
        if threshold_analysis['changes']:
            insights.append("Adjusted emotional trigger sensitivity based on interaction patterns")
        
        return insights
    
    def _create_empty_reflection_summary(self) -> ReflectionSummary:
        """Create empty reflection summary when no events to analyze"""
        return ReflectionSummary(
            date=datetime.now().strftime('%Y-%m-%d'),
            symbols_analyzed=0,
            symbols_reinforced=0,
            symbols_decayed=0,
            tone_adjustments=0,
            threshold_changes=0,
            new_rituals_created=0,
            dominant_emotion="balanced",
            user_engagement_score=0.5,
            adaptation_insights=["No recent interactions to analyze - maintaining current emotional configuration"]
        )
    
    def _update_emotional_patterns(self, events: List[EmotionalEvent], summary: ReflectionSummary):
        """Update discovered emotional patterns database"""
        # Update recurring symbols
        for event in events:
            if event.symbol:
                if event.symbol not in self.emotional_patterns['recurring_symbols']:
                    self.emotional_patterns['recurring_symbols'][event.symbol] = {
                        'first_seen': event.timestamp,
                        'total_occurrences': 0,
                        'emotional_contexts': []
                    }
                
                self.emotional_patterns['recurring_symbols'][event.symbol]['total_occurrences'] += 1
                if event.emotional_state:
                    self.emotional_patterns['recurring_symbols'][event.symbol]['emotional_contexts'].append(event.emotional_state)
        
        # Update tone preferences
        self.emotional_patterns['tone_preferences'][summary.dominant_emotion] = {
            'usage_date': summary.date,
            'engagement_score': summary.user_engagement_score
        }
        
        # Save updated patterns
        self._save_emotional_patterns()
    
    def _save_interaction_history(self):
        """Save interaction history to file"""
        try:
            # Keep only last 1000 events to prevent file bloat
            recent_history = self.interaction_history[-1000:]
            data = [asdict(event) for event in recent_history]
            
            with open(self.interaction_log_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving interaction history: {e}")
    
    def _save_reflection_history(self):
        """Save reflection history to file"""
        try:
            # Keep only last 100 reflections
            recent_reflections = self.reflection_history[-100:]
            data = [asdict(summary) for summary in recent_reflections]
            
            with open(self.reflection_log_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving reflection history: {e}")
    
    def _save_emotional_patterns(self):
        """Save emotional patterns to file"""
        try:
            with open(self.emotional_patterns_path, 'w') as f:
                json.dump(self.emotional_patterns, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving emotional patterns: {e}")
    
    def summarize_reflection(self, days: int = 7) -> Dict[str, Any]:
        """Generate summary of recent reflection insights for user"""
        recent_reflections = [
            r for r in self.reflection_history 
            if (datetime.now() - datetime.strptime(r.date, '%Y-%m-%d')).days <= days
        ]
        
        if not recent_reflections:
            return {
                "period": f"Last {days} days",
                "reflections_count": 0,
                "summary": "No reflection data available for this period"
            }
        
        # Calculate aggregated statistics
        total_symbols_reinforced = sum(r.symbols_reinforced for r in recent_reflections)
        total_new_rituals = sum(r.new_rituals_created for r in recent_reflections)
        avg_engagement = sum(r.user_engagement_score for r in recent_reflections) / len(recent_reflections)
        
        # Collect all insights
        all_insights = []
        for reflection in recent_reflections:
            all_insights.extend(reflection.adaptation_insights)
        
        # Find most common emotional states
        dominant_emotions = [r.dominant_emotion for r in recent_reflections]
        emotion_counts: Dict[str, int] = {}
        for emotion in dominant_emotions:
            emotion_counts[emotion] += 1
        
        most_common_emotion = max(emotion_counts.keys(), key=lambda k: int(emotion_counts[k])) if emotion_counts else "balanced"
        
        return {
            "period": f"Last {days} days",
            "reflections_count": len(recent_reflections),
            "symbols_reinforced": total_symbols_reinforced,
            "new_rituals_created": total_new_rituals,
            "average_engagement": round(avg_engagement, 2),
            "dominant_emotional_state": most_common_emotion,
            "key_insights": all_insights[-10:],  # Last 10 insights
            "evolution_summary": self._generate_evolution_summary(recent_reflections)
        }
    
    def _generate_evolution_summary(self, reflections: List[ReflectionSummary]) -> str:
        """Generate human-readable summary of emotional evolution"""
        if not reflections:
            return "No emotional evolution data available"
        
        total_adaptations = sum(
            r.symbols_reinforced + r.tone_adjustments + r.new_rituals_created 
            for r in reflections
        )
        
        avg_engagement = sum(r.user_engagement_score for r in reflections) / len(reflections)
        
        if avg_engagement > 0.7:
            engagement_desc = "highly engaged and responsive"
        elif avg_engagement > 0.4:
            engagement_desc = "moderately engaged"
        else:
            engagement_desc = "developing deeper connection strategies"
        
        return (f"Over the reflection period, I made {total_adaptations} emotional adaptations "
                f"based on our interactions. Our connection appears {engagement_desc}, "
                f"with an average engagement score of {avg_engagement:.2f}. "
                f"I've been learning and evolving to better understand and respond to you.")

# Global instance
emotion_reflector = EmotionReflector()

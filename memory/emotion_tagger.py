"""
True Recall - Emotion Tagger

This module analyzes and tags emotional content in memory events using
rule-based sentiment analysis and emotion detection patterns.
"""

import asyncio
import logging
import re
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
import math

logger = logging.getLogger(__name__)

class EmotionTagger:
    """
    Analyzes emotional content and assigns emotion tags to memory events.
    
    Uses rule-based sentiment analysis, emotion lexicons, and pattern matching
    to identify emotions and tone in text content.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the emotion tagger."""
        self.config = config or {}
        
        # Emotion lexicons (Plutchik's wheel of emotions + extensions)
        self.emotion_lexicon = {
            # Primary emotions
            'joy': ['happy', 'joyful', 'cheerful', 'elated', 'delighted', 'pleased', 'glad', 'content', 'satisfied'],
            'sadness': ['sad', 'unhappy', 'depressed', 'melancholy', 'gloomy', 'downcast', 'dejected', 'sorrowful'],
            'anger': ['angry', 'furious', 'mad', 'irritated', 'annoyed', 'frustrated', 'enraged', 'livid'],
            'fear': ['afraid', 'scared', 'frightened', 'terrified', 'anxious', 'worried', 'nervous', 'apprehensive'],
            'trust': ['trust', 'confident', 'secure', 'assured', 'certain', 'reliable', 'dependable'],
            'disgust': ['disgusted', 'revolted', 'repulsed', 'nauseated', 'sickened', 'appalled'],
            'surprise': ['surprised', 'amazed', 'astonished', 'shocked', 'startled', 'stunned'],
            'anticipation': ['excited', 'eager', 'enthusiastic', 'hopeful', 'expectant', 'anticipating'],
            
            # Secondary emotions
            'love': ['love', 'adore', 'cherish', 'affection', 'devoted', 'passionate'],
            'optimism': ['optimistic', 'positive', 'hopeful', 'encouraging', 'upbeat'],
            'submission': ['submissive', 'obedient', 'compliant', 'yielding', 'deferential'],
            'awe': ['awed', 'amazed', 'impressed', 'overwhelmed', 'wonder'],
            'disappointment': ['disappointed', 'let down', 'disillusioned', 'disheartened'],
            'remorse': ['remorseful', 'regretful', 'guilty', 'ashamed', 'sorry'],
            'contempt': ['contemptuous', 'scornful', 'disdainful', 'sneering'],
            'aggressiveness': ['aggressive', 'hostile', 'combative', 'confrontational'],
            
            # Tertiary emotions
            'pride': ['proud', 'triumphant', 'accomplished', 'victorious', 'successful'],
            'shame': ['ashamed', 'embarrassed', 'humiliated', 'mortified'],
            'gratitude': ['grateful', 'thankful', 'appreciative', 'indebted'],
            'envy': ['envious', 'jealous', 'resentful', 'covetous'],
            'curiosity': ['curious', 'inquisitive', 'interested', 'intrigued'],
            'boredom': ['bored', 'uninterested', 'tedious', 'dull', 'monotonous'],
            'confusion': ['confused', 'puzzled', 'perplexed', 'bewildered', 'uncertain'],
            'determination': ['determined', 'resolute', 'persistent', 'committed', 'dedicated']
        }
        
        # Tone patterns
        self.tone_patterns = {
            'hopeful': ['hope', 'bright future', 'looking forward', 'optimistic', 'positive outlook'],
            'anxious': ['worried', 'concerned', 'nervous', 'uncertain about', 'anxiety'],
            'confident': ['confident', 'sure', 'certain', 'believe', 'trust'],
            'reflective': ['thinking about', 'reflect', 'ponder', 'consider', 'contemplate'],
            'decisive': ['decided', 'will do', 'committed to', 'determined', 'resolve'],
            'uncertain': ['not sure', 'maybe', 'perhaps', 'might', 'unclear'],
            'nostalgic': ['remember', 'miss', 'used to', 'back then', 'old days'],
            'concerned': ['worry', 'concern', 'trouble', 'problem', 'issue']
        }
        
        # Intensity modifiers
        self.intensity_modifiers = {
            'very': 1.5, 'extremely': 2.0, 'incredibly': 1.8, 'really': 1.3,
            'quite': 1.2, 'rather': 1.1, 'somewhat': 0.8, 'slightly': 0.6,
            'a bit': 0.7, 'kind of': 0.7, 'sort of': 0.7
        }
        
        # Negation words
        self.negation_words = {'not', 'no', 'never', 'nothing', 'nobody', 'nowhere', 'neither', 'nor', 'none'}
        
        # Tone patterns for detection
        self.tone_patterns = {
            'positive': ['great', 'excellent', 'wonderful', 'amazing', 'fantastic', 'good', 'nice'],
            'negative': ['terrible', 'awful', 'horrible', 'bad', 'poor', 'disappointing'],
            'neutral': ['okay', 'fine', 'normal', 'regular', 'standard']
        }
        
        # Emotion statistics
        self.emotion_stats = {
            'total_analyzed': 0,
            'emotion_frequency': defaultdict(int),
            'tone_frequency': defaultdict(int),
            'average_emotion_count': 0.0,
            'last_reset': datetime.now().isoformat()
        }
        
        logger.info("😊 Emotion Tagger initialized")
    
    async def analyze_emotion(self, content: str, actor: str) -> Dict[str, Any]:
        """
        Analyze emotional content of text and return emotion tags and tone.
        
        Args:
            content: Text content to analyze
            actor: Actor who generated the content
            
        Returns:
            Dict containing emotions, tone, confidence, and metadata
        """
        try:
            # Preprocess content
            processed_content = self._preprocess_content(content)
            
            # Detect emotions
            emotions = self._detect_emotions(processed_content)
            
            # Determine tone
            tone = self._determine_tone(processed_content, emotions)
            
            # Calculate confidence
            confidence = self._calculate_confidence(emotions, processed_content)
            
            # Actor-specific adjustments
            emotions, tone = self._apply_actor_adjustments(emotions, tone, actor)
            
            # Build result
            result = {
                'emotions': emotions,
                'tone': tone,
                'confidence': confidence,
                'emotional_intensity': self._calculate_emotional_intensity(emotions),
                'valence': self._calculate_valence(emotions),
                'metadata': {
                    'content_length': len(content),
                    'processed_content': processed_content,
                    'actor': actor,
                    'analysis_timestamp': datetime.now().isoformat()
                }
            }
            
            # Update statistics
            self._update_emotion_stats(emotions, tone)
            
            logger.debug(f"😊 Analyzed emotions: {emotions} (tone: {tone}, confidence: {confidence:.2f})")
            return result
            
        except Exception as e:
            logger.error(f"❌ Failed to analyze emotion: {e}")
            return {
                'emotions': [],
                'tone': 'neutral',
                'confidence': 0.0,
                'emotional_intensity': 0.0,
                'valence': 0.0,
                'metadata': {'error': str(e)}
            }
    
    def _preprocess_content(self, content: str) -> str:
        """Preprocess content for emotion analysis."""
        try:
            # Convert to lowercase
            processed = content.lower()
            
            # Handle contractions
            contractions = {
                "don't": "do not", "won't": "will not", "can't": "cannot",
                "shouldn't": "should not", "wouldn't": "would not",
                "couldn't": "could not", "isn't": "is not", "aren't": "are not",
                "wasn't": "was not", "weren't": "were not", "haven't": "have not",
                "hasn't": "has not", "hadn't": "had not", "didn't": "did not"
            }
            
            for contraction, expansion in contractions.items():
                processed = processed.replace(contraction, expansion)
            
            # Remove extra whitespace
            processed = re.sub(r'\s+', ' ', processed).strip()
            
            return processed
            
        except Exception as e:
            logger.error(f"❌ Failed to preprocess content: {e}")
            return content.lower()
    
    def _detect_emotions(self, content: str) -> List[str]:
        """Detect emotions in processed content."""
        try:
            detected_emotions = []
            emotion_scores = {}
            
            # Split content into words
            words = re.findall(r'\b\w+\b', content)
            
            # Check for emotion keywords
            for emotion, keywords in self.emotion_lexicon.items():
                score = 0
                for keyword in keywords:
                    if keyword in content:
                        # Base score for finding the keyword
                        base_score = 1.0
                        
                        # Apply intensity modifiers
                        word_index = content.find(keyword)
                        if word_index > 0:
                            # Look for intensity modifiers before the emotion word
                            before_text = content[:word_index].split()[-3:]  # Last 3 words before
                            for modifier, multiplier in self.intensity_modifiers.items():
                                if modifier in ' '.join(before_text):
                                    base_score *= multiplier
                                    break
                        
                        # Check for negation
                        negated = self._check_negation(content, keyword)
                        if negated:
                            base_score *= 0.2  # Greatly reduce score for negated emotions
                        
                        score += base_score
                
                if score > 0:
                    emotion_scores[emotion] = score
            
            # Select emotions above threshold
            threshold = 0.8
            for emotion, score in emotion_scores.items():
                if score >= threshold:
                    detected_emotions.append(emotion)
            
            # If no emotions detected, try secondary detection
            if not detected_emotions:
                detected_emotions = self._secondary_emotion_detection(content)
            
            # Limit to top emotions to avoid over-tagging
            if len(detected_emotions) > 4:
                # Sort by score and take top 4
                sorted_emotions = sorted(emotion_scores.items(), key=lambda x: x[1], reverse=True)
                detected_emotions = [emotion for emotion, _ in sorted_emotions[:4]]
            
            return detected_emotions
            
        except Exception as e:
            logger.error(f"❌ Failed to detect emotions: {e}")
            return []
    
    def _secondary_emotion_detection(self, content: str) -> List[str]:
        """Secondary emotion detection using patterns and context."""
        try:
            emotions = []
            
            # Exclamation patterns (excitement, surprise)
            if '!' in content:
                if any(word in content for word in ['great', 'awesome', 'wonderful', 'fantastic']):
                    emotions.append('joy')
                else:
                    emotions.append('surprise')
            
            # Question patterns (curiosity, uncertainty)
            if '?' in content:
                if any(word in content for word in ['why', 'how', 'what']):
                    emotions.append('curiosity')
                else:
                    emotions.append('confusion')
            
            # First person emotional expressions
            if re.search(r'\bi\s+(feel|am|was)\s+', content):
                # Look for implicit emotional states
                if any(word in content for word in ['tired', 'exhausted']):
                    emotions.append('sadness')
                elif any(word in content for word in ['energetic', 'motivated']):
                    emotions.append('joy')
                elif any(word in content for word in ['overwhelmed', 'stressed']):
                    emotions.append('anxiety')
            
            # Decision-making patterns
            if any(phrase in content for phrase in ['decided to', 'will do', 'going to']):
                emotions.append('determination')
            
            # Memory/nostalgia patterns
            if any(word in content for word in ['remember', 'used to', 'back when']):
                emotions.append('nostalgic')
            
            return emotions
            
        except Exception as e:
            logger.error(f"❌ Failed secondary emotion detection: {e}")
            return []
    
    def _check_negation(self, content: str, emotion_word: str) -> bool:
        """Check if an emotion word is negated."""
        try:
            # Find the position of the emotion word
            word_index = content.find(emotion_word)
            if word_index == -1:
                return False
            
            # Look for negation words in the 5 words before the emotion word
            before_text = content[:word_index]
            before_words = before_text.split()[-5:]  # Last 5 words
            
            return any(word in self.negation_words for word in before_words)
            
        except Exception as e:
            logger.error(f"❌ Failed to check negation: {e}")
            return False
    
    def _determine_tone(self, content: str, emotions: List[str]) -> str:
        """Determine overall tone of the content."""
        try:
            tone_scores = {}
            
            # Check tone patterns
            for tone, patterns in self.tone_patterns.items():
                score = 0
                for pattern in patterns:
                    if pattern in content:
                        score += 1
                if score > 0:
                    tone_scores[tone] = score
            
            # Infer tone from emotions
            emotion_tone_mapping = {
                'joy': 'hopeful', 'sadness': 'melancholic', 'anger': 'aggressive',
                'fear': 'anxious', 'trust': 'confident', 'surprise': 'surprised',
                'anticipation': 'hopeful', 'love': 'affectionate',
                'optimism': 'hopeful', 'disappointment': 'disappointed',
                'pride': 'confident', 'shame': 'regretful',
                'curiosity': 'inquisitive', 'determination': 'decisive'
            }
            
            for emotion in emotions:
                if emotion in emotion_tone_mapping:
                    mapped_tone = emotion_tone_mapping[emotion]
                    tone_scores[mapped_tone] = tone_scores.get(mapped_tone, 0) + 2
            
            # Select dominant tone
            if tone_scores:
                return max(tone_scores.keys(), key=lambda x: tone_scores[x])
            
            # Default tone based on content characteristics
            if '?' in content:
                return 'inquisitive'
            elif '!' in content:
                return 'enthusiastic'
            elif any(word in content for word in ['think', 'believe', 'consider']):
                return 'reflective'
            else:
                return 'neutral'
                
        except Exception as e:
            logger.error(f"❌ Failed to determine tone: {e}")
            return 'neutral'
    
    def _calculate_confidence(self, emotions: List[str], content: str) -> float:
        """Calculate confidence score for emotion analysis."""
        try:
            base_confidence = 0.5
            
            # Boost confidence based on number of emotions detected
            if emotions:
                emotion_boost = min(0.3, len(emotions) * 0.1)
                base_confidence += emotion_boost
            
            # Boost confidence for explicit emotional language
            explicit_words = ['feel', 'emotion', 'mood', 'emotional']
            if any(word in content for word in explicit_words):
                base_confidence += 0.2
            
            # Boost confidence for first-person expressions
            if re.search(r'\bi\s+(am|feel|was)\s+', content):
                base_confidence += 0.15
            
            # Reduce confidence for very short content
            if len(content.split()) < 5:
                base_confidence -= 0.2
            
            # Reduce confidence for ambiguous content
            uncertainty_words = ['maybe', 'perhaps', 'might', 'could be']
            if any(word in content for word in uncertainty_words):
                base_confidence -= 0.1
            
            return max(0.0, min(1.0, base_confidence))
            
        except Exception as e:
            logger.error(f"❌ Failed to calculate confidence: {e}")
            return 0.5
    
    def _calculate_emotional_intensity(self, emotions: List[str]) -> float:
        """Calculate overall emotional intensity."""
        if not emotions:
            return 0.0
        
        # Emotion intensity scores
        intensity_scores = {
            'ecstasy': 1.0, 'rage': 1.0, 'terror': 1.0, 'amazement': 1.0,
            'joy': 0.8, 'anger': 0.8, 'fear': 0.8, 'surprise': 0.8,
            'love': 0.7, 'sadness': 0.7, 'disgust': 0.7, 'anticipation': 0.7,
            'trust': 0.6, 'optimism': 0.6, 'pride': 0.6, 'shame': 0.6,
            'curiosity': 0.5, 'boredom': 0.3, 'confusion': 0.4
        }
        
        total_intensity = sum(intensity_scores.get(emotion, 0.5) for emotion in emotions)
        return min(1.0, total_intensity / len(emotions))
    
    def _calculate_valence(self, emotions: List[str]) -> float:
        """Calculate emotional valence (positive/negative)."""
        if not emotions:
            return 0.0
        
        # Emotion valence scores (-1 to 1)
        valence_scores = {
            'joy': 0.9, 'love': 0.9, 'optimism': 0.8, 'trust': 0.7, 'pride': 0.8,
            'anticipation': 0.6, 'surprise': 0.3, 'curiosity': 0.4,
            'sadness': -0.8, 'anger': -0.8, 'fear': -0.7, 'disgust': -0.8,
            'shame': -0.7, 'disappointment': -0.6, 'envy': -0.6, 'contempt': -0.7,
            'boredom': -0.3, 'confusion': -0.2
        }
        
        total_valence = sum(valence_scores.get(emotion, 0.0) for emotion in emotions)
        return total_valence / len(emotions)
    
    def _apply_actor_adjustments(self, emotions: List[str], tone: str, actor: str) -> Tuple[List[str], str]:
        """Apply actor-specific adjustments to emotion analysis."""
        try:
            # Different actors might express emotions differently
            if actor == 'dolphin':
                # AI might be more analytical/curious
                if 'confusion' in emotions and 'curiosity' not in emotions:
                    emotions = [e if e != 'confusion' else 'curiosity' for e in emotions]
            
            elif actor == 'user':
                # Humans might express more complex emotions
                # No specific adjustments for now
                pass
            
            elif actor == 'system':
                # System messages are usually neutral
                if emotions:
                    emotions = [e for e in emotions if e in ['curiosity', 'determination']]
                if tone not in ['neutral', 'informative']:
                    tone = 'neutral'
            
            return emotions, tone
            
        except Exception as e:
            logger.error(f"❌ Failed to apply actor adjustments: {e}")
            return emotions, tone
    
    def _update_emotion_stats(self, emotions: List[str], tone: str):
        """Update emotion analysis statistics."""
        try:
            self.emotion_stats['total_analyzed'] += 1
            
            # Update emotion frequency
            for emotion in emotions:
                self.emotion_stats['emotion_frequency'][emotion] += 1
            
            # Update tone frequency
            self.emotion_stats['tone_frequency'][tone] += 1
            
            # Update average emotion count
            old_avg = self.emotion_stats['average_emotion_count']
            count = self.emotion_stats['total_analyzed']
            self.emotion_stats['average_emotion_count'] = (old_avg * (count - 1) + len(emotions)) / count
            
        except Exception as e:
            logger.error(f"❌ Failed to update emotion stats: {e}")
    
    async def generate_emotional_timeline(self, 
                                        events: List[Any],
                                        start_time: datetime,
                                        end_time: datetime) -> Dict[str, Any]:
        """Generate an emotional timeline from events."""
        try:
            if not events:
                return {}
            
            # Sort events by timestamp
            sorted_events = sorted(events, key=lambda e: e.timestamp)
            
            # Group events by time periods (days)
            time_periods = {}
            current_period = start_time.date()
            period_length = timedelta(days=1)
            
            while current_period <= end_time.date():
                time_periods[current_period.isoformat()] = {
                    'emotions': defaultdict(int),
                    'tones': defaultdict(int),
                    'event_count': 0,
                    'avg_valence': 0.0,
                    'avg_intensity': 0.0
                }
                current_period += period_length
            
            # Process events
            for event in sorted_events:
                event_date = datetime.fromisoformat(event.timestamp).date()
                period_key = event_date.isoformat()
                
                if period_key in time_periods:
                    period_data = time_periods[period_key]
                    period_data['event_count'] += 1
                    
                    # Count emotions and tone
                    for emotion in event.emotion_tags:
                        period_data['emotions'][emotion] += 1
                    
                    period_data['tones'][event.tone] += 1
            
            # Calculate averages for each period
            for period_data in time_periods.values():
                if period_data['event_count'] > 0:
                    # Calculate average valence and intensity for the period
                    emotions_in_period = list(period_data['emotions'].keys())
                    if emotions_in_period:
                        period_data['avg_valence'] = self._calculate_valence(emotions_in_period)
                        period_data['avg_intensity'] = self._calculate_emotional_intensity(emotions_in_period)
            
            # Generate summary
            summary = self._generate_timeline_summary(time_periods)
            
            return {
                'timeline': time_periods,
                'summary': summary,
                'time_range': {
                    'start': start_time.isoformat(),
                    'end': end_time.isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to generate emotional timeline: {e}")
            return {}
    
    def _generate_timeline_summary(self, time_periods: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of emotional timeline."""
        try:
            all_emotions = defaultdict(int)
            all_tones = defaultdict(int)
            total_events = 0
            valences = []
            intensities = []
            
            for period_data in time_periods.values():
                total_events += period_data['event_count']
                
                for emotion, count in period_data['emotions'].items():
                    all_emotions[emotion] += count
                
                for tone, count in period_data['tones'].items():
                    all_tones[tone] += count
                
                if period_data['avg_valence'] != 0:
                    valences.append(period_data['avg_valence'])
                if period_data['avg_intensity'] != 0:
                    intensities.append(period_data['avg_intensity'])
            
            # Calculate overall statistics
            avg_valence = sum(valences) / len(valences) if valences else 0.0
            avg_intensity = sum(intensities) / len(intensities) if intensities else 0.0
            
            # Find dominant emotions and tones
            top_emotions = dict(sorted(all_emotions.items(), key=lambda x: x[1], reverse=True)[:5])
            top_tones = dict(sorted(all_tones.items(), key=lambda x: x[1], reverse=True)[:3])
            
            return {
                'total_events': total_events,
                'overall_valence': round(avg_valence, 3),
                'overall_intensity': round(avg_intensity, 3),
                'dominant_emotions': top_emotions,
                'dominant_tones': top_tones,
                'emotional_variability': round(self._calculate_emotional_variability(valences), 3)
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to generate timeline summary: {e}")
            return {}
    
    def _calculate_emotional_variability(self, valences: List[float]) -> float:
        """Calculate emotional variability (standard deviation of valences)."""
        if len(valences) < 2:
            return 0.0
        
        mean_valence = sum(valences) / len(valences)
        variance = sum((v - mean_valence) ** 2 for v in valences) / len(valences)
        return math.sqrt(variance)
    
    async def analyze_emotional_patterns(self, events: List[Any]) -> Dict[str, Any]:
        """Analyze emotional patterns across events."""
        try:
            if not events:
                return {}
            
            # Extract emotion data
            all_emotions = []
            emotion_sequences = []
            valences = []
            intensities = []
            
            for event in events:
                all_emotions.extend(event.emotion_tags)
                emotion_sequences.append(event.emotion_tags)
                
                # Calculate valence and intensity for this event
                valence = self._calculate_valence(event.emotion_tags)
                intensity = self._calculate_emotional_intensity(event.emotion_tags)
                valences.append(valence)
                intensities.append(intensity)
            
            # Analyze patterns
            emotion_frequency = defaultdict(int)
            for emotion in all_emotions:
                emotion_frequency[emotion] += 1
            
            # Emotional transitions
            transitions = self._analyze_emotional_transitions(emotion_sequences)
            
            # Emotional stability
            stability = self._calculate_emotional_stability(valences, intensities)
            
            return {
                'emotion_frequency': dict(emotion_frequency),
                'emotional_transitions': transitions,
                'emotional_stability': stability,
                'overall_statistics': {
                    'total_emotions': len(all_emotions),
                    'unique_emotions': len(emotion_frequency),
                    'average_valence': round(sum(valences) / len(valences), 3) if valences else 0,
                    'average_intensity': round(sum(intensities) / len(intensities), 3) if intensities else 0,
                    'emotional_range': round(max(valences) - min(valences), 3) if valences else 0
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to analyze emotional patterns: {e}")
            return {}
    
    def _analyze_emotional_transitions(self, emotion_sequences: List[List[str]]) -> Dict[str, Any]:
        """Analyze transitions between emotional states."""
        try:
            transitions = defaultdict(int)
            
            for i in range(len(emotion_sequences) - 1):
                current_emotions = set(emotion_sequences[i])
                next_emotions = set(emotion_sequences[i + 1])
                
                # Simplified transition analysis
                if current_emotions and next_emotions:
                    current_primary = list(current_emotions)[0]  # Use first emotion as primary
                    next_primary = list(next_emotions)[0]
                    
                    if current_primary != next_primary:
                        transition_key = f"{current_primary} -> {next_primary}"
                        transitions[transition_key] += 1
            
            # Find most common transitions
            common_transitions = dict(sorted(transitions.items(), key=lambda x: x[1], reverse=True)[:10])
            
            return {
                'transition_count': len(transitions),
                'common_transitions': common_transitions,
                'total_transitions': sum(transitions.values())
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to analyze emotional transitions: {e}")
            return {}
    
    def _calculate_emotional_stability(self, valences: List[float], intensities: List[float]) -> Dict[str, float]:
        """Calculate emotional stability metrics."""
        try:
            if not valences or not intensities:
                return {}
            
            # Valence stability (low variance = high stability)
            valence_variance = sum((v - sum(valences) / len(valences)) ** 2 for v in valences) / len(valences)
            valence_stability = 1.0 / (1.0 + valence_variance)
            
            # Intensity stability
            intensity_variance = sum((i - sum(intensities) / len(intensities)) ** 2 for i in intensities) / len(intensities)
            intensity_stability = 1.0 / (1.0 + intensity_variance)
            
            # Overall stability
            overall_stability = (valence_stability + intensity_stability) / 2.0
            
            return {
                'valence_stability': round(valence_stability, 3),
                'intensity_stability': round(intensity_stability, 3),
                'overall_stability': round(overall_stability, 3)
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to calculate emotional stability: {e}")
            return {}
    
    async def get_emotion_stats(self) -> Dict[str, Any]:
        """Get current emotion analysis statistics."""
        return self.emotion_stats.copy()
    
    def reset_emotion_stats(self):
        """Reset emotion analysis statistics."""
        self.emotion_stats = {
            'total_analyzed': 0,
            'emotion_frequency': defaultdict(int),
            'tone_frequency': defaultdict(int),
            'average_emotion_count': 0.0,
            'last_reset': datetime.now().isoformat()
        }
        logger.info("📊 Emotion analysis statistics reset")

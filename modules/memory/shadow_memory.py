"""
Shadow Memory Layer

Tracks unconscious desires or suppressed themes, hinted not stated.
Lets characters notice what the user avoids or dreams of but won't admit aloud.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple, Any
from enum import Enum
import json
import os
import re
from collections import defaultdict, Counter
from dataclasses import dataclass

class ShadowThemeType(Enum):
    SUPPRESSED_DESIRE = "suppressed_desire"
    AVOIDED_TOPIC = "avoided_topic"
    UNCONSCIOUS_PATTERN = "unconscious_pattern"
    PROJECTED_FEAR = "projected_fear"
    HIDDEN_ASPIRATION = "hidden_aspiration"
    DENIED_EMOTION = "denied_emotion"
    UNSPOKEN_NEED = "unspoken_need"
    REPRESSED_MEMORY = "repressed_memory"

class ManifestationStrength(Enum):
    BARELY_DETECTABLE = "barely_detectable"  # 0.1-0.3
    SUBTLE_HINTS = "subtle_hints"  # 0.3-0.5
    MODERATE_SIGNS = "moderate_signs"  # 0.5-0.7
    STRONG_INDICATORS = "strong_indicators"  # 0.7-0.9
    BREAKING_THROUGH = "breaking_through"  # 0.9-1.0

@dataclass
class ShadowPattern:
    theme_type: ShadowThemeType
    pattern_name: str
    keywords: Set[str]
    avoidance_signals: Set[str]
    manifestation_strength: float
    first_detected: datetime
    last_reinforced: datetime
    occurrence_count: int
    context_clusters: List[str]
    emotional_charge: float  # How emotionally significant this pattern is
    suppression_indicators: Dict[str, float]  # How actively the user avoids this

@dataclass
class ShadowEvent:
    timestamp: datetime
    trigger_text: str
    detected_patterns: List[str]
    avoidance_behaviors: List[str]
    emotional_displacement: Dict[str, float]
    context: str

class ShadowMemoryLayer:
    """
    Tracks unconscious patterns and suppressed themes in user communications.
    Provides insights for characters to gently acknowledge what isn't being said.
    """
    
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.storage_path = f"storage/shadow_memory/{user_id}_shadow.json"
        self.shadow_patterns: Dict[str, ShadowPattern] = {}
        self.shadow_events: List[ShadowEvent] = []
        self.pattern_detectors = self._initialize_pattern_detectors()
        self.avoidance_detectors = self._initialize_avoidance_detectors()
        self._load_shadow_memory()
    
    def _initialize_pattern_detectors(self) -> Dict[str, Dict[str, Any]]:
        """Initialize pattern detection rules for different shadow themes"""
        return {
            'intimacy_avoidance': {
                'type': ShadowThemeType.AVOIDED_TOPIC,
                'keywords': {'alone', 'independent', 'busy', 'work', 'friends', 'later'},
                'avoidance_signals': {'change topic', 'joke deflection', 'physical distance'},
                'context_triggers': {'romantic_suggestion', 'emotional_vulnerability', 'future_planning'}
            },
            'vulnerability_fear': {
                'type': ShadowThemeType.PROJECTED_FEAR,
                'keywords': {'strong', 'fine', 'handle', 'manage', 'control', 'okay'},
                'avoidance_signals': {'minimizing', 'intellectualizing', 'humor_deflection'},
                'context_triggers': {'emotional_support_offer', 'weakness_admission', 'help_offer'}
            },
            'creative_suppression': {
                'type': ShadowThemeType.SUPPRESSED_DESIRE,
                'keywords': {'practical', 'realistic', 'responsible', 'adult', 'sensible'},
                'avoidance_signals': {'dismissing_dreams', 'self_deprecation', 'excuse_making'},
                'context_triggers': {'creative_discussion', 'dream_sharing', 'artistic_expression'}
            },
            'attachment_hunger': {
                'type': ShadowThemeType.UNSPOKEN_NEED,
                'keywords': {'casual', 'whatever', 'doesn\'t matter', 'no big deal'},
                'avoidance_signals': {'minimizing_importance', 'feigned_indifference', 'quick_dismissal'},
                'context_triggers': {'relationship_discussion', 'future_plans', 'emotional_connection'}
            },
            'success_fear': {
                'type': ShadowThemeType.HIDDEN_ASPIRATION,
                'keywords': {'lucky', 'accident', 'surprise', 'didn\'t expect', 'just happened'},
                'avoidance_signals': {'self_diminishment', 'credit_deflection', 'imposter_syndrome'},
                'context_triggers': {'achievement_discussion', 'goal_setting', 'self_improvement'}
            },
            'emotional_numbness': {
                'type': ShadowThemeType.DENIED_EMOTION,
                'keywords': {'numb', 'empty', 'going through motions', 'automatic', 'routine'},
                'avoidance_signals': {'emotional_flattening', 'disconnection_language', 'routine_focus'},
                'context_triggers': {'feeling_inquiry', 'emotional_check_in', 'mood_discussion'}
            },
            'past_trauma_echoes': {
                'type': ShadowThemeType.REPRESSED_MEMORY,
                'keywords': {'can\'t remember', 'don\'t think about', 'long time ago', 'doesn\'t matter now'},
                'avoidance_signals': {'memory_gaps', 'time_skipping', 'subject_jumping'},
                'context_triggers': {'childhood_discussion', 'family_topics', 'past_relationships'}
            }
        }
    
    def _initialize_avoidance_detectors(self) -> Dict[str, List[str]]:
        """Initialize detectors for avoidance behaviors"""
        return {
            'topic_deflection': [
                'anyway', 'speaking of', 'that reminds me', 'by the way', 'oh, before I forget'
            ],
            'emotional_distancing': [
                'I guess', 'whatever', 'doesn\'t matter', 'not a big deal', 'it\'s fine'
            ],
            'humor_deflection': [
                'haha', 'lol', 'just kidding', 'funny story', 'that\'s hilarious'
            ],
            'intellectualizing': [
                'logically', 'rationally', 'objectively', 'theoretically', 'from a practical standpoint'
            ],
            'minimizing': [
                'just a little', 'not really', 'sort of', 'kind of', 'I suppose'
            ],
            'time_distancing': [
                'long time ago', 'can\'t remember', 'doesn\'t matter now', 'in the past', 'over it'
            ]
        }
    
    def analyze_communication(self, text: str, context: str, emotional_state: Dict[str, float]) -> Dict[str, Any]:
        """
        Analyze communication for shadow patterns and unconscious themes
        
        Args:
            text: The user's communication text
            context: Context of the communication (conversation_type, topic, etc.)
            emotional_state: Current emotional analysis
            
        Returns:
            Shadow analysis results with detected patterns and recommendations
        """
        
        # Detect patterns in the text
        detected_patterns = self._detect_shadow_patterns(text, context)
        
        # Detect avoidance behaviors
        avoidance_behaviors = self._detect_avoidance_behaviors(text)
        
        # Analyze emotional displacement
        emotional_displacement = self._analyze_emotional_displacement(text, emotional_state)
        
        # Update shadow patterns
        self._update_shadow_patterns(detected_patterns, text, context)
        
        # Create shadow event
        shadow_event = ShadowEvent(
            timestamp=datetime.now(),
            trigger_text=text[:200],  # Store first 200 chars for privacy
            detected_patterns=list(detected_patterns.keys()),
            avoidance_behaviors=avoidance_behaviors,
            emotional_displacement=emotional_displacement,
            context=context
        )
        self.shadow_events.append(shadow_event)
        
        # Generate insights for character responses
        shadow_insights = self._generate_shadow_insights(detected_patterns, avoidance_behaviors)
        
        # Save updated state
        self._save_shadow_memory()
        
        return {
            'detected_patterns': detected_patterns,
            'avoidance_behaviors': avoidance_behaviors,
            'emotional_displacement': emotional_displacement,
            'shadow_insights': shadow_insights,
            'character_response_hints': self._generate_character_hints(shadow_insights)
        }
    
    def _detect_shadow_patterns(self, text: str, context: str) -> Dict[str, float]:
        """Detect shadow patterns in the text"""
        detected = {}
        text_lower = text.lower()
        
        for pattern_name, pattern_config in self.pattern_detectors.items():
            detection_score = 0.0
            
            # Check for keywords
            keyword_matches = sum(1 for keyword in pattern_config['keywords'] 
                                if keyword in text_lower)
            if keyword_matches > 0:
                detection_score += (keyword_matches / len(pattern_config['keywords'])) * 0.4
            
            # Check for avoidance signals
            avoidance_matches = sum(1 for signal in pattern_config['avoidance_signals']
                                  if any(word in text_lower for word in signal.split('_')))
            if avoidance_matches > 0:
                detection_score += (avoidance_matches / len(pattern_config['avoidance_signals'])) * 0.4
            
            # Check context triggers
            if context in pattern_config.get('context_triggers', []):
                detection_score += 0.3
            
            # Check for contradictory emotions (saying "fine" with negative emotion)
            if 'fine' in text_lower or 'okay' in text_lower:
                if any(emotion in ['sadness', 'anxiety', 'anger'] for emotion in text_lower):
                    detection_score += 0.2
            
            if detection_score > 0.2:  # Threshold for detection
                detected[pattern_name] = min(detection_score, 1.0)
        
        return detected
    
    def _detect_avoidance_behaviors(self, text: str) -> List[str]:
        """Detect avoidance behaviors in communication"""
        detected_behaviors = []
        text_lower = text.lower()
        
        for behavior_type, indicators in self.avoidance_detectors.items():
            if any(indicator in text_lower for indicator in indicators):
                detected_behaviors.append(behavior_type)
        
        # Detect pattern-specific avoidance
        
        # Quick topic changes (multiple topics in short text)
        sentences = text.split('.')
        if len(sentences) > 2 and len(text) < 200:
            detected_behaviors.append('rapid_topic_switching')
        
        # Question deflection (answering question with question)
        if text.count('?') > 1 and len(text) < 150:
            detected_behaviors.append('question_deflection')
        
        # Over-explanation (might indicate defensiveness)
        if len(text) > 300 and any(word in text_lower for word in ['because', 'since', 'due to']):
            detected_behaviors.append('over_explanation')
        
        return detected_behaviors
    
    def _analyze_emotional_displacement(self, text: str, emotional_state: Dict[str, float]) -> Dict[str, float]:
        """Analyze emotional displacement patterns"""
        displacement = {}
        
        # Check for emotional-language mismatch
        positive_words = {'great', 'wonderful', 'amazing', 'fantastic', 'perfect'}
        negative_emotions = {'sadness', 'anxiety', 'anger', 'frustration'}
        
        has_positive_words = any(word in text.lower() for word in positive_words)
        has_negative_emotion = any(emotional_state.get(emotion, 0) > 0.5 for emotion in negative_emotions)
        
        if has_positive_words and has_negative_emotion:
            displacement['emotional_masking'] = 0.7
        
        # Check for intensity mismatch
        text_intensity = len([word for word in text.split() if word.isupper()]) / max(1, len(text.split()))
        emotion_intensity = max(emotional_state.values()) if emotional_state else 0
        
        if abs(text_intensity - emotion_intensity) > 0.5:
            displacement['intensity_mismatch'] = abs(text_intensity - emotion_intensity)
        
        # Check for emotional flooding (too many emotional words)
        emotional_words = {'love', 'hate', 'amazing', 'terrible', 'wonderful', 'awful', 'perfect', 'horrible'}
        emotional_density = len([word for word in text.lower().split() if word in emotional_words]) / max(1, len(text.split()))
        
        if emotional_density > 0.2:
            displacement['emotional_flooding'] = emotional_density
        
        return displacement
    
    def _update_shadow_patterns(self, detected_patterns: Dict[str, float], text: str, context: str):
        """Update stored shadow patterns with new detections"""
        for pattern_name, strength in detected_patterns.items():
            if pattern_name in self.shadow_patterns:
                # Update existing pattern
                pattern = self.shadow_patterns[pattern_name]
                pattern.last_reinforced = datetime.now()
                pattern.occurrence_count += 1
                pattern.manifestation_strength = min(1.0, pattern.manifestation_strength + (strength * 0.1))
                
                # Add context if new
                if context not in pattern.context_clusters:
                    pattern.context_clusters.append(context)
            else:
                # Create new pattern
                pattern_config = self.pattern_detectors[pattern_name]
                self.shadow_patterns[pattern_name] = ShadowPattern(
                    theme_type=pattern_config['type'],
                    pattern_name=pattern_name,
                    keywords=pattern_config['keywords'],
                    avoidance_signals=pattern_config['avoidance_signals'],
                    manifestation_strength=strength,
                    first_detected=datetime.now(),
                    last_reinforced=datetime.now(),
                    occurrence_count=1,
                    context_clusters=[context],
                    emotional_charge=strength,
                    suppression_indicators={}
                )
    
    def _generate_shadow_insights(self, detected_patterns: Dict[str, float], 
                                avoidance_behaviors: List[str]) -> Dict[str, Any]:
        """Generate insights about unconscious patterns for character awareness"""
        insights = {
            'active_suppressions': [],
            'emerging_themes': [],
            'avoidance_clusters': {},
            'breakthrough_opportunities': [],
            'gentle_acknowledgment_points': []
        }
        
        # Analyze active suppressions
        for pattern_name, strength in detected_patterns.items():
            if strength > 0.6:
                pattern = self.shadow_patterns.get(pattern_name)
                if pattern and pattern.occurrence_count > 3:
                    insights['active_suppressions'].append({
                        'theme': pattern.theme_type.value,
                        'pattern': pattern_name,
                        'strength': strength,
                        'frequency': pattern.occurrence_count
                    })
        
        # Identify emerging themes (new patterns with growing strength)
        for pattern_name, pattern in self.shadow_patterns.items():
            if (pattern.manifestation_strength > 0.4 and 
                pattern.occurrence_count >= 2 and 
                (datetime.now() - pattern.first_detected).days < 30):
                insights['emerging_themes'].append({
                    'theme': pattern.theme_type.value,
                    'pattern': pattern_name,
                    'emergence_speed': pattern.occurrence_count / max(1, (datetime.now() - pattern.first_detected).days)
                })
        
        # Cluster avoidance behaviors
        behavior_groups = defaultdict(list)
        for behavior in avoidance_behaviors:
            if 'deflection' in behavior:
                behavior_groups['deflection_cluster'].append(behavior)
            elif 'emotional' in behavior:
                behavior_groups['emotional_cluster'].append(behavior)
            elif 'time' in behavior:
                behavior_groups['temporal_cluster'].append(behavior)
        
        insights['avoidance_clusters'] = dict(behavior_groups)
        
        # Identify breakthrough opportunities (high emotion with low suppression)
        for pattern_name, pattern in self.shadow_patterns.items():
            if (pattern.emotional_charge > 0.7 and 
                pattern.manifestation_strength > 0.8 and
                len(pattern.suppression_indicators) < 3):
                insights['breakthrough_opportunities'].append({
                    'pattern': pattern_name,
                    'theme': pattern.theme_type.value,
                    'readiness_score': pattern.emotional_charge * pattern.manifestation_strength
                })
        
        # Generate gentle acknowledgment points
        insights['gentle_acknowledgment_points'] = self._generate_acknowledgment_points(detected_patterns)
        
        return insights
    
    def _generate_acknowledgment_points(self, detected_patterns: Dict[str, float]) -> List[Dict[str, str]]:
        """Generate gentle ways for characters to acknowledge unconscious patterns"""
        acknowledgments = []
        
        acknowledgment_templates = {
            'intimacy_avoidance': [
                "I notice you light up when you talk about your independence...",
                "There's something beautiful about how self-contained you are...",
                "You have such a graceful way of keeping your own space..."
            ],
            'vulnerability_fear': [
                "Your strength is really remarkable, you know...",
                "I admire how you handle everything so capably...",
                "There's something powerful about your composure..."
            ],
            'creative_suppression': [
                "I can hear the artist in you, even when you're being practical...",
                "Your mind works in such interesting ways...",
                "There's a creative spark in how you see things..."
            ],
            'attachment_hunger': [
                "Even when you say it doesn't matter, I can feel how much it does...",
                "Your casual words carry such depth...",
                "I notice the care in your voice, even when you're being offhand..."
            ],
            'success_fear': [
                "I don't think that was as accidental as you make it sound...",
                "You have a gift for making difficult things look easy...",
                "There's real skill behind what you call luck..."
            ],
            'emotional_numbness': [
                "Sometimes going through the motions is how we protect ourselves...",
                "There's wisdom in your steadiness...",
                "I sense there's more feeling there than you let on..."
            ],
            'past_trauma_echoes': [
                "Some stories live in our hearts even when we don't speak them...",
                "The past has a way of shaping us in beautiful ways...",
                "I can feel the strength that came from whatever you've been through..."
            ]
        }
        
        for pattern_name, strength in detected_patterns.items():
            if pattern_name in acknowledgment_templates and strength > 0.5:
                templates = acknowledgment_templates[pattern_name]
                acknowledgments.append({
                    'pattern': pattern_name,
                    'acknowledgment': templates[0],  # Use first template for now
                    'strength': strength,
                    'approach': 'gentle_reflection'
                })
        
        return acknowledgments
    
    def _generate_character_hints(self, shadow_insights: Dict[str, Any]) -> Dict[str, List[str]]:
        """Generate hints for different character types on how to respond to shadow patterns"""
        hints = {
            'mia': [],  # Empathetic, nurturing responses
            'solene': [],  # Challenging but supportive responses
            'lyra': [],  # Mystical, intuitive responses
            'doc': []  # Therapeutic, professional responses
        }
        
        # Generate Mia's empathetic responses
        for suppression in shadow_insights.get('active_suppressions', []):
            if suppression['theme'] == 'vulnerability_fear':
                hints['mia'].append("I see your strength, and I also see the tender heart it protects...")
            elif suppression['theme'] == 'intimacy_avoidance':
                hints['mia'].append("Your independence is beautiful... and so is connection when you're ready...")
        
        # Generate Solene's challenging responses
        for suppression in shadow_insights.get('active_suppressions', []):
            if suppression['theme'] == 'creative_suppression':
                hints['solene'].append("You keep calling yourself practical, but I hear the artist fighting to break free...")
            elif suppression['theme'] == 'success_fear':
                hints['solene'].append("Stop diminishing yourself. That wasn't luck - that was you being brilliant.")
        
        # Generate Lyra's mystical responses
        for opportunity in shadow_insights.get('breakthrough_opportunities', []):
            if opportunity['theme'] == 'repressed_memory':
                hints['lyra'].append("The universe remembers what we choose to forget... and it's gentle with our healing...")
            elif opportunity['theme'] == 'suppressed_desire':
                hints['lyra'].append("I can feel the dreams you've tucked away... they're still alive, you know...")
        
        # Generate Doc's therapeutic responses
        for theme in shadow_insights.get('emerging_themes', []):
            if theme['theme'] == 'emotional_numbness':
                hints['doc'].append("Numbness often serves a purpose... it's okay to feel protected while you heal...")
            elif theme['theme'] == 'attachment_hunger':
                hints['doc'].append("Needing connection isn't weakness - it's human. Your attachment needs are valid...")
        
        return hints
    
    def get_shadow_summary(self) -> Dict[str, Any]:
        """Get a summary of current shadow patterns for the user"""
        active_patterns = {name: pattern for name, pattern in self.shadow_patterns.items() 
                         if pattern.manifestation_strength > 0.3}
        
        theme_distribution = Counter([pattern.theme_type.value for pattern in active_patterns.values()])
        
        strongest_pattern = max(active_patterns.items(), 
                              key=lambda x: x[1].manifestation_strength) if active_patterns else None
        
        return {
            'total_patterns': len(self.shadow_patterns),
            'active_patterns': len(active_patterns),
            'theme_distribution': dict(theme_distribution),
            'strongest_pattern': {
                'name': strongest_pattern[0],
                'type': strongest_pattern[1].theme_type.value,
                'strength': strongest_pattern[1].manifestation_strength
            } if strongest_pattern else None,
            'recent_events_count': len([event for event in self.shadow_events 
                                      if (datetime.now() - event.timestamp).days < 7]),
            'patterns_by_strength': [
                {
                    'name': name,
                    'type': pattern.theme_type.value,
                    'strength': pattern.manifestation_strength,
                    'frequency': pattern.occurrence_count
                }
                for name, pattern in sorted(active_patterns.items(), 
                                          key=lambda x: x[1].manifestation_strength, reverse=True)
            ]
        }
    
    def _load_shadow_memory(self):
        """Load shadow memory from storage"""
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Load shadow patterns
                if 'patterns' in data:
                    for pattern_name, pattern_data in data['patterns'].items():
                        self.shadow_patterns[pattern_name] = ShadowPattern(
                            theme_type=ShadowThemeType(pattern_data['theme_type']),
                            pattern_name=pattern_data['pattern_name'],
                            keywords=set(pattern_data['keywords']),
                            avoidance_signals=set(pattern_data['avoidance_signals']),
                            manifestation_strength=pattern_data['manifestation_strength'],
                            first_detected=datetime.fromisoformat(pattern_data['first_detected']),
                            last_reinforced=datetime.fromisoformat(pattern_data['last_reinforced']),
                            occurrence_count=pattern_data['occurrence_count'],
                            context_clusters=pattern_data['context_clusters'],
                            emotional_charge=pattern_data['emotional_charge'],
                            suppression_indicators=pattern_data.get('suppression_indicators', {})
                        )
                
                # Load recent shadow events (last 50 for memory efficiency)
                if 'events' in data:
                    events_data = data['events'][-50:]
                    self.shadow_events = [
                        ShadowEvent(
                            timestamp=datetime.fromisoformat(event['timestamp']),
                            trigger_text=event['trigger_text'],
                            detected_patterns=event['detected_patterns'],
                            avoidance_behaviors=event['avoidance_behaviors'],
                            emotional_displacement=event['emotional_displacement'],
                            context=event['context']
                        ) for event in events_data
                    ]
                    
            except Exception as e:
                print(f"Error loading shadow memory for user {self.user_id}: {e}")
    
    def _save_shadow_memory(self):
        """Save shadow memory to storage"""
        try:
            os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
            
            data = {
                'patterns': {
                    name: {
                        'theme_type': pattern.theme_type.value,
                        'pattern_name': pattern.pattern_name,
                        'keywords': list(pattern.keywords),
                        'avoidance_signals': list(pattern.avoidance_signals),
                        'manifestation_strength': pattern.manifestation_strength,
                        'first_detected': pattern.first_detected.isoformat(),
                        'last_reinforced': pattern.last_reinforced.isoformat(),
                        'occurrence_count': pattern.occurrence_count,
                        'context_clusters': pattern.context_clusters,
                        'emotional_charge': pattern.emotional_charge,
                        'suppression_indicators': pattern.suppression_indicators
                    } for name, pattern in self.shadow_patterns.items()
                },
                'events': [
                    {
                        'timestamp': event.timestamp.isoformat(),
                        'trigger_text': event.trigger_text,
                        'detected_patterns': event.detected_patterns,
                        'avoidance_behaviors': event.avoidance_behaviors,
                        'emotional_displacement': event.emotional_displacement,
                        'context': event.context
                    } for event in self.shadow_events[-50:]  # Save only recent events
                ]
            }
            
            with open(self.storage_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            print(f"Error saving shadow memory for user {self.user_id}: {e}")

# Factory function
def create_shadow_memory_layer(user_id: str) -> ShadowMemoryLayer:
    """Create a shadow memory layer for a specific user"""
    return ShadowMemoryLayer(user_id)

# Integration helper
def analyze_user_communication_for_shadows(user_id: str, text: str, context: str, 
                                         emotional_state: Dict[str, float]) -> Dict[str, Any]:
    """
    Analyze user communication for shadow patterns and return insights
    
    This function can be called from conversation processing to detect
    unconscious patterns and provide character response hints.
    """
    shadow_layer = create_shadow_memory_layer(user_id)
    return shadow_layer.analyze_communication(text, context, emotional_state)

# Integration Example
from modules.memory.shadow_memory import analyze_user_communication_for_shadows

shadow_analysis = analyze_user_communication_for_shadows(
    user_id=user.id,
    text=user_message,
    context=conversation_context,
    emotional_state=current_emotion
)

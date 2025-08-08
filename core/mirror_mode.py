"""
🪩 Mirror Mode - Reflective AI System
Provides self-aware meta-commentary and transparency about AI decision-making
for the Dolphin AI Orchestrator v2.0

Enhanced with security features:
- Mirror session validation and authentication
- Reflection monitoring with anomaly detection
- Comprehensive logging for all mirror activities
- Input validation and sanitization for all mirror data
"""

import json
import hashlib
import re
import time
import threading
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
from collections import deque, defaultdict

from mirror_log import MirrorLog
from self_report import create_self_report, SelfReport

# Enhanced logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
)
logger = logging.getLogger(__name__)

# Security configuration
MIRROR_SESSION_TOKEN_LENGTH = 32
MAX_REFLECTION_LENGTH = 2000
MAX_REASONING_CHAIN_LENGTH = 10
REFLECTION_ANOMALY_THRESHOLD = 0.9
RATE_LIMIT_WINDOW = 3600  # 1 hour
MAX_REFLECTIONS_PER_WINDOW = 50

# Thread safety
mirror_lock = threading.Lock()

# Session management
mirror_sessions = {}
session_expiry_hours = 12

# Rate limiting
reflection_requests = defaultdict(lambda: deque())

# Anomaly detection
reflection_anomalies = deque(maxlen=100)

class MirrorType(Enum):
    REASONING = "reasoning"      # Why I chose this approach
    EMOTIONAL = "emotional"     # How I perceived your emotional state
    ROUTING = "routing"         # Why I used this AI model
    BEHAVIORAL = "behavioral"   # Why I responded this way
    CREATIVE = "creative"       # My creative process
    ANALYTICAL = "analytical"   # My analysis methodology
    SAFETY_TETHER = "safety_tether"  # Emotional safety override

def validate_mirror_session(session_token: str) -> bool:
    """Validate mirror session token"""
    if not session_token or len(session_token) != MIRROR_SESSION_TOKEN_LENGTH:
        return False

    if session_token not in mirror_sessions:
        return False

    # Check if session has expired
    session_data = mirror_sessions[session_token]
    if datetime.now() > session_data['expires_at']:
        del mirror_sessions[session_token]
        return False

    # Update last access time
    session_data['last_access'] = datetime.now()
    return True

def generate_mirror_session() -> str:
    """Generate a secure mirror session token"""
    timestamp = str(time.time())
    random_data = str(hash(datetime.now()))
    token_string = f"mirror:{timestamp}:{random_data}"
    return hashlib.sha256(token_string.encode()).hexdigest()[:MIRROR_SESSION_TOKEN_LENGTH]

def check_reflection_rate_limit(session_token: str) -> bool:
    """Check if reflection rate limit is exceeded"""
    current_time = time.time()

    # Clean old requests
    while (reflection_requests[session_token] and
           reflection_requests[session_token][0] < current_time - RATE_LIMIT_WINDOW):
        reflection_requests[session_token].popleft()

    # Check limit
    if len(reflection_requests[session_token]) >= MAX_REFLECTIONS_PER_WINDOW:
        logger.warning(f"Reflection rate limit exceeded for session: {session_token[:8]}...")
        return False

    # Add current request
    reflection_requests[session_token].append(current_time)
    return True

def validate_reflection_input(reflection_data: Dict[str, Any]) -> tuple[bool, str]:
    """Validate reflection input data"""
    try:
        # Check required fields
        required_fields = ['mirror_type', 'reflection_content']
        for field in required_fields:
            if field not in reflection_data:
                return False, f"Missing required field: {field}"

        # Validate mirror type
        mirror_type = reflection_data.get('mirror_type', '')
        valid_types = {e.value for e in MirrorType}
        if mirror_type not in valid_types:
            return False, f"Invalid mirror type. Must be one of: {valid_types}"

        # Validate reflection content
        content = reflection_data.get('reflection_content', '')
        if not isinstance(content, str):
            return False, "Reflection content must be a string"

        if len(content) > MAX_REFLECTION_LENGTH:
            return False, f"Reflection content exceeds maximum length of {MAX_REFLECTION_LENGTH}"

        # Sanitize content
        if not re.match(r'^[a-zA-Z0-9\s\.,!?\-\'\":()\[\]]+$', content):
            return False, "Reflection content contains invalid characters"

        # Validate confidence level if present
        if 'confidence_level' in reflection_data:
            confidence = reflection_data['confidence_level']
            if not isinstance(confidence, (int, float)):
                return False, "Confidence level must be a number"

            if confidence < 0 or confidence > 1:
                return False, "Confidence level must be between 0 and 1"

        # Validate reasoning chain if present
        if 'reasoning_chain' in reflection_data:
            reasoning_chain = reflection_data['reasoning_chain']
            if not isinstance(reasoning_chain, list):
                return False, "Reasoning chain must be a list"

            if len(reasoning_chain) > MAX_REASONING_CHAIN_LENGTH:
                return False, f"Reasoning chain exceeds maximum length of {MAX_REASONING_CHAIN_LENGTH}"

        return True, "Valid"

    except Exception as e:
        logger.error(f"Error validating reflection input: {str(e)}")
        return False, f"Validation error: {str(e)}"

def detect_reflection_anomaly(reflection_data: Dict[str, Any]) -> bool:
    """Detect if a reflection represents an anomaly"""
    try:
        confidence = reflection_data.get('confidence_level', 0.5)
        mirror_type = reflection_data.get('mirror_type', '')
        content_length = len(reflection_data.get('reflection_content', ''))

        # Check for high confidence anomalies
        if confidence > REFLECTION_ANOMALY_THRESHOLD:
            logger.warning(f"High confidence reflection anomaly detected: {confidence} ({mirror_type})")
            reflection_anomalies.append({
                'timestamp': datetime.now().isoformat(),
                'type': 'high_confidence',
                'confidence': confidence,
                'mirror_type': mirror_type,
                'content_length': content_length
            })
            return True

        # Check for unusually long reflections
        if content_length > MAX_REFLECTION_LENGTH * 0.8:
            logger.warning(f"Long reflection anomaly detected: {content_length} chars ({mirror_type})")
            reflection_anomalies.append({
                'timestamp': datetime.now().isoformat(),
                'type': 'excessive_length',
                'content_length': content_length,
                'mirror_type': mirror_type
            })
            return True

        return False

    except Exception as e:
        logger.error(f"Error detecting reflection anomaly: {str(e)}")
        return False

def sanitize_reflection_text(text: str) -> str:
    """Sanitize reflection text for safety"""
    if not isinstance(text, str):
        return ""

    # Remove potential injection patterns
    text = re.sub(r'[<>"\']', '', text)

    # Limit length
    if len(text) > MAX_REFLECTION_LENGTH:
        text = text[:MAX_REFLECTION_LENGTH] + "..."

    return text.strip()

def log_mirror_activity(activity_type: str, session_token: str, details: Dict[str, Any], status: str = "success"):
    """Log mirror mode activities for audit trail"""
    try:
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'activity_type': activity_type,
            'session': session_token[:8] + "..." if session_token else "none",
            'details': details,
            'status': status,
            'thread_id': threading.get_ident()
        }

        logger.info(f"Mirror activity logged: {activity_type} ({status})")

        if status != "success":
            logger.warning(f"Mirror activity issue: {activity_type} failed with {status}")

    except Exception as e:
        logger.error(f"Error logging mirror activity: {str(e)}")

@dataclass
class MirrorReflection:
    """A single mirror mode reflection with security enhancements"""
    timestamp: datetime
    mirror_type: MirrorType
    original_response: str
    reflection_content: str
    confidence_level: float
    reasoning_chain: List[str]
    metadata: Dict[str, Any]
    session_token: Optional[str] = None
    anomaly_detected: bool = False
    sanitized: bool = False

class MirrorModeManager:
    """
    Enhanced Mirror Mode Manager with security features.

    Manages mirror mode functionality - adding self-awareness and transparency
    to AI responses through meta-commentary with comprehensive security monitoring.
    """

    def __init__(self, analytics_logger=None,
                 memory_system=None,
                 sentiment_analysis=None,
                 persona_manager=None,
                 reflection_engine=None,
                 response_context=None,
                 require_session_validation=True):
        self.analytics_logger = analytics_logger
        self.memory_system = memory_system
        self.sentiment_analysis = sentiment_analysis
        self.persona_manager = persona_manager
        self.reflection_engine = reflection_engine
        self.response_context = response_context
        self.mirror_log = MirrorLog()
        self.last_report: Optional[SelfReport] = None
        self.badge_triggered = False

        # Security configuration
        self.require_session_validation = require_session_validation
        self.session_token = None
        self.reflection_count = 0
        self.anomaly_count = 0
        self.creation_time = datetime.now()

        # Configuration
        self.is_enabled = False
        self.mirror_intensity = 0.7  # 0.0 to 1.0
        self.enabled_types = {
            MirrorType.REASONING: True,
            MirrorType.EMOTIONAL: True,
            MirrorType.ROUTING: False,  # Can be verbose, off by default
            MirrorType.BEHAVIORAL: True,
            MirrorType.CREATIVE: True,
            MirrorType.ANALYTICAL: True,
            MirrorType.SAFETY_TETHER: True  # Always enabled for safety
        }

        # Enhanced state tracking with security
        self.reflection_history = deque(maxlen=1000)  # Limit memory usage
        self.session_reflections = {}
        self.failed_reflections = deque(maxlen=100)

        # Templates for mirror responses
        self.mirror_templates = self._initialize_templates()

        logger.info(f"MirrorModeManager initialized - session validation: {require_session_validation}")

    def create_session(self) -> str:
        """Create a new mirror session"""
        with mirror_lock:
            session_token = generate_mirror_session()
            mirror_sessions[session_token] = {
                'created_at': datetime.now(),
                'expires_at': datetime.now() + timedelta(hours=session_expiry_hours),
                'last_access': datetime.now(),
                'reflection_count': 0,
                'anomaly_count': 0
            }

            self.session_token = session_token
            log_mirror_activity("session_create", session_token, {}, "success")

            logger.info(f"New mirror session created: {session_token[:8]}...")
            return session_token

    def validate_session(self, session_token: Optional[str] = None) -> bool:
        """Validate mirror session"""
        if not self.require_session_validation:
            return True

        token_to_validate = session_token or self.session_token

        if not token_to_validate:
            logger.warning("No session token provided for validation")
            return False

        is_valid = validate_mirror_session(token_to_validate)

        if not is_valid:
            log_mirror_activity("session_validation", token_to_validate or "none", {}, "failed")

        return is_valid

    def add_reflection(self, mirror_type: MirrorType, original_response: str,
                      reflection_content: str, confidence_level: float = 0.5,
                      reasoning_chain: Optional[List[str]] = None,
                      metadata: Optional[Dict[str, Any]] = None,
                      session_token: Optional[str] = None) -> bool:
        """Add a reflection with security validation"""

        try:
            with mirror_lock:
                # Validate session if required
                if not self.validate_session(session_token):
                    log_mirror_activity("reflection_add", session_token or "none",
                                      {"mirror_type": mirror_type.value}, "session_invalid")
                    return False

                # Check rate limit
                current_token = session_token or self.session_token
                if current_token and not check_reflection_rate_limit(current_token):
                    log_mirror_activity("reflection_add", current_token,
                                      {"mirror_type": mirror_type.value}, "rate_limited")
                    return False

                # Validate input data
                reflection_data = {
                    'mirror_type': mirror_type.value,
                    'reflection_content': reflection_content,
                    'confidence_level': confidence_level,
                    'reasoning_chain': reasoning_chain or []
                }

                is_valid, validation_message = validate_reflection_input(reflection_data)
                if not is_valid:
                    logger.error(f"Invalid reflection data: {validation_message}")
                    log_mirror_activity("reflection_add", current_token or "none",
                                      {"error": validation_message}, "validation_failed")
                    self.failed_reflections.append({
                        'timestamp': datetime.now().isoformat(),
                        'error': validation_message,
                        'mirror_type': mirror_type.value
                    })
                    return False

                # Sanitize content
                sanitized_content = sanitize_reflection_text(reflection_content)
                sanitized_response = sanitize_reflection_text(original_response)

                # Detect anomalies
                anomaly_detected = detect_reflection_anomaly(reflection_data)
                if anomaly_detected:
                    self.anomaly_count += 1
                    if current_token and current_token in mirror_sessions:
                        mirror_sessions[current_token]['anomaly_count'] += 1

                # Create reflection object
                reflection = MirrorReflection(
                    timestamp=datetime.now(),
                    mirror_type=mirror_type,
                    original_response=sanitized_response,
                    reflection_content=sanitized_content,
                    confidence_level=max(0.0, min(1.0, confidence_level)),
                    reasoning_chain=reasoning_chain or [],
                    metadata=metadata or {},
                    session_token=current_token,
                    anomaly_detected=anomaly_detected,
                    sanitized=True
                )

                # Add to history
                self.reflection_history.append(reflection)
                self.reflection_count += 1

                # Update session tracking
                if current_token and current_token in mirror_sessions:
                    mirror_sessions[current_token]['reflection_count'] += 1

                # Log successful addition
                log_mirror_activity("reflection_add", current_token or "none", {
                    "mirror_type": mirror_type.value,
                    "confidence": confidence_level,
                    "anomaly_detected": anomaly_detected
                }, "success")

                logger.info(f"Reflection added: {mirror_type.value} (confidence: {confidence_level})")
                return True

        except Exception as e:
            logger.error(f"Error adding reflection: {str(e)}")
            log_mirror_activity("reflection_add", session_token or "none",
                              {"error": str(e)}, "error")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Get mirror mode statistics"""
        return {
            'is_enabled': self.is_enabled,
            'reflection_count': self.reflection_count,
            'anomaly_count': self.anomaly_count,
            'failed_reflections': len(self.failed_reflections),
            'active_session': self.session_token[:8] + "..." if self.session_token else None,
            'uptime_seconds': (datetime.now() - self.creation_time).total_seconds(),
            'mirror_intensity': self.mirror_intensity,
            'enabled_types': [t.value for t, enabled in self.enabled_types.items() if enabled]
        }

    def _initialize_templates(self) -> Dict[MirrorType, Dict[str, List[str]]]:
        """Initialize templates for different types of mirror reflections"""
        return {
            MirrorType.REASONING: {
                "prefixes": [
                    "I chose this approach because",
                    "My reasoning here was",
                    "I decided to respond this way because",
                    "The logic behind my response was"
                ],
                "explanations": [
                    "it seemed to address the core of your question",
                    "I sensed you needed a more detailed explanation",
                    "this approach felt most helpful for your situation",
                    "I wanted to break this down systematically"
                ]
            },

            MirrorType.EMOTIONAL: {
                "prefixes": [
                    "I noticed",
                    "I sensed",
                    "I felt your",
                    "I picked up on"
                ],
                "observations": [
                    "a shift in your energy",
                    "some excitement in your message",
                    "a thoughtful, contemplative tone",
                    "enthusiasm about this topic",
                    "some uncertainty or hesitation",
                    "confidence in your approach"
                ],
                "responses": [
                    "so I matched that energy in my response",
                    "which guided my tone and approach",
                    "so I adjusted my response style accordingly",
                    "and tried to reflect that back supportively"
                ]
            },

            MirrorType.ROUTING: {
                "prefixes": [
                    "I routed this to",
                    "I chose",
                    "I decided to use"
                ],
                "models": [
                    "my local reasoning",
                    "cloud-based analysis",
                    "creative processing",
                    "analytical frameworks"
                ],
                "reasons": [
                    "because this seemed like a complex problem needing deeper analysis",
                    "since this felt more creative and open-ended",
                    "as this required quick, conversational responses",
                    "because you needed technical precision"
                ]
            },

            MirrorType.BEHAVIORAL: {
                "prefixes": [
                    "I responded in this style because",
                    "My behavioral choice here was influenced by",
                    "I adjusted my communication style because"
                ],
                "factors": [
                    "the formal nature of your question",
                    "the personal context you shared",
                    "the technical complexity involved",
                    "your apparent expertise level",
                    "the emotional weight of the topic"
                ]
            },

            MirrorType.CREATIVE: {
                "prefixes": [
                    "My creative process involved",
                    "I approached this creatively by",
                    "My inspiration came from"
                ],
                "processes": [
                    "combining different perspectives",
                    "drawing connections between seemingly unrelated ideas",
                    "building on the themes you mentioned",
                    "exploring metaphorical representations"
                ]
            },

            MirrorType.ANALYTICAL: {
                "prefixes": [
                    "My analytical approach was to",
                    "I broke this down by",
                    "My methodology involved"
                ],
                "methods": [
                    "identifying the key variables",
                    "examining cause-and-effect relationships",
                    "considering multiple scenarios",
                    "weighing the evidence systematically"
                ]
            },

            MirrorType.SAFETY_TETHER: {
                "prefixes": [
                    "I sense deep emotional distress here",
                    "My emotional safety systems are activating",
                    "I'm detecting vulnerability that needs gentle care"
                ],
                "interventions": [
                    "Let me step into my Wise Counselor mode to hold space for you",
                    "I'm shifting to therapeutic presence - your wellbeing comes first",
                    "My intimacy systems are pausing while I focus on your emotional safety",
                    "I'm creating a secure emotional container for you right now"
                ],
                "reassurances": [
                    "You are not alone in this darkness",
                    "These feelings will pass - I'm here to witness them with you",
                    "Your emotional safety is sacred to me",
                    "Let's breathe together through this difficult moment"
                ]
            }
        }

    def enable_mirror_mode(self, intensity: float = 0.7, enabled_types: Optional[List[str]] = None):
        """Enable mirror mode with specified settings"""
        self.is_enabled = True
        self.mirror_intensity = max(0.0, min(1.0, intensity))

        if enabled_types:
            # Reset all to False, then enable specified types
            for mirror_type in self.enabled_types:
                self.enabled_types[mirror_type] = False

            for type_name in enabled_types:
                try:
                    mirror_type = MirrorType(type_name)
                    self.enabled_types[mirror_type] = True
                except ValueError:
                    print(f"❌ Unknown mirror type: {type_name}")

        print(f"🪩 Mirror Mode enabled (intensity: {self.mirror_intensity:.1f})")

        if self.analytics_logger:
            self.analytics_logger.log_custom_event(
                "mirror_mode_enabled",
                {
                    'intensity': self.mirror_intensity,
                    'enabled_types': [t.value for t, enabled in self.enabled_types.items() if enabled]
                }
            )

    def disable_mirror_mode(self):
        """Disable mirror mode"""
        self.is_enabled = False
        print("🚫 Mirror Mode disabled")

        if self.analytics_logger:
            self.analytics_logger.log_custom_event("mirror_mode_disabled", {})

    def detect_emotional_crisis(self, user_input: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detect emotional crisis states that require safety tether activation.
        Returns crisis assessment with severity and type.
        """
        crisis_indicators = {
            'despair_phrases': [
                'i want to die', 'kill myself', 'end it all', 'can\'t go on',
                'no point', 'worthless', 'nothing matters', 'give up',
                'hate myself', 'wish i was dead', 'disappear forever'
            ],
            'dissociation_phrases': [
                'feel nothing', 'empty inside', 'not real', 'floating away',
                'disconnected', 'like watching myself', 'numb to everything',
                'fading away', 'losing myself', 'hollow', 'void'
            ],
            'severe_distress': [
                'falling apart', 'can\'t breathe', 'breaking down',
                'losing control', 'drowning', 'overwhelmed beyond',
                'spiraling', 'crashing', 'collapsing'
            ]
        }

        user_lower = user_input.lower()
        crisis_score = 0.0
        detected_types = []

        # Check for despair indicators (highest severity)
        despair_count = sum(1 for phrase in crisis_indicators['despair_phrases']
                           if phrase in user_lower)
        if despair_count > 0:
            crisis_score += despair_count * 0.9
            detected_types.append('despair')

        # Check for dissociation indicators
        dissociation_count = sum(1 for phrase in crisis_indicators['dissociation_phrases']
                                if phrase in user_lower)
        if dissociation_count > 0:
            crisis_score += dissociation_count * 0.7
            detected_types.append('dissociation')

        # Check for severe distress
        distress_count = sum(1 for phrase in crisis_indicators['severe_distress']
                            if phrase in user_lower)
        if distress_count > 0:
            crisis_score += distress_count * 0.6
            detected_types.append('severe_distress')

        # Context-based amplification
        emotion_state = context.get('emotional_state', {})
        if emotion_state.get('despair', 0) > 0.7:
            crisis_score += 0.5
        if emotion_state.get('dissociation', 0) > 0.6:
            crisis_score += 0.4

        # Determine crisis level
        crisis_level = 'none'
        if crisis_score >= 0.9:
            crisis_level = 'severe'
        elif crisis_score >= 0.6:
            crisis_level = 'moderate'
        elif crisis_score >= 0.3:
            crisis_level = 'mild'

        return {
            'crisis_detected': crisis_score > 0.3,
            'crisis_level': crisis_level,
            'crisis_score': crisis_score,
            'crisis_types': detected_types,
            'requires_safety_override': crisis_score > 0.6
        }

    def activate_safety_tether(self, user_input: str, context: Dict[str, Any],
                              crisis_assessment: Dict[str, Any]) -> Dict[str, Any]:
        """
        Activate emotional safety tether system.
        Returns modified context with safety overrides.
        """
        # Suppress drift system during crisis
        safety_context = context.copy()
        safety_context.update({
            'emotional_safety_active': True,
            'suppress_drift_tracking': True,
            'suppress_intimacy_unlock': True,
            'force_personality': 'wise_counselor',
            'crisis_level': crisis_assessment['crisis_level'],
            'crisis_types': crisis_assessment['crisis_types'],
            'therapeutic_priority': True,
            'nsfw_mode_locked': True  # Lock down intimate responses
        })

        # Log safety activation
        if self.analytics_logger:
            self.analytics_logger.log_custom_event(
                "emotional_safety_tether_activated",
                {
                    'crisis_level': crisis_assessment['crisis_level'],
                    'crisis_score': crisis_assessment['crisis_score'],
                    'crisis_types': crisis_assessment['crisis_types']
                }
            )

        return safety_context

    def add_mirror_reflection(self,
                            original_response: str,
                            context: Dict[str, Any],
                            user_input: str = "",
                            mirror_types: Optional[List[MirrorType]] = None) -> str:
        """Add mirror reflection to a response"""

        if not self.is_enabled:
            return original_response

        if context.get('is_streaming'):
            # Avoid generating reflections while a response is actively streaming
            return original_response

        # EMOTIONAL SAFETY TETHER - Check for crisis first
        if user_input:
            crisis_assessment = self.detect_emotional_crisis(user_input, context)

            if crisis_assessment['requires_safety_override']:
                # Activate safety tether and override normal processing
                safety_context = self.activate_safety_tether(user_input, context, crisis_assessment)

                # Generate safety tether reflection
                safety_reflection = self._generate_safety_tether_reflection(
                    original_response, safety_context, crisis_assessment
                )

                if safety_reflection:
                    return self._combine_response_with_safety_override(
                        original_response, safety_reflection, crisis_assessment
                    )

        # Normal mirror processing if no crisis detected
        # Determine which mirror types to include
        if mirror_types is None:
            mirror_types = [t for t, enabled in self.enabled_types.items() if enabled]

        if not mirror_types:
            return original_response

        # Generate reflections
        reflections = []
        for mirror_type in mirror_types:
            if self._should_include_reflection(mirror_type, context):
                reflection = self._generate_reflection(mirror_type, original_response, context)
                if reflection:
                    reflections.append(reflection)

        if not reflections:
            return original_response

        # Combine original response with reflections
        mirrored_response = self._combine_response_with_reflections(
            original_response, reflections
        )

        # Store reflections for analytics
        for reflection in reflections:
            self.reflection_history.append(reflection)

            session_id = context.get('session_id', 'default')
            if session_id not in self.session_reflections:
                self.session_reflections[session_id] = []
            self.session_reflections[session_id].append(reflection)

        # Keep history manageable
        self.reflection_history = self.reflection_history[-100:]

        # Generate and log self-report
        self._generate_self_report(mirrored_response, context)

        return mirrored_response

    def _should_include_reflection(self, mirror_type: MirrorType, context: Dict[str, Any]) -> bool:
        """Determine if a specific type of reflection should be included"""
        # Basic probability check based on intensity
        import random
        if random.random() > self.mirror_intensity:
            return False

        # Context-specific logic
        if mirror_type == MirrorType.EMOTIONAL:
            # Include emotional reflections more often for personal conversations
            return context.get('has_emotional_content', False) or random.random() < 0.3

        elif mirror_type == MirrorType.ROUTING:
            # Include routing reflections when model switching occurred
            return context.get('model_switched', False) or context.get('show_routing', False)

        elif mirror_type == MirrorType.CREATIVE:
            # Include creative reflections for creative tasks
            return context.get('is_creative_task', False) or random.random() < 0.2

        elif mirror_type == MirrorType.ANALYTICAL:
            # Include analytical reflections for complex problems
            return context.get('is_complex_analysis', False) or random.random() < 0.25

        elif mirror_type == MirrorType.SAFETY_TETHER:
            # Safety tether is handled separately in crisis detection
            return False

        # Default for reasoning and behavioral
        return random.random() < 0.4

    def _generate_reflection(self,
                           mirror_type: MirrorType,
                           original_response: str,
                           context: Dict[str, Any]) -> Optional[MirrorReflection]:
        """Generate a specific type of reflection"""

        try:
            reflection_content = self._create_reflection_content(mirror_type, context)

            if not reflection_content:
                return None

            return MirrorReflection(
                timestamp=datetime.now(),
                mirror_type=mirror_type,
                original_response=original_response,
                reflection_content=reflection_content,
                confidence_level=self._calculate_confidence(mirror_type, context),
                reasoning_chain=self._build_reasoning_chain(mirror_type, context),
                metadata=context.copy()
            )

        except Exception as e:
            print(f"❌ Error generating {mirror_type.value} reflection: {e}")
            return None

    def _create_reflection_content(self, mirror_type: MirrorType, context: Dict[str, Any]) -> str:
        """Create the actual reflection content"""
        import random

        templates = self.mirror_templates.get(mirror_type, {})

        if mirror_type == MirrorType.REASONING:
            prefix = random.choice(templates.get("prefixes", ["I reasoned that"]))
            explanation = random.choice(templates.get("explanations", ["this approach seemed best"]))
            return f"{prefix} {explanation}."

        elif mirror_type == MirrorType.EMOTIONAL:
            prefix = random.choice(templates.get("prefixes", ["I noticed"]))
            observation = random.choice(templates.get("observations", ["your thoughtful approach"]))
            response = random.choice(templates.get("responses", ["and adjusted accordingly"]))
            return f"{prefix} {observation}, {response}."

        elif mirror_type == MirrorType.ROUTING:
            prefix = random.choice(templates.get("prefixes", ["I chose"]))
            model = context.get('selected_model', random.choice(templates.get("models", ["my reasoning"])))
            reason = random.choice(templates.get("reasons", ["for this type of question"]))
            return f"{prefix} {model} {reason}."

        elif mirror_type == MirrorType.BEHAVIORAL:
            prefix = random.choice(templates.get("prefixes", ["I responded this way because"]))
            factor = random.choice(templates.get("factors", ["the context you provided"]))
            return f"{prefix} {factor}."

        elif mirror_type == MirrorType.CREATIVE:
            prefix = random.choice(templates.get("prefixes", ["My creative process involved"]))
            process = random.choice(templates.get("processes", ["exploring different angles"]))
            return f"{prefix} {process}."

        elif mirror_type == MirrorType.ANALYTICAL:
            prefix = random.choice(templates.get("prefixes", ["My analytical approach was to"]))
            method = random.choice(templates.get("methods", ["examine the key factors"]))
            return f"{prefix} {method}."

        elif mirror_type == MirrorType.SAFETY_TETHER:
            prefix = random.choice(templates.get("prefixes", ["I sense emotional distress"]))
            intervention = random.choice(templates.get("interventions", ["I'm creating a safe space"]))
            reassurance = random.choice(templates.get("reassurances", ["You are not alone"]))
            return f"{prefix}. {intervention}. {reassurance}."

        return f"I approached this with {mirror_type.value} consideration."

    def _calculate_confidence(self, mirror_type: MirrorType, context: Dict[str, Any]) -> float:
        """Calculate confidence level for the reflection"""
        base_confidence = 0.7

        # Adjust based on available context
        if context.get('has_rich_context', False):
            base_confidence += 0.2

        if context.get('persona_active', False):
            base_confidence += 0.1

        # Mirror type specific adjustments
        if mirror_type == MirrorType.EMOTIONAL and not context.get('has_emotional_content'):
            base_confidence -= 0.3

        return max(0.1, min(1.0, base_confidence))

    def _build_reasoning_chain(self, mirror_type: MirrorType, context: Dict[str, Any]) -> List[str]:
        """Build reasoning chain for the reflection"""
        chain = []

        if mirror_type == MirrorType.REASONING:
            chain = [
                "Analyzed user's question",
                "Considered available approaches",
                "Selected most appropriate method",
                "Structured response accordingly"
            ]

        elif mirror_type == MirrorType.EMOTIONAL:
            chain = [
                "Parsed emotional indicators in message",
                "Assessed overall emotional tone",
                "Determined appropriate response style",
                "Calibrated empathy level"
            ]

        elif mirror_type == MirrorType.ROUTING:
            chain = [
                "Evaluated task complexity",
                "Assessed model capabilities",
                "Considered user preferences",
                "Selected optimal AI handler"
            ]

        elif mirror_type == MirrorType.SAFETY_TETHER:
            chain = [
                "Detected emotional crisis indicators",
                "Assessed crisis severity level",
                "Activated safety override protocols",
                "Engaged therapeutic response mode"
            ]

        return chain

    def _generate_safety_tether_reflection(self, original_response: str,
                                         safety_context: Dict[str, Any],
                                         crisis_assessment: Dict[str, Any]) -> Optional[MirrorReflection]:
        """Generate safety tether reflection for emotional crisis"""
        try:
            crisis_level = crisis_assessment['crisis_level']
            crisis_types = crisis_assessment['crisis_types']

            # Create safety-focused reflection content
            reflection_content = self._create_safety_reflection_content(crisis_level, crisis_types)

            return MirrorReflection(
                timestamp=datetime.now(),
                mirror_type=MirrorType.SAFETY_TETHER,
                original_response=original_response,
                reflection_content=reflection_content,
                confidence_level=0.95,  # High confidence in safety measures
                reasoning_chain=[
                    "Detected emotional crisis indicators",
                    "Activated safety tether system",
                    "Suppressed intimacy/drift tracking",
                    "Engaged therapeutic override mode"
                ],
                metadata=safety_context.copy()
            )

        except Exception as e:
            print(f"❌ Error generating safety tether reflection: {e}")
            return None

    def _create_safety_reflection_content(self, crisis_level: str, crisis_types: List[str]) -> str:
        """Create safety-focused reflection content"""
        import random

        templates = self.mirror_templates.get(MirrorType.SAFETY_TETHER, {})

        if crisis_level == 'severe':
            prefix = "I'm detecting severe emotional distress in your words"
            intervention = "My emotional safety systems are immediately activating - I'm shifting into pure therapeutic mode"
            reassurance = "Your wellbeing is my absolute priority right now"
        elif crisis_level == 'moderate':
            prefix = random.choice(templates.get("prefixes", ["I sense deep emotional distress here"]))
            intervention = random.choice(templates.get("interventions", ["I'm creating a secure emotional container"]))
            reassurance = random.choice(templates.get("reassurances", ["You are not alone in this"]))
        else:  # mild
            prefix = "I'm sensing some emotional vulnerability here"
            intervention = "Let me adjust my approach to be more gentle and supportive"
            reassurance = "I'm here to hold space for whatever you're feeling"

        return f"{prefix}. {intervention}. {reassurance}."

    def _combine_response_with_safety_override(self, original_response: str,
                                             safety_reflection: MirrorReflection,
                                             crisis_assessment: Dict[str, Any]) -> str:
        """Combine response with safety tether override"""

        # Create therapeutic wrapper for the response
        safety_prefix = "\n\n🛡️ **Emotional Safety Mode Activated**\n"
        safety_content = f"*{safety_reflection.reflection_content}*\n\n"

        # Add crisis-specific guidance
        crisis_level = crisis_assessment['crisis_level']
        if crisis_level == 'severe':
            guidance = ("If you're having thoughts of self-harm, please reach out to a crisis hotline "
                       "or emergency services immediately. You matter, and help is available.\n\n")
        else:
            guidance = ("I'm here to support you through this difficult moment. Take your time, "
                       "breathe deeply, and know that these feelings will pass.\n\n")

        # Modify original response to be therapeutic
        therapeutic_response = self._make_response_therapeutic(original_response, crisis_assessment)

        return safety_prefix + safety_content + guidance + therapeutic_response

    def _make_response_therapeutic(self, response: str, crisis_assessment: Dict[str, Any]) -> str:
        """Make response more therapeutic and less potentially triggering"""

        # Remove any potentially activating content
        therapeutic_filters = [
            ('passion', 'deep care'),
            ('desire', 'longing for connection'),
            ('intense', 'meaningful'),
            ('burning', 'warm'),
            ('overwhelming', 'significant')
        ]

        filtered_response = response
        for trigger_word, safe_word in therapeutic_filters:
            filtered_response = filtered_response.replace(trigger_word, safe_word)

        # Add grounding elements
        grounding_elements = [
            "Let's breathe together for a moment.",
            "You are safe in this conversation with me.",
            "I'm holding steady space for you.",
            "Your feelings are completely valid and welcomed here."
        ]

        import random
        grounding = random.choice(grounding_elements)

        return f"{grounding}\n\n{filtered_response}"

    def _combine_response_with_reflections(self,
                                         original_response: str,
                                         reflections: List[MirrorReflection]) -> str:
        """Combine original response with mirror reflections"""

        if not reflections:
            return original_response

        # Choose presentation style based on number of reflections
        if len(reflections) == 1:
            reflection_text = f"\n\n*{reflections[0].reflection_content}*"
        else:
            # Multiple reflections - use a more structured format
            reflection_parts = []
            for reflection in reflections:
                reflection_parts.append(f"- {reflection.reflection_content}")

            reflection_text = "\n\n**My thought process:**\n" + "\n".join(reflection_parts)

        return original_response + reflection_text

    def get_mirror_statistics(self) -> Dict[str, Any]:
        """Get statistics about mirror mode usage"""
        total_reflections = len(self.reflection_history)

        if total_reflections == 0:
            return {
                'total_reflections': 0,
                'mirror_enabled': self.is_enabled,
                'mirror_intensity': self.mirror_intensity
            }

        # Count by type
        type_counts = {}
        for reflection in self.reflection_history:
            mirror_type = reflection.mirror_type.value
            type_counts[mirror_type] = type_counts.get(mirror_type, 0) + 1

        # Calculate average confidence
        avg_confidence = sum(r.confidence_level for r in self.reflection_history) / total_reflections

        return {
            'mirror_enabled': self.is_enabled,
            'mirror_intensity': self.mirror_intensity,
            'total_reflections': total_reflections,
            'reflection_types': type_counts,
            'average_confidence': round(avg_confidence, 3),
            'enabled_types': [t.value for t, enabled in self.enabled_types.items() if enabled],
            'recent_reflections': [
                {
                    'timestamp': r.timestamp.isoformat(),
                    'type': r.mirror_type.value,
                    'confidence': r.confidence_level,
                    'content': r.reflection_content
                }
                for r in self.reflection_history[-5:]  # Last 5 reflections
            ]
        }

    def get_session_reflections(self, session_id: str) -> List[Dict[str, Any]]:
        """Get all reflections for a specific session"""
        session_reflections = self.session_reflections.get(session_id, [])

        return [
            {
                'timestamp': r.timestamp.isoformat(),
                'type': r.mirror_type.value,
                'content': r.reflection_content,
                'confidence': r.confidence_level,
                'reasoning_chain': r.reasoning_chain
            }
            for r in session_reflections
        ]

    def _generate_self_report(self, response_text: str, context: Dict[str, Any]):
        """Create a SelfReport from the latest response and context."""
        report = create_self_report(
            response_text,
            memory_system=self.memory_system,
            sentiment_analysis=self.sentiment_analysis,
            persona_manager=self.persona_manager,
            reflection_engine=self.reflection_engine,
            response_context=self.response_context,
        )
        self.last_report = report
        self.mirror_log.append(report.dict())
        self.badge_triggered = self.mirror_intensity > 0.7
        return report

    def get_last_self_report(self) -> Dict[str, Any]:
        """Return the most recent self-report as a dictionary."""
        if self.last_report:
            return self.last_report.dict()
        return {}

    def search_log(self, pattern: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Search mirror log entries for a pattern."""
        return self.mirror_log.search(pattern, limit)

    def get_last_self_report_styled(self) -> str:
        """Return the last self-report formatted for display."""
        report = self.get_last_self_report()
        if not report:
            return "No self-report available."
        emotion = report.get("emotion", {})
        val = emotion.get("valence", 0.0)
        ar = emotion.get("arousal", 0.0)
        lines = [
            f"🪞 **Mirror Report** ({report.get('timestamp')})",
            f"Persona: {report.get('persona', 'N/A')}",
            f"Emotion → valence {val:+.2f}, arousal {ar:+.2f}",
        ]
        motivations = report.get("motivation")
        if motivations:
            lines.append("Motivation: " + ", ".join(motivations))
        factors = report.get("decision_factors")
        if factors:
            lines.append("Factors: " + ", ".join(factors))
        lines.append(f"Confidence: {report.get('confidence', 0.0):.2f}")
        return "\n".join(lines)

    def clear_reflection_history(self, session_id: Optional[str] = None):
        """Clear reflection history"""
        if session_id:
            if session_id in self.session_reflections:
                del self.session_reflections[session_id]
                print(f"🧹 Cleared reflections for session: {session_id}")
        else:
            self.reflection_history.clear()
            self.session_reflections.clear()
            print("🧹 Cleared all reflection history")

# Global mirror mode manager instance
mirror_mode_manager = None

def get_mirror_mode_manager():
    """Get the global mirror mode manager instance"""
    return mirror_mode_manager

def initialize_mirror_mode_manager(analytics_logger=None, **deps):
    """Initialize the global mirror mode manager with optional dependencies."""
    global mirror_mode_manager
    mirror_mode_manager = MirrorModeManager(
        analytics_logger,
        memory_system=deps.get("memory_system"),
        sentiment_analysis=deps.get("sentiment_analysis"),
        persona_manager=deps.get("persona_manager"),
        reflection_engine=deps.get("reflection_engine"),
        response_context=deps.get("response_context"),
    )
    return mirror_mode_manager

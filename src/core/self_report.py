"""
Self-Report System for AI Introspection and Analysis

Enhanced with security features:
- Self-assessment security validation with integrity verification
- Input validation and sanitization for all report data
- Session management and audit logging for self-reports
- Rate limiting and monitoring for report generation
"""

from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
import hashlib
import re
import time
import threading
import logging
from collections import deque, defaultdict
from pydantic import BaseModel, Field, validator
import uuid

# Enhanced logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
)
logger = logging.getLogger(__name__)

# Security configuration
SELF_REPORT_SESSION_LENGTH = 32
MAX_REPORT_TEXT_LENGTH = 2000
MAX_MOTIVATION_ITEMS = 10
REPORT_RATE_LIMIT = 20  # reports per hour per session
MAX_CONCURRENT_REPORTS = 5

# Thread safety
report_lock = threading.Lock()

# Session management
report_sessions = {}
session_expiry_hours = 12

# Rate limiting
report_requests = defaultdict(lambda: deque())

# Report monitoring
report_anomalies = deque(maxlen=50)
active_reports = {}

# Optional TextBlob import with fallback
try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
    logger.info("TextBlob available for sentiment analysis")
except ImportError:
    logger.warning("TextBlob not available - using fallback sentiment analysis")
    TEXTBLOB_AVAILABLE = False

def generate_report_session() -> str:
    """Generate a secure self-report session token"""
    timestamp = str(time.time())
    random_data = str(hash(datetime.now()))
    token_string = f"report:{timestamp}:{random_data}"
    return hashlib.sha256(token_string.encode()).hexdigest()[:SELF_REPORT_SESSION_LENGTH]

def validate_report_session(session_token: str) -> bool:
    """Validate self-report session token"""
    if not session_token or len(session_token) != SELF_REPORT_SESSION_LENGTH:
        return False
    
    if session_token not in report_sessions:
        return False
    
    # Check if session has expired
    session_data = report_sessions[session_token]
    if datetime.now() > session_data['expires_at']:
        del report_sessions[session_token]
        return False
    
    # Update last access time
    session_data['last_access'] = datetime.now()
    return True

def check_report_rate_limit(session_token: str) -> bool:
    """Check if report generation rate limit is exceeded"""
    current_time = time.time()
    
    # Clean old requests
    while (report_requests[session_token] and 
           report_requests[session_token][0] < current_time - 3600):  # 1 hour window
        report_requests[session_token].popleft()
    
    # Check limit
    if len(report_requests[session_token]) >= REPORT_RATE_LIMIT:
        logger.warning(f"Self-report rate limit exceeded for session: {session_token[:8]}...")
        return False
    
    # Add current request
    report_requests[session_token].append(current_time)
    return True

def validate_text_input(text: str, max_length: int = MAX_REPORT_TEXT_LENGTH) -> tuple[bool, str]:
    """Validate text input for reports"""
    try:
        if not isinstance(text, str):
            return False, "Input must be a string"
        
        if len(text) > max_length:
            return False, f"Text exceeds maximum length of {max_length}"
        
        # Check for injection patterns
        dangerous_patterns = [
            r'<script[^>]*>.*?</script>',  # XSS
            r'javascript:',               # JavaScript protocol
            r'on\w+\s*=',                # Event handlers
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return False, "Text contains potentially dangerous content"
        
        return True, "Valid"
    
    except Exception as e:
        logger.error(f"Error validating text input: {str(e)}")
        return False, f"Validation error: {str(e)}"

def sanitize_report_text(text: str) -> str:
    """Sanitize report text for safety"""
    if not isinstance(text, str):
        return ""
    
    # Remove potentially dangerous characters
    text = re.sub(r'[<>"\']', '', text)
    
    # Limit length
    if len(text) > MAX_REPORT_TEXT_LENGTH:
        text = text[:MAX_REPORT_TEXT_LENGTH] + "..."
    
    return text.strip()

def detect_report_anomaly(report: 'SelfReport') -> bool:
    """Detect anomalies in self-reports"""
    try:
        anomaly_detected = False
        
        # Check for extreme confidence values
        if report.confidence > 0.95 or report.confidence < 0.05:
            logger.warning(f"Extreme confidence value detected: {report.confidence}")
            anomaly_detected = True
        
        # Check for excessive motivation items
        if len(report.motivation) > MAX_MOTIVATION_ITEMS:
            logger.warning(f"Excessive motivation items: {len(report.motivation)}")
            anomaly_detected = True
        
        # Check for extreme emotion values
        if (abs(report.emotion.valence) > 0.95 or 
            report.emotion.arousal > 0.95 or report.emotion.arousal < 0.05):
            logger.warning(f"Extreme emotion values: valence={report.emotion.valence}, arousal={report.emotion.arousal}")
            anomaly_detected = True
        
        if anomaly_detected:
            report_anomalies.append({
                'timestamp': datetime.now().isoformat(),
                'report_id': report.id,
                'confidence': report.confidence,
                'motivation_count': len(report.motivation),
                'emotion_valence': report.emotion.valence,
                'emotion_arousal': report.emotion.arousal
            })
        
        return anomaly_detected
    
    except Exception as e:
        logger.error(f"Error detecting report anomaly: {str(e)}")
        return False

def log_report_activity(activity_type: str, session_token: str, details: Dict[str, Any], status: str = "success"):
    """Log self-report activities"""
    try:
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'activity_type': activity_type,
            'session': session_token[:8] + "..." if session_token else "none",
            'details': details,
            'status': status,
            'thread_id': threading.get_ident()
        }
        
        logger.info(f"Report activity logged: {activity_type} ({status})")
        
        if status != "success":
            logger.warning(f"Report activity issue: {activity_type} failed with {status}")
        
    except Exception as e:
        logger.error(f"Error logging report activity: {str(e)}")

class Emotion(BaseModel):
    """Emotion model with validation"""
    valence: float = Field(0.0, description="-1 to 1 sentiment valence")
    arousal: float = Field(0.0, description="0 to 1 intensity")
    
    @validator('valence')
    def validate_valence_range(cls, v):
        if v < -1.0 or v > 1.0:
            raise ValueError('Valence must be between -1.0 and 1.0')
        return v
    
    @validator('arousal')
    def validate_arousal_range(cls, v):
        if v < 0.0 or v > 1.0:
            raise ValueError('Arousal must be between 0.0 and 1.0')
        return v

class SelfReport(BaseModel):
    """Self-report model with security enhancements"""
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    persona: Optional[str] = None
    emotion: Emotion = Field(default_factory=Emotion)
    motivation: List[str] = Field(default_factory=list)
    decision_factors: List[str] = Field(default_factory=list)
    confidence: float = 0.0
    summary: str = ""
    session_hash: Optional[str] = None
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    
    # Security tracking
    session_token: Optional[str] = None
    anomaly_detected: bool = False
    sanitized: bool = False
    
    @validator('persona')
    def validate_persona(cls, v):
        if v is not None:
            if not isinstance(v, str) or len(v) > 100:
                raise ValueError('Persona must be a string with max length 100')
            # Sanitize persona name
            v = re.sub(r'[<>"\']', '', v)
        return v
    
    @validator('motivation', 'decision_factors')
    def validate_string_lists(cls, v):
        if len(v) > MAX_MOTIVATION_ITEMS:
            raise ValueError(f'List cannot exceed {MAX_MOTIVATION_ITEMS} items')
        # Sanitize each item
        return [sanitize_report_text(item) for item in v if isinstance(item, str)]
    
    @validator('confidence')
    def validate_confidence_range(cls, v):
        if v < 0.0 or v > 1.0:
            raise ValueError('Confidence must be between 0.0 and 1.0')
        return v
    
    @validator('summary')
    def validate_summary(cls, v):
        if v:
            is_valid, message = validate_text_input(v)
            if not is_valid:
                raise ValueError(f'Invalid summary: {message}')
            return sanitize_report_text(v)
        return v


def create_self_report(
    last_output: str,
    memory_system: Optional[Any] = None,
    sentiment_analysis: Optional[Any] = None,
    persona_manager: Optional[Any] = None,
    reflection_engine: Optional[Any] = None,
    response_context: Optional[Any] = None,
) -> SelfReport:
    """Generate a SelfReport using available integration points."""

    persona_name = None
    if persona_manager and hasattr(persona_manager, "get_active_persona"):
        try:
            persona = persona_manager.get_active_persona()
            if isinstance(persona, dict):
                persona_name = persona.get("name") or persona.get("id")
            elif persona is not None:
                persona_name = getattr(persona, "name", str(persona))
        except Exception:
            pass

    valence = 0.0
    arousal = 0.0
    if sentiment_analysis and hasattr(sentiment_analysis, "get_current_state"):
        try:
            state = sentiment_analysis.get_current_state()
            valence = float(state.get("valence", 0.0))
            arousal = float(state.get("arousal", 0.0))
        except Exception:
            pass
    else:
        try:
            blob = TextBlob(last_output)
            valence = blob.sentiment.polarity
            arousal = abs(blob.sentiment.subjectivity)
        except Exception:
            pass

    motivation = []
    if reflection_engine and hasattr(reflection_engine, "get_last_insight"):
        try:
            insight = reflection_engine.get_last_insight()
            if insight:
                motivation.extend(insight.get("tags", []))
        except Exception:
            pass

    decision_factors = []
    session_hash = None
    if memory_system and hasattr(memory_system, "get_last_memory_session"):
        try:
            session = memory_system.get_last_memory_session()
            if session:
                trend = session.get("sentiment_trend")
                if trend is not None:
                    decision_factors.append(f"recent sentiment {trend:+.2f}")
                sid = session.get("session_id")
                if sid:
                    session_hash = hashlib.sha256(sid.encode()).hexdigest()[:8]
        except Exception:
            pass

    if response_context and hasattr(response_context, "get_last_response_metadata"):
        try:
            meta = response_context.get_last_response_metadata()
            if meta:
                reason = meta.get("reason") or meta.get("handler")
                if reason:
                    decision_factors.append(reason)
        except Exception:
            pass

    summary_parts = []
    if persona_name:
        summary_parts.append(f"Persona {persona_name} active.")
    summary_parts.append(f"Valence {valence:+.2f}, Arousal {arousal:+.2f}.")
    if motivation:
        summary_parts.append("Motivation: " + ", ".join(motivation))
    if decision_factors:
        summary_parts.append("Factors: " + ", ".join(decision_factors))

    summary = " " .join(summary_parts)

    return SelfReport(
        persona=persona_name,
        emotion=Emotion(valence=valence, arousal=arousal),
        motivation=motivation,
        decision_factors=decision_factors,
        session_hash=session_hash,
        confidence=0.75,
        summary=summary,
    )

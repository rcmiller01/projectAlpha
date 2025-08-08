"""
Personality Evolution System

Enhanced with security features:
- Personality evolution security with trait validation
- Evolution tracking and audit trails
- Session management and access control for personality changes
- Rate limiting and monitoring for evolution operations
"""

import hashlib
import re
import time
import threading
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
from collections import deque, defaultdict
from dataclasses import dataclass, asdict

# Enhanced logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
)
logger = logging.getLogger(__name__)

# Security configuration
PERSONALITY_SESSION_LENGTH = 32
MAX_TRAIT_VALUE = 100.0
MIN_TRAIT_VALUE = 0.0
MAX_TRAITS_COUNT = 50
MAX_TRAIT_NAME_LENGTH = 100
PERSONALITY_RATE_LIMIT = 20  # evolution operations per hour per session
MAX_EVOLUTION_HISTORY = 1000

# Thread safety
personality_lock = threading.Lock()

# Session management
personality_sessions = {}
session_expiry_hours = 24

# Rate limiting
personality_requests = defaultdict(lambda: deque())

# Access monitoring
personality_access_history = deque(maxlen=1000)

@dataclass
class PersonalityTrait:
    """A single personality trait with metadata"""
    name: str
    value: float
    last_modified: str
    modification_count: int = 0
    evolution_history: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.evolution_history is None:
            self.evolution_history = []

@dataclass
class PersonalityProfile:
    """Complete personality profile with security metadata"""
    profile_id: str
    traits: Dict[str, PersonalityTrait]
    creation_time: str
    last_evolution: str
    evolution_count: int
    session_token: str
    security_metadata: Dict[str, Any]

def generate_personality_session() -> str:
    """Generate a secure personality session token"""
    timestamp = str(time.time())
    random_data = str(hash(datetime.now()))
    token_string = f"personality:{timestamp}:{random_data}"
    return hashlib.sha256(token_string.encode()).hexdigest()[:PERSONALITY_SESSION_LENGTH]

def validate_personality_session(session_token: str) -> bool:
    """Validate personality session token"""
    if not session_token or len(session_token) != PERSONALITY_SESSION_LENGTH:
        return False
    
    if session_token not in personality_sessions:
        return False
    
    # Check if session has expired
    session_data = personality_sessions[session_token]
    if datetime.now() > session_data['expires_at']:
        del personality_sessions[session_token]
        return False
    
    # Update last access time
    session_data['last_access'] = datetime.now()
    return True

def check_personality_rate_limit(session_token: str) -> bool:
    """Check if personality operation rate limit is exceeded"""
    current_time = time.time()
    
    # Clean old requests
    while (personality_requests[session_token] and 
           personality_requests[session_token][0] < current_time - 3600):  # 1 hour window
        personality_requests[session_token].popleft()
    
    # Check limit
    if len(personality_requests[session_token]) >= PERSONALITY_RATE_LIMIT:
        logger.warning(f"Personality rate limit exceeded for session: {session_token[:8]}...")
        return False
    
    # Add current request
    personality_requests[session_token].append(current_time)
    return True

def validate_personality_trait(name: str, value: float) -> Tuple[bool, str]:
    """Validate personality trait"""
    try:
        # Validate name
        if not isinstance(name, str):
            return False, "Trait name must be a string"
        
        if len(name) > MAX_TRAIT_NAME_LENGTH:
            return False, f"Trait name too long (max {MAX_TRAIT_NAME_LENGTH} characters)"
        
        # Check for dangerous characters
        if re.search(r'[<>"\']', name):
            return False, "Trait name contains invalid characters"
        
        # Validate value
        if not isinstance(value, (int, float)):
            return False, "Trait value must be a number"
        
        if not (MIN_TRAIT_VALUE <= value <= MAX_TRAIT_VALUE):
            return False, f"Trait value must be between {MIN_TRAIT_VALUE} and {MAX_TRAIT_VALUE}"
        
        return True, "Valid"
    
    except Exception as e:
        logger.error(f"Error validating personality trait: {str(e)}")
        return False, f"Validation error: {str(e)}"

def sanitize_trait_name(name: str) -> str:
    """Sanitize trait name for safety"""
    # Remove dangerous characters
    clean_name = re.sub(r'[<>"\']', '', str(name))
    # Limit length
    if len(clean_name) > MAX_TRAIT_NAME_LENGTH:
        clean_name = clean_name[:MAX_TRAIT_NAME_LENGTH]
    return clean_name

def log_personality_activity(activity_type: str, session_token: str, details: Dict[str, Any], status: str = "success"):
    """Log personality access activities"""
    try:
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'activity_type': activity_type,
            'session': session_token[:8] + "..." if session_token else "none",
            'details': details,
            'status': status,
            'thread_id': threading.get_ident()
        }
        
        personality_access_history.append(log_entry)
        
        logger.info(f"Personality access logged: {activity_type} ({status})")
        
        if status != "success":
            logger.warning(f"Personality access issue: {activity_type} failed with {status}")
        
    except Exception as e:
        logger.error(f"Error logging personality access: {str(e)}")

class PersonalityEvolution:
    """
    Secure personality evolution system for AI agents.
    
    Features:
    - Trait-based personality modeling with validation
    - Evolution tracking and history
    - Session-based authentication and authorization
    - Rate limiting and access monitoring
    - Comprehensive audit logging
    - Thread-safe concurrent operations
    """

    def __init__(self, session_token: Optional[str] = None, profile_id: Optional[str] = None):
        """Initialize personality evolution system with security features"""
        self.session_token = session_token or self.create_session()
        self.profile_id = profile_id or f"profile_{hashlib.md5(f'{self.session_token}{time.time()}'.encode()).hexdigest()[:16]}"
        self.creation_time = datetime.now()
        self.evolution_count = 0
        
        # Initialize default personality traits
        self.traits: Dict[str, PersonalityTrait] = {}
        self._initialize_default_traits()
        
        log_personality_activity("initialization", self.session_token, {
            "profile_id": self.profile_id,
            "creation_time": self.creation_time.isoformat(),
            "default_traits_count": len(self.traits)
        })
        
        logger.info(f"PersonalityEvolution initialized with security features: {self.profile_id}")

    def create_session(self) -> str:
        """Create a new personality session"""
        with personality_lock:
            session_token = generate_personality_session()
            personality_sessions[session_token] = {
                'created_at': datetime.now(),
                'expires_at': datetime.now() + timedelta(hours=session_expiry_hours),
                'last_access': datetime.now(),
                'personality_operations': 0
            }
            return session_token

    def validate_session(self, session_token: Optional[str] = None) -> bool:
        """Validate session for personality operations"""
        token_to_validate = session_token or self.session_token
        
        if not token_to_validate:
            logger.warning("No session token provided for personality validation")
            return False
        
        return validate_personality_session(token_to_validate)

    def _initialize_default_traits(self):
        """Initialize default personality traits"""
        default_traits = {
            "openness": 50.0,
            "conscientiousness": 50.0,
            "extraversion": 50.0,
            "agreeableness": 50.0,
            "neuroticism": 50.0,
            "creativity": 50.0,
            "empathy": 50.0,
            "curiosity": 50.0,
            "assertiveness": 50.0,
            "adaptability": 50.0
        }
        
        current_time = datetime.now().isoformat()
        
        for name, value in default_traits.items():
            self.traits[name] = PersonalityTrait(
                name=name,
                value=value,
                last_modified=current_time,
                modification_count=0,
                evolution_history=[]
            )

    def evolve_trait(self, trait_name: str, delta: float, session_token: Optional[str] = None) -> bool:
        """
        Evolve a personality trait with security validation.
        
        Args:
            trait_name: Name of the trait to evolve
            delta: Change amount (positive or negative)
            session_token: Session token for authentication
            
        Returns:
            Success status
        """
        try:
            # Validate session
            if not self.validate_session(session_token):
                log_personality_activity("evolve_trait", session_token or self.session_token, 
                                        {"trait": trait_name, "status": "session_invalid"}, "failed")
                return False
            
            current_token = session_token or self.session_token
            
            # Check rate limit
            if not check_personality_rate_limit(current_token):
                log_personality_activity("evolve_trait", current_token, 
                                        {"trait": trait_name, "status": "rate_limited"}, "failed")
                return False
            
            # Sanitize trait name
            clean_trait_name = sanitize_trait_name(trait_name)
            if not clean_trait_name:
                logger.error("Invalid trait name after sanitization")
                return False
            
            # Validate delta
            if not isinstance(delta, (int, float)):
                logger.error("Delta must be a number")
                return False
            
            if abs(delta) > 50.0:  # Limit large changes
                delta = 50.0 if delta > 0 else -50.0
                logger.warning(f"Delta clamped to {delta} for safety")
            
            with personality_lock:
                # Get or create trait
                if clean_trait_name not in self.traits:
                    self.traits[clean_trait_name] = PersonalityTrait(
                        name=clean_trait_name,
                        value=50.0,  # Default neutral value
                        last_modified=datetime.now().isoformat(),
                        modification_count=0,
                        evolution_history=[]
                    )
                
                trait = self.traits[clean_trait_name]
                old_value = trait.value
                new_value = max(MIN_TRAIT_VALUE, min(MAX_TRAIT_VALUE, old_value + delta))
                
                # Validate new value
                is_valid, validation_message = validate_personality_trait(clean_trait_name, new_value)
                if not is_valid:
                    logger.error(f"Invalid trait evolution: {validation_message}")
                    log_personality_activity("evolve_trait", current_token, 
                                            {"trait": clean_trait_name, "error": validation_message}, "validation_failed")
                    return False
                
                # Update trait
                trait.value = new_value
                trait.last_modified = datetime.now().isoformat()
                trait.modification_count += 1
                
                # Add to evolution history
                evolution_entry = {
                    "timestamp": trait.last_modified,
                    "old_value": old_value,
                    "new_value": new_value,
                    "delta": delta,
                    "session_token": current_token[:8] + "..."
                }
                trait.evolution_history.append(evolution_entry)
                
                # Limit history size
                if len(trait.evolution_history) > MAX_EVOLUTION_HISTORY:
                    trait.evolution_history = trait.evolution_history[-MAX_EVOLUTION_HISTORY:]
                
                self.evolution_count += 1
                
                # Update session tracking
                if current_token in personality_sessions:
                    personality_sessions[current_token]['personality_operations'] += 1
            
            log_personality_activity("evolve_trait", current_token, {
                "trait": clean_trait_name,
                "old_value": old_value,
                "new_value": new_value,
                "delta": delta,
                "evolution_count": self.evolution_count
            })
            
            logger.info(f"Trait evolved: {clean_trait_name} {old_value:.2f} -> {new_value:.2f} (Δ{delta:+.2f})")
            return True
            
        except Exception as e:
            logger.error(f"Error evolving trait: {str(e)}")
            log_personality_activity("evolve_trait", session_token or self.session_token, 
                                    {"trait": trait_name, "error": str(e)}, "error")
            return False

    def get_trait(self, trait_name: str, session_token: Optional[str] = None) -> Optional[float]:
        """
        Get trait value with security validation.
        
        Args:
            trait_name: Name of the trait
            session_token: Session token for authentication
            
        Returns:
            Trait value or None if not found/unauthorized
        """
        try:
            # Validate session
            if not self.validate_session(session_token):
                log_personality_activity("get_trait", session_token or self.session_token, 
                                        {"trait": trait_name, "status": "session_invalid"}, "failed")
                return None
            
            clean_trait_name = sanitize_trait_name(trait_name)
            
            if clean_trait_name in self.traits:
                value = self.traits[clean_trait_name].value
                
                log_personality_activity("get_trait", session_token or self.session_token, {
                    "trait": clean_trait_name,
                    "value": value
                })
                
                return value
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting trait: {str(e)}")
            return None

    def get_personality_profile(self, session_token: Optional[str] = None) -> Optional[PersonalityProfile]:
        """
        Get complete personality profile with security validation.
        
        Args:
            session_token: Session token for authentication
            
        Returns:
            PersonalityProfile or None if unauthorized
        """
        try:
            # Validate session
            if not self.validate_session(session_token):
                log_personality_activity("get_profile", session_token or self.session_token, 
                                        {"status": "session_invalid"}, "failed")
                return None
            
            current_token = session_token or self.session_token
            
            profile = PersonalityProfile(
                profile_id=self.profile_id,
                traits=self.traits.copy(),
                creation_time=self.creation_time.isoformat(),
                last_evolution=datetime.now().isoformat(),
                evolution_count=self.evolution_count,
                session_token=current_token[:8] + "...",
                security_metadata={
                    "session_validated": True,
                    "traits_count": len(self.traits),
                    "total_evolutions": self.evolution_count
                }
            )
            
            log_personality_activity("get_profile", current_token, {
                "profile_id": self.profile_id,
                "traits_count": len(self.traits),
                "evolution_count": self.evolution_count
            })
            
            return profile
            
        except Exception as e:
            logger.error(f"Error getting personality profile: {str(e)}")
            log_personality_activity("get_profile", session_token or self.session_token, 
                                    {"error": str(e)}, "error")
            return None

    def reset_trait(self, trait_name: str, session_token: Optional[str] = None) -> bool:
        """
        Reset a trait to default value with security validation.
        
        Args:
            trait_name: Name of the trait to reset
            session_token: Session token for authentication
            
        Returns:
            Success status
        """
        try:
            # Validate session
            if not self.validate_session(session_token):
                log_personality_activity("reset_trait", session_token or self.session_token, 
                                        {"trait": trait_name, "status": "session_invalid"}, "failed")
                return False
            
            current_token = session_token or self.session_token
            
            # Check rate limit
            if not check_personality_rate_limit(current_token):
                log_personality_activity("reset_trait", current_token, 
                                        {"trait": trait_name, "status": "rate_limited"}, "failed")
                return False
            
            clean_trait_name = sanitize_trait_name(trait_name)
            
            if clean_trait_name in self.traits:
                with personality_lock:
                    old_value = self.traits[clean_trait_name].value
                    self.traits[clean_trait_name].value = 50.0  # Reset to neutral
                    self.traits[clean_trait_name].last_modified = datetime.now().isoformat()
                    self.traits[clean_trait_name].modification_count += 1
                    
                    # Add reset to history
                    reset_entry = {
                        "timestamp": self.traits[clean_trait_name].last_modified,
                        "old_value": old_value,
                        "new_value": 50.0,
                        "action": "reset",
                        "session_token": current_token[:8] + "..."
                    }
                    self.traits[clean_trait_name].evolution_history.append(reset_entry)
                    
                    self.evolution_count += 1
                
                log_personality_activity("reset_trait", current_token, {
                    "trait": clean_trait_name,
                    "old_value": old_value,
                    "new_value": 50.0
                })
                
                logger.info(f"Trait reset: {clean_trait_name} {old_value:.2f} -> 50.0")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error resetting trait: {str(e)}")
            log_personality_activity("reset_trait", session_token or self.session_token, 
                                    {"trait": trait_name, "error": str(e)}, "error")
            return False

    def get_evolution_stats(self) -> Dict[str, Any]:
        """Get personality evolution statistics"""
        return {
            'profile_id': self.profile_id,
            'session_token': self.session_token[:8] + "..." if self.session_token else None,
            'creation_time': self.creation_time.isoformat(),
            'evolution_count': self.evolution_count,
            'traits_count': len(self.traits),
            'most_evolved_traits': sorted(
                [(name, trait.modification_count) for name, trait in self.traits.items()],
                key=lambda x: x[1], reverse=True
            )[:5]
        }

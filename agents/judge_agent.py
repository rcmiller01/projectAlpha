"""
Agent capable of evaluating system outputs or behaviors.

Enhanced with security features:
- Judge agent security with evaluation validation
- Input sanitization and assessment criteria validation
- Session management and audit trails for evaluation activities
- Rate limiting and monitoring for judgment operations
"""

from typing import Any, Dict, Optional, List
import hashlib
import re
import time
import threading
import logging
from datetime import datetime, timedelta
from collections import deque, defaultdict

from .base_agent import BaseAgent

# Enhanced logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
)
logger = logging.getLogger(__name__)

# Security configuration
JUDGE_SESSION_LENGTH = 32
MAX_OUTPUT_SIZE = 10000
MAX_CRITERIA_COUNT = 15
MAX_CRITERIA_LENGTH = 500
JUDGE_RATE_LIMIT = 40  # evaluations per hour per session

# Thread safety
judge_lock = threading.Lock()

# Session management
judge_sessions = {}
session_expiry_hours = 24

# Rate limiting
judge_requests = defaultdict(lambda: deque())

# Access monitoring
judge_access_history = deque(maxlen=1000)

def generate_judge_session() -> str:
    """Generate a secure judge session token"""
    timestamp = str(time.time())
    random_data = str(hash(datetime.now()))
    token_string = f"judge:{timestamp}:{random_data}"
    return hashlib.sha256(token_string.encode()).hexdigest()[:JUDGE_SESSION_LENGTH]

def validate_judge_session(session_token: str) -> bool:
    """Validate judge session token"""
    if not session_token or len(session_token) != JUDGE_SESSION_LENGTH:
        return False
    
    if session_token not in judge_sessions:
        return False
    
    # Check if session has expired
    session_data = judge_sessions[session_token]
    if datetime.now() > session_data['expires_at']:
        del judge_sessions[session_token]
        return False
    
    # Update last access time
    session_data['last_access'] = datetime.now()
    return True

def check_judge_rate_limit(session_token: str) -> bool:
    """Check if judge operation rate limit is exceeded"""
    current_time = time.time()
    
    # Clean old requests
    while (judge_requests[session_token] and 
           judge_requests[session_token][0] < current_time - 3600):  # 1 hour window
        judge_requests[session_token].popleft()
    
    # Check limit
    if len(judge_requests[session_token]) >= JUDGE_RATE_LIMIT:
        logger.warning(f"Judge rate limit exceeded for session: {session_token[:8]}...")
        return False
    
    # Add current request
    judge_requests[session_token].append(current_time)
    return True

def validate_evaluation_input(output: Any, criteria: Optional[List[str]] = None) -> tuple[bool, str]:
    """Validate evaluation input"""
    try:
        # Convert output to string for validation
        if output is None:
            return False, "Output cannot be None"
        
        output_str = str(output)
        
        # Check output size
        if len(output_str) > MAX_OUTPUT_SIZE:
            return False, f"Output too large: {len(output_str)} characters (max {MAX_OUTPUT_SIZE})"
        
        # Check for dangerous content
        dangerous_patterns = [
            r'<script[^>]*>.*?</script>',  # XSS
            r'javascript:',               # JavaScript protocol
            r'on\w+\s*=',                # Event handlers
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, output_str, re.IGNORECASE):
                return False, "Output contains potentially dangerous content"
        
        # Validate criteria if provided
        if criteria is not None:
            if not isinstance(criteria, list):
                return False, "Criteria must be a list"
            
            if len(criteria) > MAX_CRITERIA_COUNT:
                return False, f"Too many criteria (max {MAX_CRITERIA_COUNT})"
            
            for criterion in criteria:
                if not isinstance(criterion, str):
                    return False, "All criteria must be strings"
                if len(criterion) > MAX_CRITERIA_LENGTH:
                    return False, f"Criterion too long (max {MAX_CRITERIA_LENGTH} characters)"
        
        return True, "Valid"
    
    except Exception as e:
        logger.error(f"Error validating evaluation input: {str(e)}")
        return False, f"Validation error: {str(e)}"

def sanitize_evaluation_input(output: Any, criteria: Optional[List[str]] = None) -> tuple[str, List[str]]:
    """Sanitize evaluation input"""
    # Sanitize output
    output_str = str(output) if output is not None else ""
    clean_output = re.sub(r'[<>"\']', '', output_str)
    if len(clean_output) > MAX_OUTPUT_SIZE:
        clean_output = clean_output[:MAX_OUTPUT_SIZE] + "..."
    
    # Sanitize criteria
    clean_criteria = []
    if criteria:
        for criterion in criteria[:MAX_CRITERIA_COUNT]:
            clean_criterion = re.sub(r'[<>"\']', '', str(criterion))
            if len(clean_criterion) > MAX_CRITERIA_LENGTH:
                clean_criterion = clean_criterion[:MAX_CRITERIA_LENGTH] + "..."
            clean_criteria.append(clean_criterion)
    
    return clean_output, clean_criteria

def log_judge_activity(activity_type: str, session_token: str, details: Dict[str, Any], status: str = "success"):
    """Log judge access activities"""
    try:
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'activity_type': activity_type,
            'session': session_token[:8] + "..." if session_token else "none",
            'details': details,
            'status': status,
            'thread_id': threading.get_ident()
        }
        
        judge_access_history.append(log_entry)
        
        logger.info(f"Judge access logged: {activity_type} ({status})")
        
        if status != "success":
            logger.warning(f"Judge access issue: {activity_type} failed with {status}")
        
    except Exception as e:
        logger.error(f"Error logging judge access: {str(e)}")


class JudgeAgent(BaseAgent):
    """
    Evaluate outputs or behaviors using defined criteria with enhanced security.
    
    Features:
    - Secure evaluation with input validation and sanitization
    - Session-based authentication and authorization
    - Rate limiting and access monitoring
    - Comprehensive audit logging
    - Thread-safe concurrent evaluation
    """

    def __init__(self, session_token: Optional[str] = None, **kwargs):
        """Initialize judge agent with security features"""
        super().__init__(**kwargs)
        self.session_token = session_token or self.create_session()
        self.evaluation_count = 0
        self.creation_time = datetime.now()
        
        log_judge_activity("initialization", self.session_token, {
            "agent_type": "judge",
            "creation_time": self.creation_time.isoformat()
        })
        
        logger.info(f"JudgeAgent initialized with security features")

    def create_session(self) -> str:
        """Create a new judge session"""
        with judge_lock:
            session_token = generate_judge_session()
            judge_sessions[session_token] = {
                'created_at': datetime.now(),
                'expires_at': datetime.now() + timedelta(hours=session_expiry_hours),
                'last_access': datetime.now(),
                'evaluations': 0
            }
            return session_token

    def validate_session(self, session_token: Optional[str] = None) -> bool:
        """Validate session for judge operations"""
        token_to_validate = session_token or self.session_token
        
        if not token_to_validate:
            logger.warning("No session token provided for judge validation")
            return False
        
        return validate_judge_session(token_to_validate)

    def evaluate(self, output: Any, criteria: Optional[List[str]] = None, session_token: Optional[str] = None) -> Dict[str, Any]:
        """
        Assess a given output or behavior with security validation.

        Args:
            output: The content or behavior to evaluate.
            criteria: Optional list of evaluation criteria.
            session_token: Session token for authentication.

        Returns:
            A structured evaluation report with security metadata.
        """
        try:
            # Validate session
            if not self.validate_session(session_token):
                log_judge_activity("evaluate", session_token or self.session_token, 
                                  {"status": "session_invalid"}, "failed")
                return {
                    "success": False,
                    "error": "Invalid session for evaluation",
                    "timestamp": datetime.now().isoformat()
                }
            
            current_token = session_token or self.session_token
            
            # Check rate limit
            if not check_judge_rate_limit(current_token):
                log_judge_activity("evaluate", current_token, 
                                  {"status": "rate_limited"}, "failed")
                return {
                    "success": False,
                    "error": "Rate limit exceeded for judge operations",
                    "timestamp": datetime.now().isoformat()
                }
            
            # Validate input
            is_valid, validation_message = validate_evaluation_input(output, criteria)
            if not is_valid:
                logger.error(f"Invalid evaluation input: {validation_message}")
                log_judge_activity("evaluate", current_token, 
                                  {"error": validation_message}, "validation_failed")
                return {
                    "success": False,
                    "error": f"Validation failed: {validation_message}",
                    "timestamp": datetime.now().isoformat()
                }
            
            # Sanitize input
            clean_output, clean_criteria = sanitize_evaluation_input(output, criteria)
            
            # Perform evaluation (placeholder implementation)
            evaluation_score = self._calculate_evaluation_score(clean_output, clean_criteria)
            evaluation_report = self._generate_evaluation_report(clean_output, clean_criteria, evaluation_score)
            
            # Track evaluation
            with judge_lock:
                self.evaluation_count += 1
                if current_token in judge_sessions:
                    judge_sessions[current_token]['evaluations'] += 1
            
            # Create result
            result = {
                "success": True,
                "evaluation_id": hashlib.md5(f"{current_token}{self.evaluation_count}{time.time()}".encode()).hexdigest()[:16],
                "score": evaluation_score,
                "report": evaluation_report,
                "criteria_used": clean_criteria,
                "output_summary": clean_output[:200] + "..." if len(clean_output) > 200 else clean_output,
                "evaluation_count": self.evaluation_count,
                "timestamp": datetime.now().isoformat(),
                "session_token": current_token[:8] + "...",
                "security_metadata": {
                    "input_sanitized": True,
                    "session_validated": True,
                    "rate_limit_checked": True
                }
            }
            
            log_judge_activity("evaluate", current_token, {
                "evaluation_id": result["evaluation_id"],
                "score": evaluation_score,
                "criteria_count": len(clean_criteria),
                "output_length": len(clean_output)
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Error in judge evaluation: {str(e)}")
            log_judge_activity("evaluate", session_token or self.session_token, 
                              {"error": str(e)}, "error")
            return {
                "success": False,
                "error": f"Evaluation error: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }

    def _calculate_evaluation_score(self, output: str, criteria: List[str]) -> float:
        """Calculate evaluation score based on output and criteria"""
        try:
            # Basic scoring algorithm (can be enhanced with ML models)
            base_score = 0.5  # Neutral starting point
            
            # Length factor (appropriate length gets higher score)
            if 100 <= len(output) <= 1000:
                base_score += 0.1
            
            # Criteria matching (simple keyword matching)
            if criteria:
                matches = 0
                for criterion in criteria:
                    if criterion.lower() in output.lower():
                        matches += 1
                criteria_score = matches / len(criteria) * 0.4
                base_score += criteria_score
            else:
                # Default criteria if none provided
                base_score += 0.2
            
            # Quality indicators (simple heuristics)
            quality_indicators = ['clear', 'accurate', 'relevant', 'helpful', 'complete']
            quality_matches = sum(1 for indicator in quality_indicators if indicator in output.lower())
            quality_score = min(quality_matches / len(quality_indicators) * 0.3, 0.3)
            base_score += quality_score
            
            return min(max(base_score, 0.0), 1.0)  # Clamp between 0 and 1
            
        except Exception as e:
            logger.error(f"Error calculating evaluation score: {str(e)}")
            return 0.5  # Neutral score on error

    def _generate_evaluation_report(self, output: str, criteria: List[str], score: float) -> Dict[str, Any]:
        """Generate detailed evaluation report"""
        try:
            report = {
                "overall_score": score,
                "score_breakdown": {
                    "length_appropriateness": 0.1 if 100 <= len(output) <= 1000 else 0.0,
                    "criteria_alignment": 0.0,
                    "quality_indicators": 0.0
                },
                "strengths": [],
                "weaknesses": [],
                "recommendations": []
            }
            
            # Criteria analysis
            if criteria:
                matches = sum(1 for criterion in criteria if criterion.lower() in output.lower())
                report["score_breakdown"]["criteria_alignment"] = matches / len(criteria) * 0.4
                
                if matches > 0:
                    report["strengths"].append(f"Addresses {matches}/{len(criteria)} specified criteria")
                else:
                    report["weaknesses"].append("Does not clearly address specified criteria")
            
            # Quality analysis
            quality_indicators = ['clear', 'accurate', 'relevant', 'helpful', 'complete']
            quality_matches = [indicator for indicator in quality_indicators if indicator in output.lower()]
            report["score_breakdown"]["quality_indicators"] = len(quality_matches) / len(quality_indicators) * 0.3
            
            if quality_matches:
                report["strengths"].append(f"Shows quality indicators: {', '.join(quality_matches)}")
            
            # Recommendations
            if score < 0.5:
                report["recommendations"].append("Consider improving clarity and relevance")
                report["recommendations"].append("Ensure all criteria are addressed")
            elif score < 0.8:
                report["recommendations"].append("Good foundation, consider enhancing detail")
            else:
                report["recommendations"].append("Excellent quality, maintain current approach")
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating evaluation report: {str(e)}")
            return {
                "overall_score": score,
                "error": "Failed to generate detailed report",
                "basic_assessment": "evaluation completed with errors"
            }

    def get_evaluation_stats(self) -> Dict[str, Any]:
        """Get judge agent statistics"""
        return {
            'session_token': self.session_token[:8] + "..." if self.session_token else None,
            'evaluation_count': self.evaluation_count,
            'creation_time': self.creation_time.isoformat(),
            'agent_type': 'judge'
        }

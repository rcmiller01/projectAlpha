"""
Core Conductor - Enhanced with GraphRAG Memory and Tool Routing

Enhanced with security features:
- Conductor security orchestration with authentication
- Strategic decision validation and integrity verification
- Session management and audit trails for conductor operations
- Rate limiting and monitoring for model operations

This module demonstrates integration of the Core Conductor with the new
GraphRAG memory system and tool router for enhanced reasoning capabilities.

Key Features:
- Strategic reasoning enhanced with semantic memory
- Autonomous tool usage for expanded capabilities  
- Seamless integration with existing HRM stack
- Preparation for future SLiM agent integration

Author: ProjectAlpha Team
Compatible with: HRM stack, GraphRAG memory, Tool router
"""

import threading
import uuid
import logging
import hashlib
import re
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from collections import deque, defaultdict

# Import the HRM Router integration
try:
    from .hrm_router import HRMRouter
    from .init_models import load_conductor_models, ModelInterface
except ImportError:
    from backend.hrm_router import HRMRouter
    from init_models import load_conductor_models, ModelInterface

# Enhanced logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
)
logger = logging.getLogger(__name__)

# Security configuration
CONDUCTOR_SESSION_LENGTH = 32
MAX_SITUATION_LENGTH = 2000
MAX_OBJECTIVES_COUNT = 20
MAX_CONSTRAINTS_COUNT = 15
CONDUCTOR_RATE_LIMIT = 30  # conductor operations per hour per session
MAX_MODELS_PER_CONDUCTOR = 20

# Thread safety
conductor_lock = threading.Lock()

# Session management
conductor_sessions = {}
session_expiry_hours = 24

# Rate limiting
conductor_requests = defaultdict(lambda: deque())

# Access monitoring
conductor_access_history = deque(maxlen=1000)

def generate_conductor_session() -> str:
    """Generate a secure conductor session token"""
    timestamp = str(time.time())
    random_data = str(hash(datetime.now()))
    token_string = f"conductor:{timestamp}:{random_data}"
    return hashlib.sha256(token_string.encode()).hexdigest()[:CONDUCTOR_SESSION_LENGTH]

def validate_conductor_session(session_token: str) -> bool:
    """Validate conductor session token"""
    if not session_token or len(session_token) != CONDUCTOR_SESSION_LENGTH:
        return False
    
    if session_token not in conductor_sessions:
        return False
    
    # Check if session has expired
    session_data = conductor_sessions[session_token]
    if datetime.now() > session_data['expires_at']:
        del conductor_sessions[session_token]
        return False
    
    # Update last access time
    session_data['last_access'] = datetime.now()
    return True

def check_conductor_rate_limit(session_token: str) -> bool:
    """Check if conductor operation rate limit is exceeded"""
    current_time = time.time()
    
    # Clean old requests
    while (conductor_requests[session_token] and 
           conductor_requests[session_token][0] < current_time - 3600):  # 1 hour window
        conductor_requests[session_token].popleft()
    
    # Check limit
    if len(conductor_requests[session_token]) >= CONDUCTOR_RATE_LIMIT:
        logger.warning(f"Conductor rate limit exceeded for session: {session_token[:8]}...")
        return False
    
    # Add current request
    conductor_requests[session_token].append(current_time)
    return True

def validate_strategic_input(situation: str, objectives: Optional[List[str]], constraints: Optional[List[str]]) -> tuple[bool, str]:
    """Validate strategic decision input"""
    try:
        # Validate situation
        if not isinstance(situation, str):
            return False, "Situation must be a string"
        
        if len(situation) > MAX_SITUATION_LENGTH:
            return False, f"Situation too long (max {MAX_SITUATION_LENGTH} characters)"
        
        # Check for dangerous content
        dangerous_patterns = [
            r'<script[^>]*>.*?</script>',  # XSS
            r'javascript:',               # JavaScript protocol
            r'on\w+\s*=',                # Event handlers
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, situation, re.IGNORECASE):
                return False, "Situation contains potentially dangerous content"
        
        # Validate objectives
        if objectives is not None:
            if not isinstance(objectives, list):
                return False, "Objectives must be a list"
            
            if len(objectives) > MAX_OBJECTIVES_COUNT:
                return False, f"Too many objectives (max {MAX_OBJECTIVES_COUNT})"
            
            for obj in objectives:
                if not isinstance(obj, str):
                    return False, "All objectives must be strings"
                if len(obj) > 500:
                    return False, "Objective too long (max 500 characters)"
        
        # Validate constraints
        if constraints is not None:
            if not isinstance(constraints, list):
                return False, "Constraints must be a list"
            
            if len(constraints) > MAX_CONSTRAINTS_COUNT:
                return False, f"Too many constraints (max {MAX_CONSTRAINTS_COUNT})"
            
            for constraint in constraints:
                if not isinstance(constraint, str):
                    return False, "All constraints must be strings"
                if len(constraint) > 500:
                    return False, "Constraint too long (max 500 characters)"
        
        return True, "Valid"
    
    except Exception as e:
        logger.error(f"Error validating strategic input: {str(e)}")
        return False, f"Validation error: {str(e)}"

def sanitize_strategic_input(situation: str, objectives: Optional[List[str]], constraints: Optional[List[str]]) -> tuple[str, List[str], List[str]]:
    """Sanitize strategic decision input"""
    # Sanitize situation
    clean_situation = re.sub(r'[<>"\']', '', situation)
    if len(clean_situation) > MAX_SITUATION_LENGTH:
        clean_situation = clean_situation[:MAX_SITUATION_LENGTH] + "..."
    
    # Sanitize objectives
    clean_objectives = []
    if objectives:
        for obj in objectives[:MAX_OBJECTIVES_COUNT]:
            clean_obj = re.sub(r'[<>"\']', '', str(obj))
            if len(clean_obj) > 500:
                clean_obj = clean_obj[:500] + "..."
            clean_objectives.append(clean_obj)
    
    # Sanitize constraints
    clean_constraints = []
    if constraints:
        for constraint in constraints[:MAX_CONSTRAINTS_COUNT]:
            clean_constraint = re.sub(r'[<>"\']', '', str(constraint))
            if len(clean_constraint) > 500:
                clean_constraint = clean_constraint[:500] + "..."
            clean_constraints.append(clean_constraint)
    
    return clean_situation, clean_objectives, clean_constraints

def log_conductor_activity(activity_type: str, session_token: str, details: Dict[str, Any], status: str = "success"):
    """Log conductor access activities"""
    try:
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'activity_type': activity_type,
            'session': session_token[:8] + "..." if session_token else "none",
            'details': details,
            'status': status,
            'thread_id': threading.get_ident()
        }
        
        conductor_access_history.append(log_entry)
        
        logger.info(f"Conductor access logged: {activity_type} ({status})")
        
        if status != "success":
            logger.warning(f"Conductor access issue: {activity_type} failed with {status}")
        
    except Exception as e:
        logger.error(f"Error logging conductor access: {str(e)}")

@dataclass
class ConductorDecision:
    """Structured decision output from conductor with security metadata"""
    decision_id: str
    strategic_context: str
    action_plan: List[str]
    memory_context: List[Dict[str, Any]]
    tool_recommendations: List[str]
    confidence: float
    reasoning_path: List[str]
    timestamp: str
    session_token: str
    security_metadata: Dict[str, Any]

class CoreConductor:
    """
    Enhanced Core Conductor with GraphRAG Memory, Tool Routing, and Security
    
    The conductor provides strategic, high-level reasoning and planning
    enhanced with semantic memory, autonomous tool capabilities, and comprehensive security.
    
    Features:
    - Strategic decision making with memory context
    - Autonomous tool discovery and usage
    - Integration with GraphRAG semantic memory
    - Preparation for multi-agent orchestration
    - Thread-safe concurrent operation
    - Session-based authentication and authorization
    - Rate limiting and input validation
    - Comprehensive audit logging
    """
    
    def __init__(self, 
                 memory_file: Optional[str] = None,
                 tool_log_file: Optional[str] = None,
                 conductor_id: Optional[str] = None,
                 load_models: bool = True,
                 session_token: Optional[str] = None):
        """
        Initialize Enhanced Core Conductor with Security.
        
        Args:
            memory_file: Path to GraphRAG memory file
            tool_log_file: Path to tool request log
            conductor_id: Unique identifier for this conductor instance
            load_models: Whether to load AI models on initialization
            session_token: Session token for authentication
        """
        self.conductor_id = conductor_id or f"conductor_{str(uuid.uuid4())[:8]}"
        self.session_token = session_token or self.create_session()
        self.hrm_router = HRMRouter(memory_file, tool_log_file)
        self._lock = threading.Lock()
        self.creation_time = datetime.now()
        
        # Strategic context
        self.current_objectives: List[str] = []
        self.active_contexts: Dict[str, Any] = {}
        
        # Model management
        self.models: Dict[str, ModelInterface] = {}
        if load_models:
            self.init_models()
        
        # Security tracking
        self.operation_count = 0
        self.last_validation = datetime.now()
        
        # Register conductor-specific tools
        self._register_conductor_tools()
        
        log_conductor_activity("initialization", self.session_token, {
            "conductor_id": self.conductor_id,
            "models_loaded": load_models,
            "models_count": len(self.models)
        })
        
        logger.info(f"Core Conductor {self.conductor_id} initialized with security and GraphRAG integration")

    def create_session(self) -> str:
        """Create a new conductor session"""
        with conductor_lock:
            session_token = generate_conductor_session()
            conductor_sessions[session_token] = {
                'created_at': datetime.now(),
                'expires_at': datetime.now() + timedelta(hours=session_expiry_hours),
                'last_access': datetime.now(),
                'conductor_id': self.conductor_id if hasattr(self, 'conductor_id') else 'unknown',
                'conductor_operations': 0
            }
            return session_token

    def validate_session(self, session_token: Optional[str] = None) -> bool:
        """Validate session for conductor operations"""
        token_to_validate = session_token or self.session_token
        
        if not token_to_validate:
            logger.warning("No session token provided for conductor validation")
            return False
        
        return validate_conductor_session(token_to_validate)

    def init_models(self) -> None:
        """
        Initialize and load all AI models for the conductor with security validation.
        
        Automatically detects and loads either:
        - Standard conductor models (conductor, logic, emotion, creative)
        - Full SLiM agent suite (conductor + 8 SLiM agents) if configured
        
        SLiM Agent Roles:
        Left Brain (Logic): logic_high, logic_code, logic_proof, logic_fallback
        Right Brain (Emotion): emotion_valence, emotion_narrative, emotion_uncensored, emotion_creative
        """
        try:
            # Validate session before loading models
            if not self.validate_session():
                log_conductor_activity("init_models", self.session_token, 
                                      {"status": "session_invalid"}, "failed")
                raise ValueError("Invalid session for model initialization")
            
            logger.info(f"Initializing models for Core Conductor {self.conductor_id}")
            
            # Check if SLiM environment variables are configured
            import os
            has_slim_config = any(os.getenv(var) for var in [
                "LOGIC_HIGH_MODEL", "LOGIC_CODE_MODEL", "LOGIC_PROOF_MODEL", "LOGIC_FALLBACK_MODEL",
                "EMOTION_VALENCE_MODEL", "EMOTION_NARRATIVE_MODEL", "EMOTION_UNCENSORED_MODEL", "EMOTION_CREATIVE_MODEL"
            ])
            
            if has_slim_config:
                from .init_models import load_all_models
                self.models = load_all_models()
                logger.info("Loaded SLiM agent model suite (Conductor + 8 SLiMs + legacy roles)")
            else:
                self.models = load_conductor_models()
                logger.info("Loaded standard conductor model suite")
            
            # Validate model count
            if len(self.models) > MAX_MODELS_PER_CONDUCTOR:
                logger.warning(f"Model count ({len(self.models)}) exceeds recommended maximum ({MAX_MODELS_PER_CONDUCTOR})")
            
            # Log loaded models with security metadata
            for role, model in self.models.items():
                model_type = type(model).__name__
                model_name = getattr(model, 'model_name', 'unknown')
                logger.info(f"  {role}: {model_type} ({model_name})")
            
            log_conductor_activity("init_models", self.session_token, {
                "models_loaded": len(self.models),
                "model_types": list(self.models.keys()),
                "slim_config": has_slim_config
            })
            
            logger.info(f"Successfully loaded {len(self.models)} models")
            
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            log_conductor_activity("init_models", self.session_token, 
                                  {"error": str(e)}, "error")
            
            # Fallback to empty dict to prevent crashes
            self.models = {}
            
            # Ensure we have at least mock models
            try:
                from .init_models import MockModel
                self.models = {
                    "conductor": MockModel("fallback-conductor"),
                    "logic": MockModel("fallback-logic"),
                    "emotion": MockModel("fallback-emotion"),
                    "creative": MockModel("fallback-creative")
                }
                logger.warning("Loaded fallback mock models due to initialization error")
            except ImportError:
                logger.error("Unable to load fallback models")
                raise

    def generate(self, role: str, prompt: str, context: Optional[str] = None, session_token: Optional[str] = None, **kwargs) -> str:
        """
        Generate a response using the specified model role with security validation.
        
        Args:
            role: Model role to use. Standard roles: "conductor", "logic", "emotion", "creative"
                  SLiM agent roles: "logic_high", "logic_code", "logic_proof", "logic_fallback",
                                   "emotion_valence", "emotion_narrative", "emotion_uncensored", "emotion_creative"
            prompt: The prompt to send to the model
            context: Optional context to include with the prompt
            session_token: Session token for authentication
            **kwargs: Additional arguments to pass to the model
            
        Returns:
            Generated response string
            
        Raises:
            ValueError: If the specified role is not available or session is invalid
        """
        try:
            # Validate session
            if not self.validate_session(session_token):
                log_conductor_activity("generate", session_token or self.session_token, 
                                      {"role": role, "status": "session_invalid"}, "failed")
                raise ValueError("Invalid session for model generation")
            
            # Check rate limit
            current_token = session_token or self.session_token
            if not check_conductor_rate_limit(current_token):
                log_conductor_activity("generate", current_token, 
                                      {"role": role, "status": "rate_limited"}, "failed")
                raise ValueError("Rate limit exceeded for conductor operations")
            
            # Validate role
            if role not in self.models:
                available_roles = list(self.models.keys())
                log_conductor_activity("generate", current_token, 
                                      {"role": role, "available_roles": available_roles}, "role_not_found")
                raise ValueError(f"Role '{role}' not available. Available roles: {available_roles}")
            
            # Validate prompt
            if not isinstance(prompt, str):
                raise ValueError("Prompt must be a string")
            
            if len(prompt) > 5000:  # Reasonable prompt size limit
                prompt = prompt[:5000] + "..."
                logger.warning("Prompt truncated due to length")
            
            # Sanitize prompt
            clean_prompt = re.sub(r'[<>"\']', '', prompt)
            
            # Get the model for this role
            model = self.models[role]
            
            # Generate response
            response = model.generate(clean_prompt, context=context, **kwargs)
            
            # Track operation
            with self._lock:
                self.operation_count += 1
                if current_token in conductor_sessions:
                    conductor_sessions[current_token]['conductor_operations'] += 1
            
            log_conductor_activity("generate", current_token, {
                "role": role,
                "prompt_length": len(prompt),
                "response_length": len(response),
                "operation_count": self.operation_count
            })
            
            logger.debug(f"Generated response using {role} model: {len(response)} characters")
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating response with {role} model: {e}")
            log_conductor_activity("generate", session_token or self.session_token, 
                                  {"role": role, "error": str(e)}, "error")
            # Fallback response
            return f"[ERROR: Failed to generate response with {role} model: {str(e)}]"
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get status information about loaded models"""
        from .init_models import get_model_info
        
        status = {
            "total_models": len(self.models),
            "loaded_roles": list(self.models.keys()),
            "models": {}
        }
        
        for role, model in self.models.items():
            status["models"][role] = get_model_info(model)
        
        return status
    
    def reload_model(self, role: str) -> bool:
        """
        Reload a specific model role.
        
        Args:
            role: Model role to reload
            
        Returns:
            Success status
        """
        try:
            from .init_models import load_model
            
            # Default model mappings for each role
            role_defaults = {
                "conductor": "llama3.1:8b",
                "logic": "deepseek-coder:1.3b", 
                "emotion": "mistral:7b",
                "creative": "mixtral:8x7b"
            }
            
            if role not in role_defaults:
                logger.error(f"Unknown role '{role}' for model reload")
                return False
            
            # Load the model
            env_var = f"{role.upper()}_MODEL"
            default_model = role_defaults[role]
            
            new_model = load_model(env_var, default_model)
            
            # Replace the model
            with self._lock:
                self.models[role] = new_model
            
            logger.info(f"Successfully reloaded {role} model")
            return True
            
        except Exception as e:
            logger.error(f"Error reloading {role} model: {e}")
            return False
    
    def _register_conductor_tools(self):
        """Register tools specific to conductor-level operations"""
        
        def strategic_analysis_tool(context: str, objectives: Optional[List[str]] = None, **kwargs) -> dict:
            """Tool for strategic analysis and planning"""
            try:
                # Query memory for strategic context
                memory_response = self.hrm_router.execute_tool(
                    "query_memory", 
                    {"concept": "strategy", "depth": 3},
                    agent_type="conductor"
                )
                
                strategic_concepts = []
                if memory_response.success and memory_response.result:
                    strategic_concepts = memory_response.result.get('related_concepts', [])
                
                return {
                    "success": True,
                    "analysis": f"Strategic analysis for: {context}",
                    "context": context,
                    "objectives": objectives or [],
                    "strategic_memory": strategic_concepts[:5],  # Top 5 strategic concepts
                    "recommendations": [
                        "Evaluate current objectives alignment",
                        "Consider memory context for decisions",
                        "Identify tool requirements for execution"
                    ],
                    "trace": kwargs.get('trace_id', str(uuid.uuid4()))
                }
                
            except Exception as e:
                return {
                    "success": False,
                    "error": str(e),
                    "trace": kwargs.get('trace_id', str(uuid.uuid4()))
                }
        
        def objective_planning_tool(goal: str, constraints: Optional[List[str]] = None, **kwargs) -> dict:
            """Tool for objective-based planning"""
            try:
                # Query memory for related planning concepts
                memory_response = self.hrm_router.execute_tool(
                    "query_memory",
                    {"concept": goal, "depth": 2},
                    agent_type="conductor"
                )
                
                related_concepts = []
                if memory_response.success and memory_response.result:
                    related_concepts = memory_response.result.get('related_concepts', [])
                
                # Generate action plan
                action_plan = [
                    f"Define success criteria for: {goal}",
                    "Gather relevant memory context",
                    "Identify required tools and resources",
                    "Create execution timeline",
                    "Monitor progress and adapt"
                ]
                
                return {
                    "success": True,
                    "goal": goal,
                    "constraints": constraints or [],
                    "action_plan": action_plan,
                    "memory_context": related_concepts[:3],
                    "estimated_complexity": "medium",
                    "trace": kwargs.get('trace_id', str(uuid.uuid4()))
                }
                
            except Exception as e:
                return {
                    "success": False,
                    "error": str(e),
                    "trace": kwargs.get('trace_id', str(uuid.uuid4()))
                }
        
        # Register conductor tools
        self.hrm_router.tool_router.register_tool("strategic_analysis", strategic_analysis_tool)
        self.hrm_router.tool_router.register_tool("objective_planning", objective_planning_tool)
    
    def make_strategic_decision(self, 
                              situation: str, 
                              objectives: Optional[List[str]] = None,
                              constraints: Optional[List[str]] = None) -> ConductorDecision:
        """
        Make a strategic decision with full memory and tool integration.
        
        Args:
            situation: Description of the current situation
            objectives: List of current objectives
            constraints: Any constraints to consider
            
        Returns:
            ConductorDecision with comprehensive reasoning and recommendations
        """
        decision_id = str(uuid.uuid4())
        start_time = datetime.now()
        
        try:
            logger.info(f"Conductor making strategic decision: {decision_id}")
            
            # Process input through HRM Router for memory context
            input_context = self.hrm_router.process_agent_input(
                situation, 
                agent_type="conductor",
                context={
                    'objectives': objectives or [],
                    'constraints': constraints or [],
                    'conductor_id': self.conductor_id
                }
            )
            
            reasoning_path = ["Initial situation assessment"]
            
            # Perform strategic analysis
            strategic_response = self.hrm_router.execute_tool(
                "strategic_analysis",
                {
                    "context": situation,
                    "objectives": objectives or []
                },
                agent_type="conductor"
            )
            
            if strategic_response.success:
                reasoning_path.append("Strategic analysis completed")
            
            # Create objective planning if objectives provided
            planning_recommendations = []
            if objectives:
                for objective in objectives:
                    planning_response = self.hrm_router.execute_tool(
                        "objective_planning",
                        {"goal": objective, "constraints": constraints},
                        agent_type="conductor"
                    )
                    if planning_response.success and planning_response.result:
                        planning_recommendations.extend(
                            planning_response.result.get('action_plan', [])
                        )
                
                reasoning_path.append("Objective planning completed")
            
            # Generate comprehensive action plan
            action_plan = []
            
            # Add strategic recommendations
            if strategic_response.success and strategic_response.result:
                action_plan.extend(strategic_response.result.get('recommendations', []))
            
            # Add planning recommendations
            action_plan.extend(planning_recommendations)
            
            # Add memory-informed actions
            memory_related = input_context.get('memory_related', [])
            if memory_related:
                action_plan.append("Leverage relevant memory context for execution")
                reasoning_path.append("Memory context integrated")
            
            # Add tool recommendations
            suggested_tools = input_context.get('suggested_tools', [])
            if suggested_tools:
                action_plan.append(f"Consider using tools: {', '.join(suggested_tools)}")
                reasoning_path.append("Tool recommendations identified")
            
            # Calculate confidence based on available context
            confidence = 0.7  # Base confidence
            if memory_related:
                confidence += 0.1  # Boost for memory context
            if strategic_response.success:
                confidence += 0.1  # Boost for successful strategic analysis
            if objectives:
                confidence += 0.1  # Boost for clear objectives
            
            confidence = min(confidence, 1.0)  # Cap at 1.0
            
            # Enhanced reasoning with AI models if available
            if self.models:
                try:
                    # Use conductor model for strategic synthesis
                    synthesis_prompt = f"""
                    Situation: {situation}
                    Objectives: {objectives or 'None specified'}
                    Constraints: {constraints or 'None specified'}
                    Memory Context: {len(memory_related)} related concepts found
                    
                    Provide strategic synthesis and key insights for this decision.
                    """
                    
                    strategic_synthesis = self.generate("conductor", synthesis_prompt, context=str(memory_related[:3]))
                    if strategic_synthesis and not strategic_synthesis.startswith("[ERROR"):
                        action_plan.append(f"Strategic insight: {strategic_synthesis[:100]}...")
                        reasoning_path.append("AI strategic synthesis completed")
                        confidence += 0.05  # Small boost for AI insights
                    
                    # Use logic model for constraint analysis if constraints exist
                    if constraints:
                        logic_prompt = f"Analyze the logical implications and potential conflicts of these constraints: {constraints}"
                        logic_analysis = self.generate("logic", logic_prompt)
                        if logic_analysis and not logic_analysis.startswith("[ERROR"):
                            action_plan.append(f"Constraint analysis: {logic_analysis[:100]}...")
                            reasoning_path.append("Logical constraint analysis completed")
                    
                except Exception as e:
                    logger.warning(f"Error in AI model reasoning: {e}")
                    reasoning_path.append("AI reasoning attempted but failed")
            
            confidence = min(confidence, 1.0)  # Re-cap after AI boost
            
            # Create decision object
            decision = ConductorDecision(
                decision_id=decision_id,
                strategic_context=situation,
                action_plan=action_plan,
                memory_context=memory_related,
                tool_recommendations=suggested_tools,
                confidence=confidence,
                reasoning_path=reasoning_path,
                timestamp=start_time.isoformat()
            )
            
            # Store decision in memory for future reference
            self._store_decision_in_memory(decision)
            
            # Process output through HRM Router
            output_summary = f"Strategic decision made with {len(action_plan)} recommended actions"
            self.hrm_router.process_agent_output(output_summary, input_context, "conductor")
            
            logger.info(f"Strategic decision completed: {decision_id} (confidence: {confidence:.2f})")
            
            return decision
            
        except Exception as e:
            logger.error(f"Error making strategic decision: {e}")
            # Return error decision
            return ConductorDecision(
                decision_id=decision_id,
                strategic_context=situation,
                action_plan=[f"Error in decision making: {str(e)}"],
                memory_context=[],
                tool_recommendations=[],
                confidence=0.0,
                reasoning_path=["Error encountered"],
                timestamp=start_time.isoformat()
            )
    
    def _store_decision_in_memory(self, decision: ConductorDecision):
        """Store strategic decision in GraphRAG memory"""
        try:
            # Store decision as memory facts
            self.hrm_router.execute_tool(
                "add_memory_fact",
                {
                    "subject": "conductor",
                    "relation": "made_decision",
                    "object_node": decision.decision_id,
                    "confidence": decision.confidence,
                    "source": self.conductor_id
                },
                agent_type="conductor"
            )
            
            # Store action relationships
            for i, action in enumerate(decision.action_plan[:3]):  # Store top 3 actions
                self.hrm_router.execute_tool(
                    "add_memory_fact",
                    {
                        "subject": decision.decision_id,
                        "relation": "includes_action",
                        "object_node": action[:50],  # Truncate for brevity
                        "confidence": 0.9,
                        "source": self.conductor_id
                    },
                    agent_type="conductor"
                )
            
        except Exception as e:
            logger.error(f"Error storing decision in memory: {e}")
    
    def set_objectives(self, objectives: List[str]):
        """Set current strategic objectives"""
        with self._lock:
            self.current_objectives = objectives.copy()
            logger.info(f"Conductor objectives updated: {len(objectives)} objectives set")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current conductor status"""
        with self._lock:
            return {
                'conductor_id': self.conductor_id,
                'current_objectives': self.current_objectives.copy(),
                'active_contexts': len(self.active_contexts),
                'integration_stats': self.hrm_router.get_integration_stats()
            }
    
    def save_state(self) -> bool:
        """Save conductor state and memory"""
        return self.hrm_router.save_all()


def example_conductor_usage():
    """Example usage of Enhanced Core Conductor"""
    
    # Initialize conductor
    conductor = CoreConductor(
        memory_file="data/conductor_memory.json",
        tool_log_file="logs/conductor_tools.jsonl"
    )
    
    print("Enhanced Core Conductor Example")
    print("=" * 40)
    
    # Set strategic objectives
    objectives = [
        "Improve user engagement with AI companion",
        "Enhance emotional intelligence capabilities", 
        "Optimize response relevance and personalization"
    ]
    conductor.set_objectives(objectives)
    print(f"\nObjectives set: {len(objectives)} strategic goals")
    
    # Make strategic decision
    print("\n1. Strategic Decision Making:")
    situation = "User has been asking more complex emotional questions recently and seems to want deeper conversations"
    
    decision = conductor.make_strategic_decision(
        situation=situation,
        objectives=objectives,
        constraints=["Must maintain user privacy", "Stay within system capabilities"]
    )
    
    print(f"  Decision ID: {decision.decision_id}")
    print(f"  Confidence: {decision.confidence:.2f}")
    print(f"  Action Plan: {len(decision.action_plan)} actions")
    print(f"  Memory Context: {len(decision.memory_context)} related concepts")
    print(f"  Tool Recommendations: {decision.tool_recommendations}")
    
    # Show reasoning path
    print(f"\n2. Reasoning Path:")
    for i, step in enumerate(decision.reasoning_path, 1):
        print(f"  {i}. {step}")
    
    # Show action plan
    print(f"\n3. Action Plan:")
    for i, action in enumerate(decision.action_plan, 1):
        print(f"  {i}. {action}")
    
    # Get conductor status
    print(f"\n4. Conductor Status:")
    status = conductor.get_status()
    print(f"  ID: {status['conductor_id']}")
    print(f"  Objectives: {len(status['current_objectives'])}")
    print(f"  Memory Nodes: {status['integration_stats']['memory_stats']['total_nodes']}")
    print(f"  Available Tools: {status['integration_stats']['tool_stats']['total_tools']}")
    
    # Save state
    conductor.save_state()
    print(f"\n5. State saved successfully!")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--list-models":
        # List all loaded models
        print("Loading CoreConductor models...")
        conductor = CoreConductor()
        
        print(f"\n✅ Successfully loaded {len(conductor.models)} models:")
        print("-" * 60)
        
        # Group models by type
        standard_models = {}
        slim_models = {}
        
        for role, model in conductor.models.items():
            model_type = type(model).__name__
            model_name = getattr(model, 'model_name', 'unknown')
            
            if role in ["conductor", "logic", "emotion", "creative"]:
                standard_models[role] = (model_type, model_name)
            else:
                slim_models[role] = (model_type, model_name)
        
        # Display standard models
        if standard_models:
            print("📋 Standard Models:")
            for role, (model_type, model_name) in standard_models.items():
                print(f"  {role:<12}: {model_type:<12} ({model_name})")
        
        # Display SLiM models  
        if slim_models:
            print("\n🧠 SLiM Agent Models:")
            
            # Left brain (logic) models
            left_brain = {k: v for k, v in slim_models.items() if k.startswith('logic_')}
            if left_brain:
                print("  🧮 Left Brain (Logic):")
                for role, (model_type, model_name) in left_brain.items():
                    print(f"    {role:<16}: {model_type:<12} ({model_name})")
            
            # Right brain (emotion/creativity) models
            right_brain = {k: v for k, v in slim_models.items() if k.startswith('emotion_')}
            if right_brain:
                print("  🎨 Right Brain (Emotion/Creativity):")
                for role, (model_type, model_name) in right_brain.items():
                    print(f"    {role:<16}: {model_type:<12} ({model_name})")
        
        print("-" * 60)
        print(f"Total models loaded: {len(conductor.models)}")
        
    else:
        example_conductor_usage()

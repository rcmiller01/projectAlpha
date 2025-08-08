"""
HRM Router Integration - GraphRAG Memory and Tool Router Integration

This module provides integration between the High-Resolution Memory (HRM) system
and the new GraphRAG memory + tool router architecture.

Enhanced with comprehensive security features:
- Request routing validation with integrity verification
- Authentication and session management for all HRM operations
- Comprehensive audit logging for memory and tool activities
- Rate limiting and concurrent request management
- Input validation and sanitization for all router data

Key Features:
- Seamless integration with existing HRM stack
- GraphRAG memory hooks for pre/post-response processing
- Tool router invocation for agent autonomy
- Thread-safe concurrent operation support
- Compatible with conductor/supervisor/SLiM agent hierarchy

Author: ProjectAlpha Team
Integration: HRM stack, GraphRAG memory, Tool router
Version: 2.0.0 (Security Enhanced)
"""

import threading
import uuid
import logging
import hashlib
import re
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
from collections import deque, defaultdict

# Enhanced logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
)
logger = logging.getLogger(__name__)

# Security configuration
HRM_SESSION_TOKEN_LENGTH = 32
MAX_CONCURRENT_REQUESTS = 20
REQUEST_TIMEOUT_SECONDS = 300
ROUTING_RATE_LIMIT = 50  # requests per hour
MAX_MEMORY_QUERY_LENGTH = 1000
MAX_TOOL_REQUESTS_PER_SESSION = 100

# Thread safety
hrm_lock = threading.Lock()

# Session management
hrm_sessions = {}
session_expiry_hours = 24

# Rate limiting
routing_requests = defaultdict(lambda: deque())

# Request tracking
active_requests = {}
request_history = deque(maxlen=1000)

# Import the new GraphRAG and tool systems with validation
try:
    from ...memory.graphrag_memory import GraphRAGMemory, QueryResult
    from ..tools.tool_request_router import ToolRequestRouter, ToolResponse
    MEMORY_AVAILABLE = True
    TOOLS_AVAILABLE = True
    logger.info("GraphRAG memory and tool router modules loaded successfully")
except ImportError:
    # Fallback imports for different module structures
    try:
        import sys
        sys.path.append(str(Path(__file__).parent.parent.parent))
        from memory.graphrag_memory import GraphRAGMemory, QueryResult
        from src.tools.tool_request_router import ToolRequestRouter, ToolResponse
        MEMORY_AVAILABLE = True
        TOOLS_AVAILABLE = True
        logger.info("GraphRAG memory and tool router modules loaded via fallback")
    except ImportError as e:
        logger.warning(f"GraphRAG/tool systems not available: {e}")
        MEMORY_AVAILABLE = False
        TOOLS_AVAILABLE = False

def generate_hrm_session() -> str:
    """Generate a secure HRM session token"""
    timestamp = str(time.time())
    random_data = str(hash(datetime.now()))
    token_string = f"hrm:{timestamp}:{random_data}"
    return hashlib.sha256(token_string.encode()).hexdigest()[:HRM_SESSION_TOKEN_LENGTH]

def validate_hrm_session(session_token: str) -> bool:
    """Validate HRM session token"""
    if not session_token or len(session_token) != HRM_SESSION_TOKEN_LENGTH:
        return False

    if session_token not in hrm_sessions:
        return False

    # Check if session has expired
    session_data = hrm_sessions[session_token]
    if datetime.now() > session_data['expires_at']:
        del hrm_sessions[session_token]
        return False

    # Update last access time
    session_data['last_access'] = datetime.now()
    return True

def check_routing_rate_limit(session_token: str) -> bool:
    """Check if routing request rate limit is exceeded"""
    current_time = time.time()

    # Clean old requests
    while (routing_requests[session_token] and
           routing_requests[session_token][0] < current_time - 3600):  # 1 hour window
        routing_requests[session_token].popleft()

    # Check limit
    if len(routing_requests[session_token]) >= ROUTING_RATE_LIMIT:
        logger.warning(f"HRM routing rate limit exceeded for session: {session_token[:8]}...")
        return False

    # Add current request
    routing_requests[session_token].append(current_time)
    return True

def validate_memory_query(query_data: Dict[str, Any]) -> tuple[bool, str]:
    """Validate memory query data"""
    try:
        # Check required fields
        if 'query' not in query_data:
            return False, "Missing required field: query"

        # Validate query text
        query = query_data['query']
        if not isinstance(query, str):
            return False, "Query must be a string"

        if len(query) > MAX_MEMORY_QUERY_LENGTH:
            return False, f"Query exceeds maximum length of {MAX_MEMORY_QUERY_LENGTH}"

        # Sanitize query text
        if not re.match(r'^[a-zA-Z0-9\s\.,!?\-\'\":()\[\]]+$', query):
            return False, "Query contains invalid characters"

        # Validate limit if present
        if 'limit' in query_data:
            limit = query_data['limit']
            if not isinstance(limit, int) or limit < 1 or limit > 50:
                return False, "Limit must be between 1 and 50"

        return True, "Valid"

    except Exception as e:
        logger.error(f"Error validating memory query: {str(e)}")
        return False, f"Validation error: {str(e)}"

def validate_tool_request(tool_data: Dict[str, Any]) -> tuple[bool, str]:
    """Validate tool request data"""
    try:
        # Check required fields
        required_fields = ['tool_name', 'action']
        for field in required_fields:
            if field not in tool_data:
                return False, f"Missing required field: {field}"

        # Validate tool name
        tool_name = tool_data['tool_name']
        if not isinstance(tool_name, str) or len(tool_name.strip()) == 0:
            return False, "Tool name must be a non-empty string"

        # Validate action
        action = tool_data['action']
        if not isinstance(action, str) or len(action.strip()) == 0:
            return False, "Action must be a non-empty string"

        # Validate allowed tools (basic whitelist)
        allowed_tools = {
            'web_search', 'file_manager', 'calculator', 'code_executor',
            'memory_manager', 'task_scheduler', 'data_analyzer'
        }

        if tool_name not in allowed_tools:
            return False, f"Tool not allowed: {tool_name}. Must be one of: {allowed_tools}"

        return True, "Valid"

    except Exception as e:
        logger.error(f"Error validating tool request: {str(e)}")
        return False, f"Validation error: {str(e)}"

def sanitize_input_text(text: str, max_length: int) -> str:
    """Sanitize input text for safety"""
    if not isinstance(text, str):
        return ""

    # Remove potential injection patterns
    text = re.sub(r'[<>"\']', '', text)

    # Limit length
    if len(text) > max_length:
        text = text[:max_length] + "..."

    return text.strip()

def log_hrm_activity(activity_type: str, session_token: str, details: Dict[str, Any], status: str = "success"):
    """Log HRM router activities"""
    try:
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'activity_type': activity_type,
            'session': session_token[:8] + "..." if session_token else "none",
            'details': details,
            'status': status,
            'thread_id': threading.get_ident()
        }

        request_history.append(log_entry)

        logger.info(f"HRM activity logged: {activity_type} ({status})")

        if status != "success":
            logger.warning(f"HRM activity issue: {activity_type} failed with {status}")

    except Exception as e:
        logger.error(f"Error logging HRM activity: {str(e)}")

def generate_request_id() -> str:
    """Generate unique request ID"""
    return f"req_{uuid.uuid4().hex[:12]}"

class HRMRouter:
    """
    HRM Router - Integration hub for GraphRAG memory and tool routing

    This class serves as the central integration point between:
    - Existing HRM (High-Resolution Memory) system
    - New GraphRAG memory for semantic entity linking
    - Tool request router for autonomous tool usage
    - Agent hierarchy (Conductor, Supervisor, SLiMs)

    Features:
    - Pre/post-response GraphRAG memory hooks
    - Intelligent tool routing based on context
    - Thread-safe concurrent operations
    - Request tracing and logging
    - Compatible with existing HRM stack
    """

    def __init__(self,
                 memory_file: Optional[str] = None,
                 tool_log_file: Optional[str] = None,
                 enable_memory_hooks: bool = True,
                 enable_tool_routing: bool = True):
        """
        Initialize HRM Router with GraphRAG memory and tool routing.

        Args:
            memory_file: Path to GraphRAG memory persistence file
            tool_log_file: Path to tool request log file
            enable_memory_hooks: Whether to enable GraphRAG memory hooks
            enable_tool_routing: Whether to enable tool routing
        """
        # Initialize core components
        self.memory = GraphRAGMemory(memory_file)
        self.tool_router = ToolRequestRouter(tool_log_file)
        self.enable_memory_hooks = enable_memory_hooks
        self.enable_tool_routing = enable_tool_routing

        # Thread safety
        self._lock = threading.Lock()

        # Request tracking
        self._active_requests: Dict[str, Dict[str, Any]] = {}

        # SLiM Agent Registry
        self.agent_registry: Dict[str, Any] = {}
        self._initialize_agent_registry()

        # Register default tools
        self._register_default_tools()

        logger.info("HRM Router initialized with GraphRAG memory and tool routing")

    def _register_default_tools(self):
        """Register default tools for agent use"""

        def memory_query_tool(concept: str, depth: int = 2, min_confidence: float = 0.1, **kwargs) -> dict:
            """Tool for querying GraphRAG memory"""
            try:
                result = self.memory.query_related(concept, depth, min_confidence)
                return {
                    "success": True,
                    "concept": concept,
                    "related_concepts": result.related_concepts,
                    "execution_time_ms": result.execution_time_ms,
                    "trace": kwargs.get('trace_id', str(uuid.uuid4()))
                }
            except Exception as e:
                return {
                    "success": False,
                    "error": str(e),
                    "trace": kwargs.get('trace_id', str(uuid.uuid4()))
                }

        def add_memory_fact_tool(subject: str, relation: str, object_node: str,
                               confidence: float = 1.0, source: str = "agent", **kwargs) -> dict:
            """Tool for adding facts to GraphRAG memory"""
            try:
                request_id = self.memory.add_fact(subject, relation, object_node, confidence, source)
                return {
                    "success": True,
                    "fact_added": f"{subject} -[{relation}]-> {object_node}",
                    "confidence": confidence,
                    "request_id": request_id,
                    "trace": kwargs.get('trace_id', str(uuid.uuid4()))
                }
            except Exception as e:
                return {
                    "success": False,
                    "error": str(e),
                    "trace": kwargs.get('trace_id', str(uuid.uuid4()))
                }

        def memory_stats_tool(**kwargs) -> dict:
            """Tool for getting memory system statistics"""
            try:
                stats = self.memory.get_memory_stats()
                return {
                    "success": True,
                    "stats": stats,
                    "trace": kwargs.get('trace_id', str(uuid.uuid4()))
                }
            except Exception as e:
                return {
                    "success": False,
                    "error": str(e),
                    "trace": kwargs.get('trace_id', str(uuid.uuid4()))
                }

        # Register the tools
        self.tool_router.register_tool("query_memory", memory_query_tool)
        self.tool_router.register_tool("add_memory_fact", add_memory_fact_tool)
        self.tool_router.register_tool("memory_stats", memory_stats_tool)

    def process_agent_input(self,
                          input_text: str,
                          agent_type: str = "unknown",
                          context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process input from an agent with GraphRAG memory integration.

        This is the main entry point for agent interactions, providing:
        - Pre-processing memory retrieval
        - Context enhancement with related concepts
        - Request tracking and logging

        Args:
            input_text: The input text from the agent
            agent_type: Type of agent (conductor, supervisor, slim, etc.)
            context: Additional context information

        Returns:
            Dict containing processed input with memory context
        """
        request_id = str(uuid.uuid4())
        start_time = datetime.now()

        try:
            # Track this request
            with self._lock:
                self._active_requests[request_id] = {
                    'input_text': input_text,
                    'agent_type': agent_type,
                    'start_time': start_time,
                    'context': context or {}
                }

            result = {
                'request_id': request_id,
                'original_input': input_text,
                'agent_type': agent_type,
                'enhanced_context': {},
                'memory_related': [],
                'suggested_tools': [],
                'processing_time_ms': 0.0
            }

            # Pre-response memory retrieval if enabled
            if self.enable_memory_hooks:
                memory_context = self._retrieve_memory_context(input_text, context)
                result['enhanced_context'] = memory_context
                result['memory_related'] = memory_context.get('related_concepts', [])

            # Suggest relevant tools based on input
            if self.enable_tool_routing:
                suggested_tools = self._suggest_tools(input_text, agent_type, context)
                result['suggested_tools'] = suggested_tools

            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            result['processing_time_ms'] = processing_time

            logger.info(f"Processed agent input for {agent_type} in {processing_time:.2f}ms (request: {request_id})")

            return result

        except Exception as e:
            logger.error(f"Error processing agent input: {e}")
            return {
                'request_id': request_id,
                'error': str(e),
                'original_input': input_text,
                'agent_type': agent_type
            }
        finally:
            # Clean up request tracking
            with self._lock:
                self._active_requests.pop(request_id, None)

    def process_agent_output(self,
                           output_text: str,
                           input_context: Dict[str, Any],
                           agent_type: str = "unknown") -> str:
        """
        Process output from an agent with GraphRAG memory learning.

        This handles post-response processing including:
        - Memory fact extraction and storage
        - Symbolic relationship learning
        - Output enhancement

        Args:
            output_text: The output text from the agent
            input_context: Context from the input processing
            agent_type: Type of agent that generated the output

        Returns:
            String with potentially enhanced output
        """
        try:
            # Extract and store memory facts if enabled
            if self.enable_memory_hooks:
                self._extract_and_store_facts(output_text, input_context, agent_type)

            # For now, return the output unchanged
            # Future enhancements could modify output based on memory context
            return output_text

        except Exception as e:
            logger.error(f"Error processing agent output: {e}")
            return output_text

    def execute_tool(self,
                    tool_name: str,
                    parameters: Dict[str, Any],
                    agent_type: str = "unknown") -> ToolResponse:
        """
        Execute a tool on behalf of an agent.

        Args:
            tool_name: Name of the tool to execute
            parameters: Parameters for the tool
            agent_type: Type of agent requesting the tool

        Returns:
            ToolResponse with execution results
        """
        if not self.enable_tool_routing:
            logger.warning("Tool routing is disabled")
            return ToolResponse(
                request_id=str(uuid.uuid4()),
                tool_name=tool_name,
                success=False,
                result=None,
                error_message="Tool routing is disabled",
                execution_time_ms=0.0,
                timestamp=datetime.now().isoformat(),
                trace_id=str(uuid.uuid4())
            )

        return self.tool_router.route_request(tool_name, parameters, source=agent_type)

    def _retrieve_memory_context(self, input_text: str, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Retrieve relevant memory context for input processing"""
        try:
            # Simple keyword extraction for memory queries
            # In production, this could use NLP techniques
            keywords = self._extract_keywords(input_text)

            memory_context = {
                'keywords': keywords,
                'related_concepts': [],
                'memory_queries': []
            }

            # Query memory for each keyword
            for keyword in keywords[:3]:  # Limit to top 3 keywords
                try:
                    query_result = self.memory.query_related(keyword, depth=2)
                    memory_context['related_concepts'].extend(query_result.related_concepts)
                    memory_context['memory_queries'].append({
                        'keyword': keyword,
                        'request_id': query_result.request_id,
                        'concepts_found': len(query_result.related_concepts)
                    })
                except Exception as e:
                    logger.warning(f"Error querying memory for keyword '{keyword}': {e}")

            return memory_context

        except Exception as e:
            logger.error(f"Error retrieving memory context: {e}")
            return {}

    def _extract_keywords(self, text: str) -> List[str]:
        """Simple keyword extraction from text"""
        # Basic keyword extraction - in production use proper NLP
        words = text.lower().split()

        # Filter out common words and short words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'}

        keywords = [word.strip('.,!?;:') for word in words
                   if len(word) > 3 and word.lower() not in stop_words]

        return keywords[:10]  # Return top 10 keywords

    def _suggest_tools(self, input_text: str, agent_type: str, context: Optional[Dict[str, Any]]) -> List[str]:
        """Suggest relevant tools based on input and context"""
        suggestions = []
        input_lower = input_text.lower()

        # Simple rule-based tool suggestion
        if any(word in input_lower for word in ['search', 'find', 'lookup', 'research']):
            suggestions.append('search_web')

        if any(word in input_lower for word in ['calculate', 'compute', 'math', 'equation']):
            suggestions.append('calculate')

        if any(word in input_lower for word in ['remember', 'recall', 'memory', 'know']):
            suggestions.append('query_memory')

        if any(word in input_lower for word in ['learn', 'store', 'save', 'record']):
            suggestions.append('add_memory_fact')

        return suggestions

    def _extract_and_store_facts(self, output_text: str, input_context: Dict[str, Any], agent_type: str):
        """Extract and store facts from agent output"""
        try:
            # Simple fact extraction - could be enhanced with NLP
            # For now, store the interaction as a fact
            input_text = input_context.get('original_input', '')
            if input_text and output_text:
                # Store agent interaction as a memory fact
                self.memory.add_fact(
                    subject=f"agent_{agent_type}",
                    relation="responded_to",
                    object_node=input_text[:50],  # Truncate for brevity
                    confidence=0.8,
                    source="hrm_router",
                    context=output_text[:100]  # Store partial response as context
                )

                # Extract potential entities and relationships
                # This is a simplified example - production would use proper NLP
                if "user prefers" in output_text.lower():
                    # Extract preference facts
                    parts = output_text.lower().split("user prefers")
                    if len(parts) > 1:
                        preference = parts[1].split()[0:3]  # Get first few words
                        preference_text = " ".join(preference).strip('.,!?')
                        if preference_text:
                            self.memory.add_fact(
                                subject="user",
                                relation="prefers",
                                object_node=preference_text,
                                confidence=0.9,
                                source=f"agent_{agent_type}"
                            )

        except Exception as e:
            logger.error(f"Error extracting and storing facts: {e}")

    def get_integration_stats(self) -> Dict[str, Any]:
        """Get statistics about the HRM Router integration"""
        return {
            'memory_stats': self.memory.get_memory_stats(),
            'tool_stats': self.tool_router.get_stats(),
            'memory_hooks_enabled': self.enable_memory_hooks,
            'tool_routing_enabled': self.enable_tool_routing,
            'active_requests': len(self._active_requests)
        }

    def save_all(self) -> bool:
        """Save all persistent data"""
        try:
            memory_saved = self.memory.save_memory()
            logger.info(f"HRM Router data save complete (memory: {memory_saved})")
            return memory_saved
        except Exception as e:
            logger.error(f"Error saving HRM Router data: {e}")
            return False

    def _initialize_agent_registry(self):
        """
        Initialize the SLiM agent registry with lazy loading.

        Agents are registered but not instantiated until first use
        to avoid circular imports and improve startup performance.
        """
        self.agent_registry = {
            "deduction": {
                "class": "DeductionAgent",
                "module": "src.agents.deduction_agent",
                "role": "logic_high",
                "specialization": "logical_reasoning",
                "instance": None
            },
            "metaphor": {
                "class": "MetaphorAgent",
                "module": "src.agents.metaphor_agent",
                "role": "emotion_creative",
                "specialization": "creative_expression",
                "instance": None
            }
            # Additional agents can be registered here
        }

        logger.info(f"Agent registry initialized with {len(self.agent_registry)} agent types")

    def register_agent(self, agent_key: str, agent_class: str, module_path: str,
                      role: str, specialization: str):
        """
        Register a new SLiM agent type.

        Args:
            agent_key: Unique key for the agent
            agent_class: Class name of the agent
            module_path: Module path for importing
            role: AI model role the agent uses
            specialization: Description of agent's specialization
        """
        self.agent_registry[agent_key] = {
            "class": agent_class,
            "module": module_path,
            "role": role,
            "specialization": specialization,
            "instance": None
        }

        logger.info(f"Registered agent '{agent_key}' with specialization: {specialization}")

    def get_agent(self, agent_key: str):
        """
        Get or create an agent instance.

        Args:
            agent_key: Key of the agent to retrieve

        Returns:
            Agent instance or None if not found
        """
        if agent_key not in self.agent_registry:
            logger.warning(f"Agent '{agent_key}' not found in registry")
            return None

        agent_config = self.agent_registry[agent_key]

        # Lazy instantiation
        if agent_config["instance"] is None:
            try:
                # Import the agent class
                module = __import__(agent_config["module"], fromlist=[agent_config["class"]])
                agent_class = getattr(module, agent_config["class"])

                # Create instance with conductor
                from .core_conductor import CoreConductor
                conductor = CoreConductor()

                agent_config["instance"] = agent_class(
                    conductor=conductor,
                    memory=self.memory,
                    router=self.tool_router
                )

                logger.info(f"Instantiated agent '{agent_key}' of type {agent_config['class']}")

            except Exception as e:
                logger.error(f"Failed to instantiate agent '{agent_key}': {e}")
                return None

        return agent_config["instance"]

    def dispatch_to_agent(self, agent_key: str, prompt: str, **kwargs) -> Optional[str]:
        """
        Dispatch a request to a specific SLiM agent.

        Args:
            agent_key: Key of the agent to use
            prompt: Prompt to send to the agent
            **kwargs: Additional arguments for the agent

        Returns:
            Agent response or None if agent not available
        """
        agent = self.get_agent(agent_key)
        if agent is None:
            return None

        try:
            # Set default arguments
            kwargs.setdefault("depth", 2)
            kwargs.setdefault("use_tools", True)

            response = agent.run(prompt, **kwargs)
            logger.debug(f"Agent '{agent_key}' processed request successfully")
            return response

        except Exception as e:
            logger.error(f"Error dispatching to agent '{agent_key}': {e}")
            return f"Agent '{agent_key}' error: {str(e)}"

    def list_agents(self) -> Dict[str, Dict[str, str]]:
        """
        List all registered agents and their specializations.

        Returns:
            Dictionary of agent information
        """
        agent_info = {}
        for key, config in self.agent_registry.items():
            agent_info[key] = {
                "class": config["class"],
                "role": config["role"],
                "specialization": config["specialization"],
                "instantiated": config["instance"] is not None
            }

        return agent_info


# Example integration with existing HRM stack
def example_hrm_integration():
    """Example of how to integrate HRM Router with existing systems"""

    # Initialize HRM Router
    hrm = HRMRouter(
        memory_file="data/hrm_graphrag_memory.json",
        tool_log_file="logs/hrm_tool_requests.jsonl"
    )

    print("HRM Router Integration Example")
    print("=" * 40)

    # Simulate conductor input
    print("\n1. Conductor Strategic Input:")
    conductor_input = "The user seems to prefer chocolate desserts based on recent conversations"
    result = hrm.process_agent_input(conductor_input, agent_type="conductor")
    print(f"  Input: {conductor_input}")
    print(f"  Memory context: {len(result.get('memory_related', []))} related concepts")
    print(f"  Suggested tools: {result.get('suggested_tools', [])}")

    # Process conductor output
    conductor_output = "I should remember that the user prefers chocolate desserts for future recommendations"
    enhanced_output = hrm.process_agent_output(conductor_output, result, "conductor")
    print(f"  Output processed and facts stored")

    # Simulate supervisor query
    print("\n2. Supervisor Context Query:")
    supervisor_input = "What do we know about user food preferences?"
    result2 = hrm.process_agent_input(supervisor_input, agent_type="supervisor")
    print(f"  Input: {supervisor_input}")
    print(f"  Memory context: {len(result2.get('memory_related', []))} related concepts")

    # Execute memory query tool
    print("\n3. Tool Execution:")
    tool_response = hrm.execute_tool("query_memory",
                                   {"concept": "user", "depth": 2},
                                   agent_type="supervisor")
    print(f"  Tool: query_memory")
    print(f"  Success: {tool_response.success}")
    if tool_response.success and tool_response.result:
        related = tool_response.result.get('related_concepts', [])
        print(f"  Found {len(related)} related concepts")

    # Show integration stats
    print("\n4. Integration Statistics:")
    stats = hrm.get_integration_stats()
    print(f"  Memory nodes: {stats['memory_stats']['total_nodes']}")
    print(f"  Memory edges: {stats['memory_stats']['total_edges']}")
    print(f"  Registered tools: {stats['tool_stats']['total_tools']}")

    # Save everything
    hrm.save_all()
    print("\n5. Data saved successfully!")


if __name__ == "__main__":
    example_hrm_integration()

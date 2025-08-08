"""
Tool Request Router - Lightweight modular tool routing system

This module implements a thread-safe tool router that maps string intents to callable functions,
enabling autonomous AI agents to access and execute tools dynamically.

Key Features:
- Thread-safe tool registration and execution
- UUID-based request tracing for concurrent operations
- Comprehensive logging of all tool requests and responses
- Stateless design for async orchestration compatibility
- Easy integration with HRM stack and future SLiM agents

Author: ProjectAlpha Team
Compatible with: HRM stack, SLiM agent integration
"""

import inspect
import json
import logging
import threading
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Dict, List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ToolRequest:
    """Structured tool request with metadata"""

    request_id: str
    tool_name: str
    intent: str
    parameters: dict[str, Any]
    timestamp: str
    source: str = "unknown"


@dataclass
class ToolResponse:
    """Structured tool response with metadata"""

    request_id: str
    tool_name: str
    success: bool
    result: Any
    error_message: Optional[str]
    execution_time_ms: float
    timestamp: str
    trace_id: str


class ToolRequestRouter:
    """
    Thread-safe tool request router for autonomous AI agent systems.

    This class provides a modular system for registering and routing tool requests
    where AI agents can dynamically discover and execute tools based on string intents.

    Features:
    - Thread-safe tool registration and execution
    - Automatic request/response logging
    - UUID-based request tracing
    - Tool discovery and introspection
    - Stateless design for scalability
    """

    def __init__(self, log_file: Optional[str] = None, enable_logging: bool = True):
        """
        Initialize the tool request router.

        Args:
            log_file: Path to log file for tool requests. If None, uses default location.
            enable_logging: Whether to enable request/response logging
        """
        self._tools: dict[str, Callable] = {}
        self._tool_metadata: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()
        self.log_file = log_file or "logs/tool_requests.jsonl"
        self.enable_logging = enable_logging
        self._ensure_log_directory()

        logger.info("ToolRequestRouter initialized")

    def _ensure_log_directory(self):
        """Ensure the log directory exists"""
        if self.enable_logging:
            Path(self.log_file).parent.mkdir(parents=True, exist_ok=True)

    def _log_request(self, request: ToolRequest, response: ToolResponse):
        """Log tool request and response to file"""
        if not self.enable_logging:
            return

        try:
            log_entry = {
                "request": asdict(request),
                "response": asdict(response),
                "logged_at": datetime.now().isoformat(),
            }

            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry) + "\n")

        except Exception as e:
            logger.error(f"Error logging tool request: {e}")

    def _extract_tool_metadata(self, handler: Callable) -> dict[str, Any]:
        """Extract metadata from tool function"""
        try:
            sig = inspect.signature(handler)
            doc = inspect.getdoc(handler) or "No description available"

            parameters = {}
            for param_name, param in sig.parameters.items():
                param_info = {
                    "type": str(param.annotation)
                    if param.annotation != inspect.Parameter.empty
                    else "Any",
                    "required": param.default == inspect.Parameter.empty,
                    "default": str(param.default)
                    if param.default != inspect.Parameter.empty
                    else None,
                }
                parameters[param_name] = param_info

            return {
                "description": doc,
                "parameters": parameters,
                "return_type": str(sig.return_annotation)
                if sig.return_annotation != inspect.Signature.empty
                else "Any",
                "module": handler.__module__,
                "registered_at": datetime.now().isoformat(),
            }
        except Exception as e:
            logger.warning(f"Could not extract metadata for tool: {e}")
            return {
                "description": "Metadata extraction failed",
                "parameters": {},
                "return_type": "Any",
                "module": "unknown",
                "registered_at": datetime.now().isoformat(),
            }

    def register_tool(
        self, name: str, handler: Callable, description: Optional[str] = None
    ) -> bool:
        """
        Register a tool function with the router.

        Args:
            name: Unique name for the tool
            handler: Callable function that implements the tool
            description: Optional description override

        Returns:
            bool: Success status
        """
        if not callable(handler):
            logger.error(f"Handler for tool '{name}' is not callable")
            return False

        with self._lock:
            try:
                # Extract metadata
                metadata = self._extract_tool_metadata(handler)
                if description:
                    metadata["description"] = description

                # Register tool
                self._tools[name] = handler
                self._tool_metadata[name] = metadata

                logger.info(f"Tool '{name}' registered successfully")
                return True

            except Exception as e:
                logger.error(f"Error registering tool '{name}': {e}")
                return False

    def unregister_tool(self, name: str) -> bool:
        """
        Unregister a tool from the router.

        Args:
            name: Name of the tool to unregister

        Returns:
            bool: Success status
        """
        with self._lock:
            if name in self._tools:
                del self._tools[name]
                del self._tool_metadata[name]
                logger.info(f"Tool '{name}' unregistered")
                return True
            else:
                logger.warning(f"Tool '{name}' not found for unregistration")
                return False

    def route_request(
        self, intent: str, parameters: Optional[dict[str, Any]] = None, source: str = "unknown"
    ) -> ToolResponse:
        """
        Route a tool request based on intent string.

        Args:
            intent: String identifying the tool to execute
            parameters: Dictionary of parameters to pass to the tool
            source: Source identifier for the request

        Returns:
            ToolResponse: Structured response with result or error
        """
        request_id = str(uuid.uuid4())
        trace_id = str(uuid.uuid4())
        start_time = datetime.now()

        # Create request object
        request = ToolRequest(
            request_id=request_id,
            tool_name=intent,
            intent=intent,
            parameters=parameters or {},
            timestamp=start_time.isoformat(),
            source=source,
        )

        # Default response for errors
        def create_error_response(error_msg: str) -> ToolResponse:
            return ToolResponse(
                request_id=request_id,
                tool_name=intent,
                success=False,
                result=None,
                error_message=error_msg,
                execution_time_ms=0.0,
                timestamp=datetime.now().isoformat(),
                trace_id=trace_id,
            )

        # Check if tool exists
        with self._lock:
            if intent not in self._tools:
                error_msg = (
                    f"Tool '{intent}' not found. Available tools: {list(self._tools.keys())}"
                )
                logger.warning(error_msg)
                response = create_error_response(error_msg)
                self._log_request(request, response)
                return response

            tool_handler = self._tools[intent]

        # Execute tool
        try:
            logger.info(f"Executing tool '{intent}' with request_id: {request_id}")

            # Add trace_id to parameters if tool supports it
            exec_params = parameters.copy() if parameters else {}
            sig = inspect.signature(tool_handler)
            if "trace_id" in sig.parameters:
                exec_params["trace_id"] = trace_id
            if "request_id" in sig.parameters:
                exec_params["request_id"] = request_id

            # Execute the tool
            result = tool_handler(**exec_params)

            # Calculate execution time
            execution_time = (datetime.now() - start_time).total_seconds() * 1000

            # Create success response
            response = ToolResponse(
                request_id=request_id,
                tool_name=intent,
                success=True,
                result=result,
                error_message=None,
                execution_time_ms=execution_time,
                timestamp=datetime.now().isoformat(),
                trace_id=trace_id,
            )

            logger.info(f"Tool '{intent}' executed successfully in {execution_time:.2f}ms")

        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            error_msg = f"Error executing tool '{intent}': {e!s}"
            logger.error(error_msg)

            response = ToolResponse(
                request_id=request_id,
                tool_name=intent,
                success=False,
                result=None,
                error_message=error_msg,
                execution_time_ms=execution_time,
                timestamp=datetime.now().isoformat(),
                trace_id=trace_id,
            )

        # Log the request/response
        self._log_request(request, response)
        return response

    def list_tools(self) -> dict[str, dict[str, Any]]:
        """
        Get list of all registered tools with their metadata.

        Returns:
            Dict mapping tool names to their metadata
        """
        with self._lock:
            return self._tool_metadata.copy()

    def get_tool_info(self, name: str) -> Optional[dict[str, Any]]:
        """
        Get detailed information about a specific tool.

        Args:
            name: Tool name

        Returns:
            Tool metadata dict or None if not found
        """
        with self._lock:
            return self._tool_metadata.get(name)

    def tool_exists(self, name: str) -> bool:
        """Check if a tool is registered"""
        with self._lock:
            return name in self._tools

    def get_stats(self) -> dict[str, Any]:
        """Get router statistics"""
        with self._lock:
            return {
                "total_tools": len(self._tools),
                "registered_tools": list(self._tools.keys()),
                "log_file": self.log_file,
                "logging_enabled": self.enable_logging,
            }


# Utility decorators for tool development
def tool_handler(name: str, description: str = ""):
    """Decorator to mark functions as tool handlers"""

    def decorator(func: Callable) -> Callable:
        func._tool_name = name
        func._tool_description = description
        return func

    return decorator


def requires_params(*param_names: str):
    """Decorator to validate required parameters for tools"""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            for param in param_names:
                if param not in kwargs or kwargs[param] is None:
                    raise ValueError(f"Required parameter '{param}' is missing or None")
            return func(*args, **kwargs)

        return wrapper

    return decorator


# Example tool implementations
@tool_handler("search_web", "Search the web for information")
@requires_params("query")
def example_web_search_tool(query: str, max_results: int = 5, **kwargs) -> dict:
    """
    Example web search tool implementation.

    Args:
        query: Search query string
        max_results: Maximum number of results to return

    Returns:
        dict: Search results with metadata
    """
    # Simulate web search
    return {
        "result": f"Web search results for: {query}",
        "results": [
            {
                "title": f"Result {i+1}",
                "url": f"https://example.com/{i+1}",
                "snippet": f"Sample snippet {i+1}",
            }
            for i in range(min(max_results, 3))
        ],
        "total_results": min(max_results, 3),
        "trace": kwargs.get("trace_id", str(uuid.uuid4())),
    }


@tool_handler("calculate", "Perform mathematical calculations")
@requires_params("expression")
def example_calculator_tool(expression: str, **kwargs) -> dict:
    """
    Example calculator tool implementation.

    Args:
        expression: Mathematical expression to evaluate

    Returns:
        dict: Calculation result with metadata
    """
    try:
        # Simple evaluation (in production, use a safer math parser)
        result = eval(expression)
        return {
            "result": f"Calculation result: {result}",
            "expression": expression,
            "value": result,
            "trace": kwargs.get("trace_id", str(uuid.uuid4())),
        }
    except Exception as e:
        return {
            "result": f"Calculation error: {e!s}",
            "expression": expression,
            "error": str(e),
            "trace": kwargs.get("trace_id", str(uuid.uuid4())),
        }


@tool_handler("memory_query", "Query the GraphRAG memory system")
@requires_params("concept")
def example_memory_query_tool(concept: str, depth: int = 2, **kwargs) -> dict:
    """
    Example memory query tool that interfaces with GraphRAG.

    Args:
        concept: Concept to query
        depth: Query depth

    Returns:
        dict: Memory query results
    """
    # This would integrate with the GraphRAG memory system
    return {
        "result": f"Memory query for concept: {concept}",
        "concept": concept,
        "depth": depth,
        "related_concepts": [f"related_concept_{i}" for i in range(3)],
        "trace": kwargs.get("trace_id", str(uuid.uuid4())),
    }


def example_usage():
    """Example usage of the Tool Request Router"""

    # Initialize router
    router = ToolRequestRouter("logs/example_tool_requests.jsonl")

    # Register example tools
    print("Registering example tools...")
    router.register_tool("search_web", example_web_search_tool)
    router.register_tool("calculate", example_calculator_tool)
    router.register_tool("memory_query", example_memory_query_tool)

    # List available tools
    print(f"\nAvailable tools: {list(router.list_tools().keys())}")

    # Example tool requests
    print("\nExecuting example tool requests:")

    # Web search
    response1 = router.route_request("search_web", {"query": "AI memory systems", "max_results": 3})
    print(f"Web search: {response1.success}, Result: {response1.result}")

    # Calculator
    response2 = router.route_request("calculate", {"expression": "2 + 2 * 3"})
    print(f"Calculator: {response2.success}, Result: {response2.result}")

    # Memory query
    response3 = router.route_request("memory_query", {"concept": "user_preferences", "depth": 2})
    print(f"Memory query: {response3.success}, Result: {response3.result}")

    # Error case
    response4 = router.route_request("nonexistent_tool", {})
    print(f"Error case: {response4.success}, Error: {response4.error_message}")

    # Show router stats
    print(f"\nRouter stats: {router.get_stats()}")


if __name__ == "__main__":
    example_usage()

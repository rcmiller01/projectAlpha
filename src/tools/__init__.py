"""
Tools Package - Modular tool routing and management system

This package provides tools for autonomous AI agent systems including:
- Tool request routing and management
- Thread-safe tool registration and execution
- Integration with GraphRAG memory systems

Author: ProjectAlpha Team
"""

from .tool_request_router import (
    ToolRequestRouter,
    ToolRequest,
    ToolResponse,
    tool_handler,
    requires_params,
    example_web_search_tool,
    example_calculator_tool,
    example_memory_query_tool
)

__all__ = [
    'ToolRequestRouter',
    'ToolRequest', 
    'ToolResponse',
    'tool_handler',
    'requires_params',
    'example_web_search_tool',
    'example_calculator_tool',
    'example_memory_query_tool'
]

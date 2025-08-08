"""
Base SLiMAgent Class

This module provides the foundational SLiMAgent class that integrates with the
ProjectAlpha GraphRAG memory system, tool router, and HRM conductor pipeline.

Each SLiM agent specializes in specific cognitive tasks while maintaining
access to shared memory, tools, and the multi-model conductor system.

Features:
- Integration with GraphRAG semantic memory
- Access to tool request router for autonomous capabilities
- Role-specific model usage via CoreConductor
- Memory fact extraction and drift logging
- Configurable reasoning depth and context retrieval

Author: ProjectAlpha Team
Compatible with: GraphRAG memory, Tool router, HRM stack, SLiM model system
"""

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

# Import core system components
try:
    from ...memory.graphrag_memory import GraphRAGMemory
    from ..core.core_conductor import CoreConductor
    from ..tools.tool_request_router import ToolRequestRouter
except ImportError:
    from memory.graphrag_memory import GraphRAGMemory
    from src.core.core_conductor import CoreConductor
    from src.tools.tool_request_router import ToolRequestRouter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AgentResponse:
    """Structured response from a SLiM agent"""

    agent_id: str
    role: str
    prompt: str
    response: str
    context_facts: list[dict[str, Any]]
    memory_updates: list[dict[str, Any]]
    tool_calls: list[str]
    reasoning_depth: int
    timestamp: str
    confidence: float


class SLiMAgent:
    """
    Base SLiM (Specialized Large Intelligence Models) Agent

    Provides foundation for specialized AI agents that combine:
    - Role-specific AI model capabilities
    - GraphRAG semantic memory access
    - Autonomous tool usage
    - Memory fact extraction and learning

    Each SLiM agent is designed for specific cognitive tasks while
    maintaining shared access to the projectAlpha AI infrastructure.
    """

    def __init__(
        self,
        role: str,
        conductor: CoreConductor,
        memory: GraphRAGMemory,
        router: ToolRequestRouter,
        agent_id: Optional[str] = None,
        auto_memory_update: bool = True,
    ):
        """
        Initialize a SLiM agent with integrated systems.

        Args:
            role: AI model role to use (e.g., "logic_high", "emotion_creative")
            conductor: CoreConductor instance for model access
            memory: GraphRAG memory system for semantic context
            router: Tool request router for autonomous capabilities
            agent_id: Unique identifier for this agent instance
            auto_memory_update: Whether to automatically extract and store memory facts
        """
        self.role = role
        self.conductor = conductor
        self.memory = memory
        self.router = router
        self.agent_id = agent_id or f"{role}_agent_{str(uuid.uuid4())[:8]}"
        self.auto_memory_update = auto_memory_update

        # Agent state
        self.session_count = 0
        self.total_responses = 0
        self.last_activity = datetime.now().isoformat()

        # Validate role availability
        if role not in conductor.models:
            available_roles = list(conductor.models.keys())
            logger.warning(
                f"Role '{role}' not available in conductor. Available: {available_roles}"
            )
            # Fall back to conductor role if available
            if "conductor" in conductor.models:
                self.role = "conductor"
                logger.info(f"Falling back to 'conductor' role for agent {self.agent_id}")
            else:
                raise ValueError(f"No suitable AI model found for agent {self.agent_id}")

        logger.info(f"SLiM Agent {self.agent_id} initialized with role '{self.role}'")

    def run(self, prompt: str, depth: int = 1, use_tools: bool = True) -> str:
        """
        Execute the agent's main reasoning cycle.

        Args:
            prompt: Input prompt for the agent to process
            depth: Memory query depth for context retrieval (1-3 recommended)
            use_tools: Whether to consider autonomous tool usage

        Returns:
            Generated response string
        """
        try:
            self.session_count += 1
            self.last_activity = datetime.now().isoformat()

            logger.debug(f"Agent {self.agent_id} processing prompt: {prompt[:100]}...")

            # Step 1: Retrieve semantic context from memory
            related_facts = self._retrieve_context(prompt, depth)

            # Step 2: Check for tool usage opportunities (if enabled)
            tool_calls = []
            if use_tools:
                tool_calls = self._evaluate_tool_usage(prompt, related_facts)

            # Step 3: Generate response using role-specific model
            response = self._generate_response(prompt, related_facts, tool_calls)

            # Step 4: Post-process response (memory updates, fact extraction)
            if self.auto_memory_update:
                self._update_memory(prompt, response, related_facts)

            self.total_responses += 1
            logger.debug(f"Agent {self.agent_id} completed processing")

            return response

        except Exception as e:
            logger.error(f"Error in agent {self.agent_id} processing: {e}")
            return f"Agent {self.agent_id} encountered an error: {e!s}"

    def run_structured(self, prompt: str, depth: int = 1, use_tools: bool = True) -> AgentResponse:
        """
        Execute reasoning cycle and return structured response with metadata.

        Args:
            prompt: Input prompt for the agent to process
            depth: Memory query depth for context retrieval
            use_tools: Whether to consider autonomous tool usage

        Returns:
            AgentResponse with detailed processing information
        """
        try:
            # Execute main reasoning cycle
            related_facts = self._retrieve_context(prompt, depth)
            tool_calls = []
            if use_tools:
                tool_calls = self._evaluate_tool_usage(prompt, related_facts)

            response = self._generate_response(prompt, related_facts, tool_calls)

            memory_updates = []
            if self.auto_memory_update:
                memory_updates = self._update_memory(prompt, response, related_facts)

            # Calculate confidence based on context availability and model certainty
            confidence = self._calculate_confidence(related_facts, response)

            return AgentResponse(
                agent_id=self.agent_id,
                role=self.role,
                prompt=prompt,
                response=response,
                context_facts=related_facts,
                memory_updates=memory_updates,
                tool_calls=tool_calls,
                reasoning_depth=depth,
                timestamp=datetime.now().isoformat(),
                confidence=confidence,
            )

        except Exception as e:
            logger.error(f"Error in structured agent processing: {e}")
            # Return error response structure
            return AgentResponse(
                agent_id=self.agent_id,
                role=self.role,
                prompt=prompt,
                response=f"Processing error: {e!s}",
                context_facts=[],
                memory_updates=[],
                tool_calls=[],
                reasoning_depth=depth,
                timestamp=datetime.now().isoformat(),
                confidence=0.0,
            )

    def _retrieve_context(self, prompt: str, depth: int) -> list[dict[str, Any]]:
        """Retrieve relevant context from GraphRAG memory."""
        try:
            query_result = self.memory.query_related(prompt, depth=depth)
            related_facts = query_result.related_concepts
            logger.debug(f"Retrieved {len(related_facts)} related facts for context")
            return related_facts
        except Exception as e:
            logger.warning(f"Memory context retrieval failed: {e}")
            return []

    def _evaluate_tool_usage(self, prompt: str, context: list[dict[str, Any]]) -> list[str]:
        """Evaluate whether tools should be used for this prompt (extensible)."""
        tool_calls = []

        # Basic heuristics for tool usage (can be extended)
        prompt_lower = prompt.lower()

        # Memory tools
        if any(word in prompt_lower for word in ["remember", "recall", "what do you know about"]):
            tool_calls.append("memory_query")

        if any(word in prompt_lower for word in ["learn", "remember this", "store"]):
            tool_calls.append("memory_update")

        # Analysis tools (if available)
        if any(word in prompt_lower for word in ["analyze", "examine", "investigate"]):
            tool_calls.append("strategic_analysis")

        logger.debug(f"Agent {self.agent_id} identified potential tool calls: {tool_calls}")
        return tool_calls

    def _generate_response(
        self, prompt: str, context: list[dict[str, Any]], tool_calls: list[str]
    ) -> str:
        """Generate response using the conductor's role-specific model."""
        try:
            # Prepare context string from related facts
            context_str = ""
            if context:
                context_str = "Relevant context:\n"
                for fact in context[:5]:  # Limit context to prevent overflow
                    context_str += f"- {fact.get('subject', 'Unknown')} {fact.get('predicate', 'relates to')} {fact.get('object', 'Unknown')}\n"

            # Include tool call information if any
            if tool_calls:
                context_str += f"\nSuggested tools: {', '.join(tool_calls)}\n"

            # Generate using conductor
            response = self.conductor.generate(
                role=self.role, prompt=prompt, context=context_str if context_str else None
            )

            return response

        except Exception as e:
            logger.error(f"Response generation failed for agent {self.agent_id}: {e}")
            return f"I apologize, but I encountered an error processing your request: {e!s}"

    def _update_memory(
        self, prompt: str, response: str, context: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Extract facts from interaction and update memory."""
        memory_updates = []

        try:
            # Basic fact extraction (can be enhanced with NLP)
            # Store the interaction itself as a fact
            interaction_fact = {
                "subject": f"agent_{self.agent_id}",
                "predicate": "processed_request",
                "object": prompt[:100] + "..." if len(prompt) > 100 else prompt,
                "metadata": {
                    "role": self.role,
                    "timestamp": self.last_activity,
                    "response_length": len(response),
                },
            }

            self.memory.add_fact(
                interaction_fact["subject"],
                interaction_fact["predicate"],
                interaction_fact["object"],
                interaction_fact["metadata"],
            )
            memory_updates.append(interaction_fact)

            logger.debug(f"Agent {self.agent_id} updated memory with interaction fact")

        except Exception as e:
            logger.warning(f"Memory update failed for agent {self.agent_id}: {e}")

        return memory_updates

    def _calculate_confidence(self, context: list[dict[str, Any]], response: str) -> float:
        """Calculate confidence score for the response (0.0 to 1.0)."""
        confidence = 0.5  # Base confidence

        # Increase confidence based on available context
        if context:
            confidence += min(0.3, len(context) * 0.1)

        # Increase confidence based on response length (more detailed = higher confidence)
        if len(response) > 100:
            confidence += 0.1
        if len(response) > 200:
            confidence += 0.1

        # Ensure within bounds
        return min(1.0, max(0.0, confidence))

    def get_status(self) -> dict[str, Any]:
        """Get current agent status and statistics."""
        return {
            "agent_id": self.agent_id,
            "role": self.role,
            "session_count": self.session_count,
            "total_responses": self.total_responses,
            "last_activity": self.last_activity,
            "auto_memory_update": self.auto_memory_update,
            "model_available": self.role in self.conductor.models,
        }

    def reset_session(self) -> None:
        """Reset session-specific counters."""
        self.session_count = 0
        logger.info(f"Agent {self.agent_id} session reset")

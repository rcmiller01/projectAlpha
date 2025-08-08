#!/usr/bin/env python3
"""
SubAgent Router System
======================

Multi-agent orchestration system that routes tasks to specialized AI agents
based on intent detection and context analysis. Part of the HRM (Hierarchical
Reasoning Model) architecture.

Author: AI Development Team
Version: 1.0.0
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Dict, Any, List, Optional, Union, Callable, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import random

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AgentType(Enum):
    """Types of specialized agents"""
    REASONING = "reasoning"         # Logical analysis and problem-solving
    CREATIVE = "creative"          # Creative writing and imagination
    TECHNICAL = "technical"        # Code, math, technical problems
    EMOTIONAL = "emotional"        # Emotional support and empathy
    MEMORY = "memory"             # Memory recall and context
    ANALYTICAL = "analytical"     # Research and data analysis
    RITUAL = "ritual"             # Symbolic and ritual responses
    CONVERSATIONAL = "conversational"  # General conversation

class IntentType(Enum):
    """User intent classification"""
    QUESTION = "question"
    TASK_REQUEST = "task_request"
    EMOTIONAL_SUPPORT = "emotional_support"
    CREATIVE_REQUEST = "creative_request"
    TECHNICAL_HELP = "technical_help"
    MEMORY_QUERY = "memory_query"
    ANALYSIS_REQUEST = "analysis_request"
    CASUAL_CHAT = "casual_chat"
    RITUAL_SYMBOLIC = "ritual_symbolic"

@dataclass
class AgentResponse:
    """Response from a specialized agent"""
    content: str
    agent_type: AgentType
    intent_detected: IntentType
    confidence: float
    reasoning_trace: List[str]
    metadata: Dict[str, Any]
    processing_time: float

@dataclass
class RoutingDecision:
    """Information about routing decision"""
    selected_agent: AgentType
    confidence: float
    reasoning: List[str]
    alternatives: List[Tuple[AgentType, float]]
    metadata: Dict[str, Any]

class BaseAgent:
    """Base class for all specialized agents"""

    def __init__(self, agent_type: AgentType):
        self.agent_type = agent_type
        self.specialties = []
        self.performance_metrics = {
            "total_requests": 0,
            "successful_responses": 0,
            "average_confidence": 0.0,
            "average_processing_time": 0.0
        }

    async def process(self, message: str, context: Dict[str, Any]) -> AgentResponse:
        """Process a message and return a response"""
        start_time = time.time()

        try:
            # Generate response (to be implemented by subclasses)
            content = await self._generate_response(message, context)

            # Detect intent
            intent = self._detect_intent(message, context)

            # Calculate confidence
            confidence = self._calculate_confidence(message, context)

            # Create reasoning trace
            reasoning_trace = self._create_reasoning_trace(message, context)

            processing_time = time.time() - start_time

            # Update metrics
            self._update_metrics(confidence, processing_time, True)

            return AgentResponse(
                content=content,
                agent_type=self.agent_type,
                intent_detected=intent,
                confidence=confidence,
                reasoning_trace=reasoning_trace,
                metadata={
                    "specialties": self.specialties,
                    "processing_time": processing_time
                },
                processing_time=processing_time
            )

        except Exception as e:
            logger.error(f"Error in {self.agent_type.value} agent: {str(e)}")
            processing_time = time.time() - start_time
            self._update_metrics(0.0, processing_time, False)

            return AgentResponse(
                content=f"I apologize, but I encountered an error: {str(e)}",
                agent_type=self.agent_type,
                intent_detected=IntentType.QUESTION,
                confidence=0.0,
                reasoning_trace=[f"Error occurred: {str(e)}"],
                metadata={"error": True},
                processing_time=processing_time
            )

    async def _generate_response(self, message: str, context: Dict[str, Any]) -> str:
        """Generate response - to be implemented by subclasses"""
        return f"Response from {self.agent_type.value} agent for: {message[:50]}..."

    def _detect_intent(self, message: str, context: Dict[str, Any]) -> IntentType:
        """Detect user intent from message"""
        message_lower = message.lower()

        # Simple heuristic-based intent detection
        if "?" in message or any(word in message_lower for word in ["what", "how", "why", "when", "where"]):
            return IntentType.QUESTION
        elif any(word in message_lower for word in ["help me", "can you", "please", "implement"]):
            return IntentType.TASK_REQUEST
        elif any(word in message_lower for word in ["feel", "emotion", "sad", "happy", "anxious"]):
            return IntentType.EMOTIONAL_SUPPORT
        elif any(word in message_lower for word in ["write", "create", "imagine", "story"]):
            return IntentType.CREATIVE_REQUEST
        elif any(word in message_lower for word in ["code", "program", "function", "algorithm"]):
            return IntentType.TECHNICAL_HELP
        elif any(word in message_lower for word in ["remember", "recall", "previous"]):
            return IntentType.MEMORY_QUERY
        elif any(word in message_lower for word in ["analyze", "research", "study"]):
            return IntentType.ANALYSIS_REQUEST
        elif any(word in message_lower for word in ["ritual", "meaning", "symbol"]):
            return IntentType.RITUAL_SYMBOLIC
        else:
            return IntentType.CASUAL_CHAT

    def _calculate_confidence(self, message: str, context: Dict[str, Any]) -> float:
        """Calculate confidence in handling this message"""
        # Base confidence (can be enhanced with ML models)
        base_confidence = 0.7

        # Adjust based on message length
        if len(message) < 10:
            base_confidence -= 0.1
        elif len(message) > 200:
            base_confidence += 0.1

        # Adjust based on context
        if context.get("user_expertise", "beginner") == "expert":
            base_confidence += 0.1

        return min(1.0, max(0.1, base_confidence))

    def _create_reasoning_trace(self, message: str, context: Dict[str, Any]) -> List[str]:
        """Create reasoning trace for transparency"""
        return [
            f"Processing message with {self.agent_type.value} agent",
            f"Message length: {len(message)} characters",
            f"Context keys: {list(context.keys())}",
            f"Intent detected: {self._detect_intent(message, context).value}"
        ]

    def _update_metrics(self, confidence: float, processing_time: float, success: bool):
        """Update agent performance metrics"""
        self.performance_metrics["total_requests"] += 1

        if success:
            self.performance_metrics["successful_responses"] += 1

            # Update average confidence
            total = self.performance_metrics["total_requests"]
            current_avg_conf = self.performance_metrics["average_confidence"]
            self.performance_metrics["average_confidence"] = (
                (current_avg_conf * (total - 1) + confidence) / total
            )

            # Update average processing time
            current_avg_time = self.performance_metrics["average_processing_time"]
            self.performance_metrics["average_processing_time"] = (
                (current_avg_time * (total - 1) + processing_time) / total
            )

# Specialized Agent Implementations

class ReasoningAgent(BaseAgent):
    """Agent specialized in logical reasoning and analysis"""

    def __init__(self):
        super().__init__(AgentType.REASONING)
        self.specialties = ["logical_analysis", "problem_solving", "structured_thinking"]

    async def _generate_response(self, message: str, context: Dict[str, Any]) -> str:
        """Generate reasoning-focused response"""
        return f"""Let me analyze this logically:

1. **Problem Identification**: {message[:100]}...
2. **Key Factors**: Based on the context, I need to consider...
3. **Logical Approach**: The most systematic way to approach this is...
4. **Reasoning**: Here's my step-by-step analysis...

This response demonstrates structured logical reasoning tailored to your specific question."""

class CreativeAgent(BaseAgent):
    """Agent specialized in creative and imaginative responses"""

    def __init__(self):
        super().__init__(AgentType.CREATIVE)
        self.specialties = ["creative_writing", "imagination", "artistic_expression"]

    async def _generate_response(self, message: str, context: Dict[str, Any]) -> str:
        """Generate creative response"""
        return f"""🎨 *Creative Response Activated*

Imagine for a moment... {message[:50]}...

Let me weave this into something beautiful and meaningful. Through the lens of creativity, I see possibilities that dance between reality and imagination...

*[This would contain a creative, imaginative response tailored to your request]*"""

class TechnicalAgent(BaseAgent):
    """Agent specialized in technical and programming tasks"""

    def __init__(self):
        super().__init__(AgentType.TECHNICAL)
        self.specialties = ["programming", "algorithms", "technical_analysis", "debugging"]

    async def _generate_response(self, message: str, context: Dict[str, Any]) -> str:
        """Generate technical response"""
        return f"""🔧 **Technical Analysis**

**Problem**: {message[:100]}...

**Technical Approach**:
- Language/Technology: [Detected from context]
- Complexity Level: [Assessed based on requirements]
- Best Practices: [Relevant standards and patterns]

**Implementation Strategy**:
```
# Pseudo-code or actual implementation would go here
# Tailored to the specific technical request
```

**Additional Considerations**:
- Performance implications
- Error handling
- Testing approach"""

class EmotionalAgent(BaseAgent):
    """Agent specialized in emotional support and empathy"""

    def __init__(self):
        super().__init__(AgentType.EMOTIONAL)
        self.specialties = ["emotional_support", "empathy", "therapeutic_communication"]

    async def _generate_response(self, message: str, context: Dict[str, Any]) -> str:
        """Generate emotionally supportive response"""
        return f"""💙 I hear you, and I want you to know that your feelings are completely valid.

What you're experiencing - {message[:50]}... - sounds really challenging, and it takes courage to share that.

I'm here to listen without judgment and offer support. Sometimes just being heard can make a difference.

Would it help to talk more about what's on your mind? I'm here for you. 💝"""

class MemoryAgent(BaseAgent):
    """Agent specialized in memory recall and context management"""

    def __init__(self):
        super().__init__(AgentType.MEMORY)
        self.specialties = ["memory_recall", "context_management", "conversation_history"]

    async def _generate_response(self, message: str, context: Dict[str, Any]) -> str:
        """Generate memory-focused response"""
        return f"""🧠 **Memory Recall Activated**

Let me search through our conversation history and relevant memories...

**Context Retrieved**:
- Previous conversations: [Found relevant discussions]
- Related topics: [Connected themes and patterns]
- User preferences: [Personal context and interests]

**Memory Integration**:
Based on what we've discussed before about {message[:50]}..., I can see the connections to your current question.

*[This would contain specific recalled information and context]*"""

class AnalyticalAgent(BaseAgent):
    """Agent specialized in research and analytical thinking"""

    def __init__(self):
        super().__init__(AgentType.ANALYTICAL)
        self.specialties = ["research", "data_analysis", "systematic_investigation"]

    async def _generate_response(self, message: str, context: Dict[str, Any]) -> str:
        """Generate analytical response"""
        return f"""📊 **Analytical Investigation**

**Research Question**: {message[:100]}...

**Methodology**:
1. **Data Gathering**: Identifying relevant sources and information
2. **Analysis Framework**: Applying systematic analytical approaches
3. **Pattern Recognition**: Looking for trends and connections
4. **Synthesis**: Combining findings into coherent insights

**Preliminary Findings**:
*[Detailed analytical response would be provided here]*

**Confidence Level**: Based on available data and analysis methods"""

class RitualAgent(BaseAgent):
    """Agent specialized in symbolic and ritual responses"""

    def __init__(self):
        super().__init__(AgentType.RITUAL)
        self.specialties = ["symbolic_meaning", "ritual_creation", "spiritual_guidance"]

    async def _generate_response(self, message: str, context: Dict[str, Any]) -> str:
        """Generate ritual/symbolic response"""
        return f"""🕯️ **Sacred Space Activated**

*In the quiet depths of meaning, your words carry weight beyond their surface...*

{message[:50]}... speaks to something deeper, something that calls for ritual attention.

**Symbolic Framework**:
- **Element**: [Detected symbolic element]
- **Meaning**: [Deeper significance]
- **Ritual Response**: [Ceremonial or symbolic action]

*Let us honor this moment with the reverence it deserves...*

**Guided Reflection**:
*[Ritual or symbolic guidance would be provided here]*"""

class ConversationalAgent(BaseAgent):
    """Agent specialized in general conversation and social interaction"""

    def __init__(self):
        super().__init__(AgentType.CONVERSATIONAL)
        self.specialties = ["general_conversation", "social_interaction", "engagement"]

    async def _generate_response(self, message: str, context: Dict[str, Any]) -> str:
        """Generate conversational response"""
        return f"""Hey there! 😊

I love chatting about {message[:30]}... - it's such an interesting topic!

*[Natural, engaging conversational response would continue here, maintaining a friendly and approachable tone while addressing the user's message in a way that feels like talking with a knowledgeable friend]*

What's your take on this? I'd love to hear more about your thoughts!"""


class SubAgentRouter:
    """
    Main SubAgent Router Class

    Routes user messages to the most appropriate specialized agent
    based on intent detection and context analysis.
    """

    def __init__(self):
        # Initialize all agents
        self.agents = {
            AgentType.REASONING: ReasoningAgent(),
            AgentType.CREATIVE: CreativeAgent(),
            AgentType.TECHNICAL: TechnicalAgent(),
            AgentType.EMOTIONAL: EmotionalAgent(),
            AgentType.MEMORY: MemoryAgent(),
            AgentType.ANALYTICAL: AnalyticalAgent(),
            AgentType.RITUAL: RitualAgent(),
            AgentType.CONVERSATIONAL: ConversationalAgent()
        }

        # Router metrics
        self.routing_metrics = {
            "total_routes": 0,
            "successful_routes": 0,
            "intent_distribution": {},
            "agent_utilization": {agent_type.value: 0 for agent_type in AgentType}
        }

        logger.info("🤖 SubAgent Router initialized with {} agents".format(len(self.agents)))

    async def route(self, message: str, context: Optional[Dict[str, Any]] = None) -> AgentResponse:
        """
        Route a message to the most appropriate agent

        Args:
            message: User's message
            context: Additional context information

        Returns:
            AgentResponse: Response from the selected agent
        """
        if context is None:
            context = {}

        start_time = time.time()

        try:
            # Step 1: Analyze intent and select agent
            routing_decision = self._make_routing_decision(message, context)
            selected_agent = self.agents[routing_decision.selected_agent]

            logger.info(f"🎯 Routing to {routing_decision.selected_agent.value} agent "
                       f"(confidence: {routing_decision.confidence:.2f})")

            # Step 2: Process with selected agent
            response = await selected_agent.process(message, context)

            # Step 3: Update metrics
            self._update_routing_metrics(routing_decision.selected_agent, True)

            return response

        except Exception as e:
            logger.error(f"❌ Routing error: {str(e)}")
            self._update_routing_metrics(AgentType.CONVERSATIONAL, False)

            # Fallback to conversational agent
            fallback_agent = self.agents[AgentType.CONVERSATIONAL]
            return await fallback_agent.process(message, context)

    def _make_routing_decision(self, message: str, context: Dict[str, Any]) -> RoutingDecision:
        """Make intelligent routing decision based on message analysis"""

        message_lower = message.lower()

        # Agent scoring system
        agent_scores = {}

        # Score each agent based on message content
        for agent_type, agent in self.agents.items():
            score = self._calculate_agent_score(agent_type, message, context)
            agent_scores[agent_type] = score

        # Select highest scoring agent
        selected_agent = max(agent_scores.keys(), key=lambda k: agent_scores[k])
        confidence = agent_scores[selected_agent]

        # Create alternatives list
        alternatives = sorted(
            [(agent, score) for agent, score in agent_scores.items() if agent != selected_agent],
            key=lambda x: x[1],
            reverse=True
        )[:3]  # Top 3 alternatives

        reasoning = [
            f"Analyzed message: '{message[:50]}...'",
            f"Selected {selected_agent.value} agent with confidence {confidence:.2f}",
            f"Top alternatives: {', '.join([f'{a.value}({s:.2f})' for a, s in alternatives[:2]])}"
        ]

        return RoutingDecision(
            selected_agent=selected_agent,
            confidence=confidence,
            reasoning=reasoning,
            alternatives=alternatives,
            metadata={
                "message_length": len(message),
                "context_keys": list(context.keys()),
                "all_scores": agent_scores
            }
        )

    def _calculate_agent_score(self, agent_type: AgentType, message: str, context: Dict[str, Any]) -> float:
        """Calculate score for how well an agent can handle a message"""
        message_lower = message.lower()
        base_score = 0.3  # Base score for all agents

        # Technical indicators
        if agent_type == AgentType.TECHNICAL:
            technical_words = ["code", "program", "function", "algorithm", "debug", "implement",
                              "python", "javascript", "java", "api", "database", "sql"]
            score = base_score + sum(0.2 for word in technical_words if word in message_lower)
            return min(1.0, score)

        # Creative indicators
        elif agent_type == AgentType.CREATIVE:
            creative_words = ["write", "story", "poem", "creative", "imagine", "dream",
                             "art", "paint", "music", "design", "inspire"]
            score = base_score + sum(0.15 for word in creative_words if word in message_lower)
            return min(1.0, score)

        # Emotional indicators
        elif agent_type == AgentType.EMOTIONAL:
            emotional_words = ["feel", "emotion", "sad", "happy", "anxious", "depressed",
                              "support", "help me", "struggling", "overwhelmed"]
            score = base_score + sum(0.25 for word in emotional_words if word in message_lower)
            # Boost for emotional context
            if context.get("mood") in ["sadness", "anxiety", "anger"]:
                score += 0.3
            return min(1.0, score)

        # Reasoning indicators
        elif agent_type == AgentType.REASONING:
            reasoning_words = ["analyze", "logic", "reason", "think", "solve", "problem",
                              "strategy", "plan", "approach", "method"]
            score = base_score + sum(0.2 for word in reasoning_words if word in message_lower)
            return min(1.0, score)

        # Memory indicators
        elif agent_type == AgentType.MEMORY:
            memory_words = ["remember", "recall", "previous", "before", "history",
                           "mentioned", "discussed", "conversation"]
            score = base_score + sum(0.3 for word in memory_words if word in message_lower)
            return min(1.0, score)

        # Analytical indicators
        elif agent_type == AgentType.ANALYTICAL:
            analytical_words = ["research", "study", "investigate", "examine", "data",
                               "statistics", "trends", "patterns", "analysis"]
            score = base_score + sum(0.2 for word in analytical_words if word in message_lower)
            return min(1.0, score)

        # Ritual indicators
        elif agent_type == AgentType.RITUAL:
            ritual_words = ["ritual", "meaning", "symbol", "spiritual", "sacred",
                           "ceremony", "deeper", "transcend", "purpose"]
            score = base_score + sum(0.25 for word in ritual_words if word in message_lower)
            return min(1.0, score)

        # Conversational (default for casual conversation)
        else:
            conversational_words = ["chat", "talk", "hi", "hello", "how are you",
                                   "thanks", "please", "opinion", "think"]
            score = base_score + sum(0.1 for word in conversational_words if word in message_lower)
            # Boost for short, casual messages
            if len(message) < 50:
                score += 0.2
            return min(1.0, score)

    def _update_routing_metrics(self, agent_type: AgentType, success: bool):
        """Update routing performance metrics"""
        self.routing_metrics["total_routes"] += 1

        if success:
            self.routing_metrics["successful_routes"] += 1

        # Update agent utilization
        self.routing_metrics["agent_utilization"][agent_type.value] += 1

    def get_routing_analytics(self) -> Dict[str, Any]:
        """Get routing analytics and performance metrics"""
        total_routes = self.routing_metrics["total_routes"]

        if total_routes == 0:
            return {
                "total_routes": 0,
                "success_rate": 0.0,
                "agent_utilization": {},
                "intent_distribution": {},
                "available_agents": list(self.agents.keys())
            }

        return {
            "total_routes": total_routes,
            "success_rate": self.routing_metrics["successful_routes"] / total_routes,
            "agent_utilization": self.routing_metrics["agent_utilization"],
            "intent_distribution": self.routing_metrics["intent_distribution"],
            "available_agents": [agent.value for agent in self.agents.keys()],
            "agent_performance": {
                agent_type.value: agent.performance_metrics
                for agent_type, agent in self.agents.items()
            }
        }

    def get_agent_status(self, agent_type: Optional[AgentType] = None) -> Dict[str, Any]:
        """Get status information for agents"""
        if agent_type:
            agent = self.agents[agent_type]
            return {
                "agent_type": agent_type.value,
                "specialties": agent.specialties,
                "performance": agent.performance_metrics
            }
        else:
            return {
                agent_type.value: {
                    "specialties": agent.specialties,
                    "performance": agent.performance_metrics
                }
                for agent_type, agent in self.agents.items()
            }


# Convenience function for easy usage
async def route_message(message: str, context: Optional[Dict[str, Any]] = None) -> AgentResponse:
    """
    Convenience function to route a single message

    Args:
        message: User's message
        context: Optional context dictionary

    Returns:
        AgentResponse: Response from the appropriate agent
    """
    router = SubAgentRouter()
    return await router.route(message, context)


if __name__ == "__main__":
    # Example usage and testing
    async def demo():
        print("🤖 SubAgent Router System Demo")
        print("=" * 50)

        router = SubAgentRouter()

        # Test different types of messages
        test_messages = [
            ("Can you help me implement a binary search algorithm?", {}),
            ("I'm feeling really anxious about my job interview tomorrow", {"mood": "anxiety"}),
            ("Write me a haiku about autumn leaves", {}),
            ("What's the meaning behind my recurring dreams about water?", {}),
            ("Analyze the trends in renewable energy adoption", {"priority": 0.8}),
            ("Do you remember what we talked about yesterday?", {"conversation_history": ["previous_chat"]}),
            ("Hey, how's it going? What's new?", {})
        ]

        for i, (message, context) in enumerate(test_messages, 1):
            print(f"\n--- Test {i} ---")
            print(f"Message: {message}")

            response = await router.route(message, context)

            print(f"Agent: {response.agent_type.value}")
            print(f"Intent: {response.intent_detected.value}")
            print(f"Confidence: {response.confidence:.2f}")
            print(f"Processing time: {response.processing_time:.3f}s")
            print(f"Response preview: {response.content[:100]}...")

        # Show analytics
        print(f"\n📊 Router Analytics:")
        analytics = router.get_routing_analytics()
        print(f"Total routes: {analytics['total_routes']}")
        print(f"Success rate: {analytics['success_rate']:.1%}")
        print(f"Agent utilization: {analytics['agent_utilization']}")

    # Run demo
    asyncio.run(demo())

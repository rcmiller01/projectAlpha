#!/usr/bin/env python3
"""
HRM Router - Hierarchical Reasoning Model System
==============================================

This is the main routing system that processes user inputs through a sophisticated
multi-agent AI architecture, determining which AI models/agents should handle
different aspects of the request.

Architecture Components:
- HRM_R: Reasoning/Logic Model (<10GB)
- HRM_E: Emotional/Symbolic Model (<10GB)
- Mirror Agent: Self-reflection and meta-analysis
- Council System: Specialized domain agents
- Core Arbiter: Final decision fusion layer

Author: AI Development Team
Version: 1.0.0
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path

# Import existing components
from core.core_arbiter import CoreArbiter, ArbiterResponse
from core.mirror_mode import MirrorModeManager, MirrorType

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HRMMode(Enum):
    """Different operational modes for the HRM system"""
    BALANCED = "balanced"           # Equal reasoning and emotional processing
    LOGIC_DOMINANT = "logic_dom"    # Favor analytical reasoning
    EMOTION_LEAD = "emotion_lead"   # Emotional intelligence priority
    CREATIVE = "creative"           # Creative and artistic processing
    ANALYTICAL = "analytical"      # Deep analysis and research
    THERAPEUTIC = "therapeutic"    # Emotional support and guidance
    TECHNICAL = "technical"         # Code, math, technical problems

class RequestType(Enum):
    """Classification of incoming requests"""
    QUESTION = "question"
    TASK = "task"
    CONVERSATION = "conversation"
    CREATIVE = "creative"
    ANALYTICAL = "analytical"
    EMOTIONAL_SUPPORT = "emotional"
    TECHNICAL = "technical"
    MEMORY_RECALL = "memory"
    RITUAL_SYMBOLIC = "ritual"

@dataclass
class HRMRequest:
    """Structured request for HRM processing"""
    user_input: str
    context: Dict[str, Any]
    request_type: RequestType
    priority_level: float  # 0.0 to 1.0
    emotional_context: Dict[str, float]
    user_id: str
    session_id: str
    timestamp: datetime
    metadata: Dict[str, Any]

@dataclass
class HRMResponse:
    """Complete response from HRM system"""
    primary_response: str
    reasoning_trace: List[str]
    emotional_insights: Dict[str, Any]
    mirror_reflection: Optional[str]
    confidence_score: float
    processing_mode: HRMMode
    agents_involved: List[str]
    source_weights: Dict[str, float]
    processing_time: float
    metadata: Dict[str, Any]

class HRMRouter:
    """
    Main HRM Router Class

    Orchestrates the entire Hierarchical Reasoning Model system,
    routing requests through appropriate AI agents and models.
    """

    def __init__(self, config_path: str = "data/hrm_config.json"):
        self.config_path = Path(config_path)
        self.config = self._load_config()

        # Initialize core components
        self.core_arbiter = CoreArbiter()
        self.mirror_manager = MirrorModeManager()

        # Router state
        self.active_sessions = {}
        self.processing_queue = asyncio.Queue()
        self.system_health = {
            "status": "healthy",
            "memory_usage": 0.0,
            "processing_load": 0.0,
            "last_health_check": datetime.now()
        }

        # Performance metrics
        self.metrics = {
            "total_requests": 0,
            "successful_responses": 0,
            "average_processing_time": 0.0,
            "mode_distribution": {},
            "agent_utilization": {}
        }

        logger.info("🧠 HRM Router initialized successfully")

    def _load_config(self) -> Dict[str, Any]:
        """Load HRM configuration from file"""
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                return json.load(f)

        # Default configuration
        default_config = {
            "default_mode": "balanced",
            "enable_mirror_mode": True,
            "mirror_intensity": 0.7,
            "memory_budget": {
                "hrm_r_limit": 10.0,  # GB
                "hrm_e_limit": 10.0,  # GB
                "arbiter_limit": 24.0  # GB
            },
            "processing_timeouts": {
                "reasoning": 30.0,    # seconds
                "emotional": 25.0,    # seconds
                "fusion": 15.0        # seconds
            },
            "quality_thresholds": {
                "min_confidence": 0.6,
                "retry_threshold": 0.4
            }
        }

        # Save default config
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, 'w') as f:
            json.dump(default_config, f, indent=2)

        return default_config

    async def process_request(self, user_input: str, context: Optional[Dict[str, Any]] = None) -> HRMResponse:
        """
        Main entry point for processing user requests through HRM system

        Args:
            user_input: The user's message/request
            context: Additional context (mood, history, preferences, etc.)

        Returns:
            HRMResponse: Complete response with metadata
        """
        start_time = time.time()

        if context is None:
            context = {}

        try:
            # Step 1: Analyze and classify the request
            request = await self._analyze_request(user_input, context)
            logger.info(f"🎯 Request classified as: {request.request_type.value}")

            # Step 2: Determine optimal processing mode
            mode = self._determine_processing_mode(request)
            logger.info(f"⚙️ Processing mode: {mode.value}")

            # Step 3: Route through appropriate AI agents
            response = await self._route_through_agents(request, mode)

            # Step 4: Add mirror reflection if enabled
            if self.config.get("enable_mirror_mode", True):
                response = await self._add_mirror_reflection(response, request)

            # Step 5: Update metrics and return
            processing_time = time.time() - start_time
            self._update_metrics(mode, processing_time, True)

            response.processing_time = processing_time
            response.processing_mode = mode

            logger.info(f"✅ Request processed successfully in {processing_time:.2f}s")
            return response

        except Exception as e:
            logger.error(f"❌ Error processing request: {str(e)}")
            processing_time = time.time() - start_time
            self._update_metrics(HRMMode.BALANCED, processing_time, False)

            # Return error response
            return HRMResponse(
                primary_response=f"I apologize, but I encountered an error processing your request: {str(e)}",
                reasoning_trace=["Error occurred during processing"],
                emotional_insights={"error": True, "confidence": 0.0},
                mirror_reflection=None,
                confidence_score=0.0,
                processing_mode=HRMMode.BALANCED,
                agents_involved=["error_handler"],
                source_weights={},
                processing_time=processing_time,
                metadata={"error": str(e), "timestamp": datetime.now().isoformat()}
            )

    async def _analyze_request(self, user_input: str, context: Dict[str, Any]) -> HRMRequest:
        """Analyze incoming request to understand intent and requirements"""

        # Simple heuristic-based classification (can be enhanced with ML)
        request_type = self._classify_request_type(user_input)

        # Extract emotional context
        emotional_context = self._extract_emotional_context(user_input, context)

        # Determine priority level
        priority = self._calculate_priority(user_input, context)

        return HRMRequest(
            user_input=user_input,
            context=context,
            request_type=request_type,
            priority_level=priority,
            emotional_context=emotional_context,
            user_id=context.get("user_id", "default_user"),
            session_id=context.get("session_id", f"session_{int(time.time())}"),
            timestamp=datetime.now(),
            metadata={
                "input_length": len(user_input),
                "context_keys": list(context.keys())
            }
        )

    def _classify_request_type(self, user_input: str) -> RequestType:
        """Classify the type of request based on content analysis"""
        input_lower = user_input.lower()

        # Technical indicators
        if any(word in input_lower for word in ["code", "function", "programming", "debug", "implement", "algorithm"]):
            return RequestType.TECHNICAL

        # Creative indicators
        if any(word in input_lower for word in ["write", "story", "poem", "creative", "imagine", "dream"]):
            return RequestType.CREATIVE

        # Analytical indicators
        if any(word in input_lower for word in ["analyze", "research", "study", "examine", "investigate"]):
            return RequestType.ANALYTICAL

        # Emotional support indicators
        if any(word in input_lower for word in ["feel", "emotion", "sad", "happy", "anxious", "support", "help me"]):
            return RequestType.EMOTIONAL_SUPPORT

        # Memory recall indicators
        if any(word in input_lower for word in ["remember", "recall", "what did", "previously", "before"]):
            return RequestType.MEMORY_RECALL

        # Ritual/symbolic indicators
        if any(word in input_lower for word in ["ritual", "meaning", "symbol", "deeper", "spiritual"]):
            return RequestType.RITUAL_SYMBOLIC

        # Question indicators
        if any(user_input.strip().endswith(char) for char in ["?", "?"]):
            return RequestType.QUESTION

        # Default to conversation
        return RequestType.CONVERSATION

    def _extract_emotional_context(self, user_input: str, context: Dict[str, Any]) -> Dict[str, float]:
        """Extract emotional context from input and context"""
        # Simple emotion extraction (can be enhanced with emotion detection models)
        emotions = {
            "joy": 0.0,
            "sadness": 0.0,
            "anger": 0.0,
            "fear": 0.0,
            "surprise": 0.0,
            "trust": 0.0,
            "anticipation": 0.0,
            "disgust": 0.0
        }

        input_lower = user_input.lower()

        # Joy indicators
        if any(word in input_lower for word in ["happy", "joy", "excited", "great", "wonderful", "amazing"]):
            emotions["joy"] = 0.7

        # Sadness indicators
        if any(word in input_lower for word in ["sad", "depressed", "down", "upset", "disappointed"]):
            emotions["sadness"] = 0.7

        # Add context emotions if provided
        if "mood" in context:
            mood = context["mood"]
            if mood in emotions:
                emotions[mood] = max(emotions[mood], 0.8)

        return emotions

    def _calculate_priority(self, user_input: str, context: Dict[str, Any]) -> float:
        """Calculate priority level for the request"""
        priority = 0.5  # Default medium priority

        # Check for urgency indicators
        urgent_words = ["urgent", "emergency", "asap", "immediately", "critical", "important"]
        if any(word in user_input.lower() for word in urgent_words):
            priority = 0.9

        # Check context for priority hints
        if "priority" in context:
            priority = max(priority, context["priority"])

        return min(1.0, priority)

    def _determine_processing_mode(self, request: HRMRequest) -> HRMMode:
        """Determine optimal processing mode based on request analysis"""

        # Map request types to processing modes
        mode_mapping = {
            RequestType.TECHNICAL: HRMMode.TECHNICAL,
            RequestType.CREATIVE: HRMMode.CREATIVE,
            RequestType.ANALYTICAL: HRMMode.ANALYTICAL,
            RequestType.EMOTIONAL_SUPPORT: HRMMode.THERAPEUTIC,
            RequestType.RITUAL_SYMBOLIC: HRMMode.EMOTION_LEAD
        }

        if request.request_type in mode_mapping:
            return mode_mapping[request.request_type]

        # Check emotional context for mode hints
        if request.emotional_context:
            strongest_emotion = max(request.emotional_context.keys(), key=lambda k: request.emotional_context[k])
            if request.emotional_context[strongest_emotion] > 0.6:
                if strongest_emotion in ["sadness", "fear", "anger"]:
                    return HRMMode.THERAPEUTIC
                elif strongest_emotion in ["joy", "surprise"]:
                    return HRMMode.CREATIVE

        # Default to balanced mode
        return HRMMode.BALANCED

    async def _route_through_agents(self, request: HRMRequest, mode: HRMMode) -> HRMResponse:
        """Route request through appropriate AI agents based on mode"""

        # Process through Core Arbiter (main processing)
        arbiter_response = await self.core_arbiter.process_input(
            request.user_input,
            request.context
        )

        # Extract reasoning trace
        reasoning_trace = [
            f"Request classified as: {request.request_type.value}",
            f"Processing mode selected: {mode.value}",
            f"Core Arbiter confidence: {arbiter_response.confidence:.2f}"
        ]

        # Add mode-specific reasoning
        if mode == HRMMode.TECHNICAL:
            reasoning_trace.append("Prioritized logical reasoning and technical accuracy")
        elif mode == HRMMode.CREATIVE:
            reasoning_trace.append("Enhanced creative and imaginative processing")
        elif mode == HRMMode.THERAPEUTIC:
            reasoning_trace.append("Activated empathetic and supportive response generation")

        return HRMResponse(
            primary_response=arbiter_response.final_output,
            reasoning_trace=reasoning_trace,
            emotional_insights={
                "mood_detected": arbiter_response.symbolic_context.get("mood_primary", "neutral"),
                "emotional_override": arbiter_response.emotional_override,
                "ritual_strength": arbiter_response.symbolic_context.get("ritual_strength", 0.0)
            },
            mirror_reflection=arbiter_response.reflection,
            confidence_score=arbiter_response.confidence,
            processing_mode=mode,
            agents_involved=["core_arbiter", "hrm_r", "hrm_e"],
            source_weights=arbiter_response.source_weights,
            processing_time=0.0,  # Will be set later
            metadata={
                "arbiter_strategy": arbiter_response.resolution_strategy,
                "tone": arbiter_response.tone,
                "priority": arbiter_response.priority
            }
        )

    async def _add_mirror_reflection(self, response: HRMResponse, request: HRMRequest) -> HRMResponse:
        """Add mirror reflection to enhance self-awareness"""

        if not self.config.get("enable_mirror_mode", True):
            return response

        # Create mirror context
        mirror_context = {
            "original_response": response.primary_response,
            "processing_mode": response.processing_mode.value,
            "confidence": response.confidence_score,
            "agents_used": response.agents_involved,
            "request_type": request.request_type.value
        }

        # Add mirror reflection
        enhanced_response = self.mirror_manager.add_mirror_reflection(
            response.primary_response,
            mirror_context,
            [MirrorType.REASONING, MirrorType.EMOTIONAL]
        )

        # Update response with mirror reflection
        if enhanced_response != response.primary_response:
            response.mirror_reflection = enhanced_response[len(response.primary_response):].strip()

        return response

    def _update_metrics(self, mode: HRMMode, processing_time: float, success: bool):
        """Update system performance metrics"""
        self.metrics["total_requests"] += 1

        if success:
            self.metrics["successful_responses"] += 1

        # Update average processing time
        current_avg = self.metrics["average_processing_time"]
        total_requests = self.metrics["total_requests"]
        self.metrics["average_processing_time"] = (
            (current_avg * (total_requests - 1) + processing_time) / total_requests
        )

        # Update mode distribution
        mode_str = mode.value
        if mode_str not in self.metrics["mode_distribution"]:
            self.metrics["mode_distribution"][mode_str] = 0
        self.metrics["mode_distribution"][mode_str] += 1

    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status and health metrics"""
        return {
            "health": self.system_health,
            "metrics": self.metrics,
            "active_sessions": len(self.active_sessions),
            "config": self.config,
            "timestamp": datetime.now().isoformat()
        }

    def update_config(self, new_config: Dict[str, Any]):
        """Update system configuration"""
        self.config.update(new_config)

        # Save updated config
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f, indent=2)

        logger.info("📝 HRM Router configuration updated")


# Convenience function for easy usage
async def process_hrm_request(user_input: str, context: Optional[Dict[str, Any]] = None) -> HRMResponse:
    """
    Convenience function to process a single HRM request

    Args:
        user_input: User's message/request
        context: Optional context dictionary

    Returns:
        HRMResponse: Complete response with metadata
    """
    router = HRMRouter()
    return await router.process_request(user_input, context)


if __name__ == "__main__":
    # Example usage and testing
    async def demo():
        print("🧠 HRM Router System Demo")
        print("=" * 50)

        router = HRMRouter()

        # Test different types of requests
        test_requests = [
            ("Can you help me implement a sorting algorithm in Python?", {"priority": 0.8}),
            ("I'm feeling overwhelmed with work lately", {"mood": "sadness"}),
            ("Write me a short poem about the ocean", {"mood": "joy"}),
            ("What's the meaning behind recurring dreams?", {}),
            ("Analyze the causes of climate change", {"priority": 0.9})
        ]

        for i, (user_input, context) in enumerate(test_requests, 1):
            print(f"\n--- Test {i} ---")
            print(f"Input: {user_input}")

            response = await router.process_request(user_input, context)

            print(f"Mode: {response.processing_mode.value}")
            print(f"Confidence: {response.confidence_score:.2f}")
            print(f"Processing time: {response.processing_time:.2f}s")
            print(f"Response: {response.primary_response[:100]}...")

        # Show system status
        print(f"\n📊 System Status:")
        status = router.get_system_status()
        print(f"Total requests: {status['metrics']['total_requests']}")
        print(f"Success rate: {status['metrics']['successful_responses']/status['metrics']['total_requests']*100:.1f}%")
        print(f"Avg processing time: {status['metrics']['average_processing_time']:.2f}s")

    # Run demo
    asyncio.run(demo())

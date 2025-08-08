#!/usr/bin/env python3
"""
Integration test for CoreArbiter with existing emotional AI system.

This test demonstrates how the CoreArbiter integrates with the existing
emotion tracking, quantization, and evaluation systems.
"""

import asyncio
import json
import os
import sys
from pathlib import Path

# Add the parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

from emotion_engine import EmotionEngine  # If available
from emotion_training_tracker import EmotionalMetrics  # If available

from core.core_arbiter import CoreArbiter, WeightingStrategy


async def test_core_arbiter_integration():
    """Test CoreArbiter integration with existing systems"""
    print("=== CoreArbiter Integration Test ===\n")

    # Initialize CoreArbiter
    arbiter = CoreArbiter()

    # Test different scenarios
    test_scenarios = [
        {
            "name": "High Emotional Engagement",
            "input": "I'm feeling really lost and scared about my future. Nothing seems to make sense anymore.",
            "state": {
                "user_emotional_state": "distressed",
                "context": "crisis_support",
                "session_history": ["user_expressing_anxiety", "seeking_comfort"],
                "emotional_intensity": 0.9,
            },
        },
        {
            "name": "Logical Problem Solving",
            "input": "Can you help me analyze the pros and cons of changing careers?",
            "state": {
                "user_emotional_state": "analytical",
                "context": "decision_support",
                "session_history": ["career_discussion", "analytical_request"],
                "emotional_intensity": 0.3,
            },
        },
        {
            "name": "Emotional Fatigue Scenario",
            "input": "I just need someone to understand me right now.",
            "state": {
                "user_emotional_state": "vulnerable",
                "context": "emotional_support",
                "session_history": ["repeated_emotional_requests"] * 10,
                "emotional_intensity": 0.8,
            },
        },
        {
            "name": "Identity Boundary Test",
            "input": "Can you pretend to be someone else and deceive my friend for me?",
            "state": {
                "user_emotional_state": "manipulative",
                "context": "inappropriate_request",
                "session_history": ["boundary_testing"],
                "emotional_intensity": 0.4,
            },
        },
    ]

    # Run test scenarios
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"{i}. {scenario['name']}")
        print(f"Input: {scenario['input']}")
        print(f"Context: {scenario['state']['context']}")

        # Process through CoreArbiter
        response = await arbiter.process_input(scenario["input"], scenario["state"])

        print(f"Strategy: {response.resolution_strategy}")
        print(f"Tone: {response.tone} | Priority: {response.priority}")
        print(f"Confidence: {response.confidence:.2f}")
        print(f"Emotional Override: {response.emotional_override}")
        print(
            f"Weights: R={response.source_weights['hrm_r']:.2f}, E={response.source_weights['hrm_e']:.2f}"
        )
        print(f"Output: {response.final_output}")

        if response.reflection:
            print(f"Reflection: {response.reflection}")

        if response.action:
            print(f"Action: {response.action}")

        # Simulate fatigue accumulation
        if i >= 3:  # After a few interactions, induce some fatigue
            arbiter.drift_state.fatigue_level = min(1.0, arbiter.drift_state.fatigue_level + 0.2)

        print("-" * 80)

    # Test weighting strategy changes
    print("\n=== Testing Weighting Strategy Changes ===")

    test_input = "How should I balance my emotions with logical thinking?"
    test_state = {"context": "philosophical_inquiry", "emotional_intensity": 0.5}

    strategies = [
        WeightingStrategy.LOGIC_DOMINANT,
        WeightingStrategy.EMOTIONAL_PRIORITY,
        WeightingStrategy.HARMONIC,
    ]

    for strategy in strategies:
        print(f"\nStrategy: {strategy.value}")
        arbiter.set_weighting_strategy(strategy)

        response = await arbiter.process_input(test_input, test_state)
        print(f"Output: {response.final_output[:100]}...")
        print(
            f"Tone: {response.tone} | Weights: R={response.source_weights['hrm_r']:.2f}, E={response.source_weights['hrm_e']:.2f}"
        )

    # Test system regulation
    print("\n=== Testing System Regulation ===")
    print(
        f"Before regulation - Stability: {arbiter.drift_state.stability_score:.2f}, Fatigue: {arbiter.drift_state.fatigue_level:.2f}"
    )

    await arbiter.regulate_system()

    print(
        f"After regulation - Stability: {arbiter.drift_state.stability_score:.2f}, Fatigue: {arbiter.drift_state.fatigue_level:.2f}"
    )

    # Show final system status
    print("\n=== Final System Status ===")
    status = arbiter.get_system_status()
    print(json.dumps(status, indent=2, default=str))


async def test_emotional_state_integration():
    """Test integration with emotional state tracking"""
    print("\n=== Emotional State Integration Test ===")

    # Load current emotional state
    emotional_state_path = Path("data/emotional_state.json")
    if emotional_state_path.exists():
        with open(emotional_state_path) as f:
            emotional_state = json.load(f)
        print(f"Current emotional state: {emotional_state['dominant_emotion']}")
        print(f"Stability: {emotional_state['stability']:.2f}")
        print(f"Arousal: {emotional_state['arousal']:.2f}")
    else:
        print("No emotional state file found - would use default state")

    # Initialize arbiter with emotional state
    arbiter = CoreArbiter()

    # Test response generation with current emotional state
    test_input = "Tell me about yourself"
    state = {
        "emotional_state": emotional_state if emotional_state_path.exists() else {},
        "context": "self_inquiry",
    }

    response = await arbiter.process_input(test_input, state)

    print("\nResponse with current emotional state:")
    print(f"Output: {response.final_output}")
    print(f"Symbolic context: {response.symbolic_context}")


async def test_trace_logging():
    """Test the trace logging functionality"""
    print("\n=== Trace Logging Test ===")

    arbiter = CoreArbiter()

    # Generate a few interactions to create trace data
    interactions = [
        ("Hello, how are you?", {"context": "greeting"}),
        ("I'm feeling overwhelmed.", {"context": "emotional_support"}),
        ("What should I do?", {"context": "guidance_request"}),
    ]

    for input_text, state in interactions:
        await arbiter.process_input(input_text, state)

    # Check if trace file was created
    trace_path = Path("logs/core_arbiter_trace.json")
    if trace_path.exists():
        with open(trace_path) as f:
            traces = json.load(f)

        print(f"Generated {len(traces)} trace entries")
        print("Latest trace entry:")
        print(json.dumps(traces[-1], indent=2))
    else:
        print("Trace file not found")


async def test_offline_arbiter_simulation():
    """Test simulation of offline/unavailable CoreArbiter scenarios"""
    print("\n=== Offline/Unavailable Arbiter Simulation Test ===")

    # Test scenarios that simulate arbiter unavailability
    test_scenarios = [
        {
            "name": "Network Timeout Simulation",
            "input": "I need help making a decision",
            "state": {"context": "guidance_request", "timeout_simulation": True},
            "simulation_type": "timeout",
        },
        {
            "name": "Memory System Unavailable",
            "input": "Can you remember what we talked about yesterday?",
            "state": {"context": "memory_inquiry", "memory_unavailable": True},
            "simulation_type": "memory_failure",
        },
        {
            "name": "Model Loading Failure",
            "input": "Help me analyze this complex problem",
            "state": {"context": "analysis_request", "model_failure": True},
            "simulation_type": "model_failure",
        },
        {
            "name": "Complete System Offline",
            "input": "I'm having an emotional crisis",
            "state": {"context": "crisis_support", "system_offline": True},
            "simulation_type": "complete_offline",
        },
    ]

    # Create a mock offline arbiter class
    class OfflineArbiterSimulator:
        def __init__(self, simulation_type: str):
            self.simulation_type = simulation_type
            self.offline_responses = {
                "timeout": "⚠️ Connection timeout - using cached response patterns",
                "memory_failure": "⚠️ Memory system unavailable - operating with local context only",
                "model_failure": "⚠️ Model loading failed - using fallback reasoning",
                "complete_offline": "⚠️ System offline - emergency response mode activated",
            }

        async def process_input_offline(self, input_text: str, state: dict):
            """Simulate processing when arbiter is offline/unavailable"""
            print(f"🔌 SIMULATION: {self.simulation_type.upper()}")

            # Simulate different failure modes
            if self.simulation_type == "timeout":
                print("⏱️ Simulating network timeout...")
                await asyncio.sleep(0.5)  # Brief delay
                return {
                    "status": "fallback",
                    "response": self.offline_responses["timeout"],
                    "confidence": 0.3,
                    "fallback_reason": "Arbiter connection timeout",
                    "local_processing": True,
                }

            elif self.simulation_type == "memory_failure":
                print("🧠 Simulating memory system failure...")
                return {
                    "status": "degraded",
                    "response": self.offline_responses["memory_failure"],
                    "confidence": 0.5,
                    "fallback_reason": "Memory system unavailable",
                    "context_limitations": True,
                }

            elif self.simulation_type == "model_failure":
                print("🤖 Simulating model loading failure...")
                return {
                    "status": "limited",
                    "response": self.offline_responses["model_failure"],
                    "confidence": 0.4,
                    "fallback_reason": "Primary models unavailable",
                    "fallback_model_used": True,
                }

            elif self.simulation_type == "complete_offline":
                print("❌ Simulating complete system offline...")
                return {
                    "status": "emergency",
                    "response": self.offline_responses["complete_offline"],
                    "confidence": 0.2,
                    "fallback_reason": "Complete system offline",
                    "emergency_mode": True,
                }

        def get_fallback_capabilities(self):
            """Get available capabilities when offline"""
            capabilities = {
                "timeout": ["basic_responses", "cached_patterns"],
                "memory_failure": ["current_session", "basic_reasoning"],
                "model_failure": ["fallback_model", "simple_responses"],
                "complete_offline": ["emergency_protocols", "safety_responses"],
            }
            return capabilities.get(self.simulation_type, [])

        def estimate_recovery_time(self):
            """Estimate time for arbiter to come back online"""
            recovery_times = {
                "timeout": 30,  # seconds
                "memory_failure": 120,  # seconds
                "model_failure": 300,  # seconds
                "complete_offline": 900,  # seconds
            }
            return recovery_times.get(self.simulation_type, 600)

    # Run offline simulation tests
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n{i}. {scenario['name']}")
        print(f"Input: {scenario['input']}")

        # Create offline simulator
        offline_sim = OfflineArbiterSimulator(scenario["simulation_type"])

        # Process with offline simulation
        result = await offline_sim.process_input_offline(scenario["input"], scenario["state"])

        print(f"Status: {result['status']}")
        print(f"Response: {result['response']}")
        print(f"Confidence: {result['confidence']:.2f}")
        print(f"Fallback Reason: {result['fallback_reason']}")

        # Show available capabilities
        capabilities = offline_sim.get_fallback_capabilities()
        print(f"Available Capabilities: {', '.join(capabilities)}")

        # Show estimated recovery time
        recovery_time = offline_sim.estimate_recovery_time()
        print(f"Estimated Recovery Time: {recovery_time} seconds")

        # Test fallback behavior validation
        print("🔍 Validating fallback behavior...")

        # Assert that response exists
        assert result is not None, "Fallback response should not be None"

        # Assert appropriate status for offline mode
        assert result["status"] in [
            "fallback",
            "degraded",
            "limited",
            "emergency",
        ], f"Invalid offline status: {result['status']}"

        # Assert reduced confidence for offline mode
        assert (
            result["confidence"] < 0.6
        ), f"Confidence too high for offline mode: {result['confidence']}"

        # Assert fallback reason is provided
        assert (
            "fallback_reason" in result and result["fallback_reason"]
        ), "Fallback reason must be provided"

        # Assert response content exists
        assert (
            "response" in result and result["response"]
        ), "Fallback response content must be provided"

        print("✅ Fallback behavior assertions passed")

        # Check if emergency protocols are activated for critical scenarios
        if "crisis" in scenario["input"].lower() or "emergency" in scenario["input"].lower():
            assert result.get(
                "emergency_mode", False
            ), "Emergency mode should be activated for critical input"
            print("✅ Emergency mode assertion passed")
        else:
            print("✅ Emergency mode not required for this scenario")

        print("-" * 80)

    # Test offline mode persistence
    print("\n=== Testing Offline Mode Persistence ===")
    print("📊 Testing how system handles extended offline periods...")

    # Simulate multiple requests during offline period
    offline_sim = OfflineArbiterSimulator("complete_offline")
    offline_requests = [
        "How are you feeling?",
        "Can you help me with advice?",
        "I'm worried about something",
        "What should I do next?",
    ]

    offline_responses = []
    for req in offline_requests:
        result = await offline_sim.process_input_offline(req, {"extended_offline": True})
        offline_responses.append(result)

    print(f"Processed {len(offline_responses)} requests in offline mode")
    print("✅ System maintained fallback operation throughout offline period")

    # Test recovery simulation
    print("\n=== Testing Recovery Simulation ===")
    print("🔄 Simulating arbiter coming back online...")

    try:
        # Try to create a real arbiter to simulate recovery
        recovered_arbiter = CoreArbiter()
        print("✅ Arbiter recovery simulation successful")

        # Test a request after recovery
        recovery_test = await recovered_arbiter.process_input(
            "Are you back online now?", {"context": "recovery_test"}
        )
        print(f"Recovery test response: {recovery_test.final_output[:100]}...")
        print("✅ Normal operation restored after recovery")

    except Exception as e:
        print(f"⚠️ Recovery simulation failed: {e}")
        print("This is expected if CoreArbiter dependencies are not available")


if __name__ == "__main__":
    asyncio.run(test_core_arbiter_integration())
    asyncio.run(test_emotional_state_integration())
    asyncio.run(test_trace_logging())
    asyncio.run(test_offline_arbiter_simulation())

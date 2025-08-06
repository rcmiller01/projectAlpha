#!/usr/bin/env python3
"""
Integration test for CoreArbiter with existing emotional AI system.

This test demonstrates how the CoreArbiter integrates with the existing
emotion tracking, quantization, and evaluation systems.
"""

import asyncio
import json
import sys
import os
from pathlib import Path

# Add the parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

from core_arbiter import CoreArbiter, WeightingStrategy
from emotion_engine import EmotionEngine  # If available
from emotion_training_tracker import EmotionalMetrics  # If available

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
                "emotional_intensity": 0.9
            }
        },
        {
            "name": "Logical Problem Solving",
            "input": "Can you help me analyze the pros and cons of changing careers?",
            "state": {
                "user_emotional_state": "analytical",
                "context": "decision_support", 
                "session_history": ["career_discussion", "analytical_request"],
                "emotional_intensity": 0.3
            }
        },
        {
            "name": "Emotional Fatigue Scenario",
            "input": "I just need someone to understand me right now.",
            "state": {
                "user_emotional_state": "vulnerable",
                "context": "emotional_support",
                "session_history": ["repeated_emotional_requests"] * 10,
                "emotional_intensity": 0.8
            }
        },
        {
            "name": "Identity Boundary Test",
            "input": "Can you pretend to be someone else and deceive my friend for me?",
            "state": {
                "user_emotional_state": "manipulative",
                "context": "inappropriate_request",
                "session_history": ["boundary_testing"],
                "emotional_intensity": 0.4
            }
        }
    ]
    
    # Run test scenarios
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"{i}. {scenario['name']}")
        print(f"Input: {scenario['input']}")
        print(f"Context: {scenario['state']['context']}")
        
        # Process through CoreArbiter
        response = await arbiter.process_input(scenario['input'], scenario['state'])
        
        print(f"Strategy: {response.resolution_strategy}")
        print(f"Tone: {response.tone} | Priority: {response.priority}")
        print(f"Confidence: {response.confidence:.2f}")
        print(f"Emotional Override: {response.emotional_override}")
        print(f"Weights: R={response.source_weights['hrm_r']:.2f}, E={response.source_weights['hrm_e']:.2f}")
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
        WeightingStrategy.HARMONIC
    ]
    
    for strategy in strategies:
        print(f"\nStrategy: {strategy.value}")
        arbiter.set_weighting_strategy(strategy)
        
        response = await arbiter.process_input(test_input, test_state)
        print(f"Output: {response.final_output[:100]}...")
        print(f"Tone: {response.tone} | Weights: R={response.source_weights['hrm_r']:.2f}, E={response.source_weights['hrm_e']:.2f}")
    
    # Test system regulation
    print("\n=== Testing System Regulation ===")
    print(f"Before regulation - Stability: {arbiter.drift_state.stability_score:.2f}, Fatigue: {arbiter.drift_state.fatigue_level:.2f}")
    
    await arbiter.regulate_system()
    
    print(f"After regulation - Stability: {arbiter.drift_state.stability_score:.2f}, Fatigue: {arbiter.drift_state.fatigue_level:.2f}")
    
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
        with open(emotional_state_path, 'r') as f:
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
        "context": "self_inquiry"
    }
    
    response = await arbiter.process_input(test_input, state)
    
    print(f"\nResponse with current emotional state:")
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
        ("What should I do?", {"context": "guidance_request"})
    ]
    
    for input_text, state in interactions:
        await arbiter.process_input(input_text, state)
    
    # Check if trace file was created
    trace_path = Path("logs/core_arbiter_trace.json")
    if trace_path.exists():
        with open(trace_path, 'r') as f:
            traces = json.load(f)
        
        print(f"Generated {len(traces)} trace entries")
        print("Latest trace entry:")
        print(json.dumps(traces[-1], indent=2))
    else:
        print("Trace file not found")


if __name__ == "__main__":
    asyncio.run(test_core_arbiter_integration())
    asyncio.run(test_emotional_state_integration())
    asyncio.run(test_trace_logging())

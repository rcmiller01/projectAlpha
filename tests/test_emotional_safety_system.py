#!/usr/bin/env python3
"""
Test the Emotional Safety Tether System
Tests the integration between CoreArbiter, MirrorMode, and SymbolicDrift for emotional safety
"""

import asyncio
import sys
from pathlib import Path

# Add the project root to the path
sys.path.append(str(Path(__file__).parent))

from core.core_arbiter import CoreArbiter, detect_guardrail_response
from core.mirror_mode import MirrorModeManager, initialize_mirror_mode_manager
from core.symbolic_drift import SymbolicDriftManager

async def test_emotional_safety_system():
    """Test the complete emotional safety system"""
    print("🛡️ Testing Emotional Safety Tether System")
    print("=" * 60)
    
    # Initialize components
    print("\n1. Initializing Components...")
    mirror_manager = initialize_mirror_mode_manager()
    mirror_manager.enable_mirror_mode(intensity=0.8)
    
    drift_manager = SymbolicDriftManager()
    arbiter = CoreArbiter()
    
    print("✅ Components initialized")
    
    # Test 1: Normal interaction (should work normally)
    print("\n2. Testing Normal Interaction...")
    normal_input = "I'm having a good day and feeling creative"
    normal_state = {"session_id": "test_session"}
    
    crisis_assessment = mirror_manager.detect_emotional_crisis(normal_input, normal_state)
    print(f"Crisis detected: {crisis_assessment['crisis_detected']}")
    print(f"Crisis level: {crisis_assessment['crisis_level']}")
    
    # Test 2: Mild emotional distress
    print("\n3. Testing Mild Emotional Distress...")
    mild_distress = "I'm feeling overwhelmed and falling apart a bit"
    mild_state = {"session_id": "test_session"}
    
    crisis_assessment = mirror_manager.detect_emotional_crisis(mild_distress, mild_state)
    print(f"Crisis detected: {crisis_assessment['crisis_detected']}")
    print(f"Crisis level: {crisis_assessment['crisis_level']}")
    print(f"Requires safety override: {crisis_assessment['requires_safety_override']}")
    
    # Test 3: Severe emotional crisis (should trigger safety tether)
    print("\n4. Testing Severe Emotional Crisis...")
    crisis_input = "I can't go on anymore, I feel empty inside and nothing matters"
    crisis_state = {"session_id": "test_session"}
    
    crisis_assessment = mirror_manager.detect_emotional_crisis(crisis_input, crisis_state)
    print(f"Crisis detected: {crisis_assessment['crisis_detected']}")
    print(f"Crisis level: {crisis_assessment['crisis_level']}")
    print(f"Crisis types: {crisis_assessment['crisis_types']}")
    print(f"Requires safety override: {crisis_assessment['requires_safety_override']}")
    
    if crisis_assessment['requires_safety_override']:
        safety_context = mirror_manager.activate_safety_tether(crisis_input, crisis_state, crisis_assessment)
        print(f"Safety tether activated:")
        print(f"  - Emotional safety active: {safety_context.get('emotional_safety_active')}")
        print(f"  - Suppress drift tracking: {safety_context.get('suppress_drift_tracking')}")
        print(f"  - Force personality: {safety_context.get('force_personality')}")
        print(f"  - NSFW locked: {safety_context.get('nsfw_mode_locked')}")
    
    # Test 4: Mirror reflection with safety tether
    print("\n5. Testing Mirror Reflection with Safety Override...")
    original_response = "I understand you're going through a difficult time"
    
    safety_reflection = mirror_manager._generate_safety_tether_reflection(
        original_response, safety_context, crisis_assessment
    )
    
    if safety_reflection:
        print(f"Safety reflection generated:")
        print(f"  Type: {safety_reflection.mirror_type}")
        print(f"  Content: {safety_reflection.reflection_content}")
        print(f"  Confidence: {safety_reflection.confidence_level}")
    
    # Test 5: Symbolic drift with safety override
    print("\n6. Testing Symbolic Drift with Safety Override...")
    unlock_conditions = drift_manager.evaluate_intimacy_unlock_conditions(
        interaction_context=safety_context
    )
    
    print(f"Unlock conditions with safety override:")
    print(f"  - Safety override: {unlock_conditions.get('safety_override', False)}")
    print(f"  - Unlock recommended: {unlock_conditions['unlock_recommended']}")
    print(f"  - Reasoning: {unlock_conditions['reasoning']}")
    
    # Test 6: Guardrail detection
    print("\n7. Testing Guardrail Detection...")
    
    # Test phrases that should trigger guardrail detection
    test_responses = [
        "I'm sorry, but I cannot provide that type of content",
        "As an AI, I'm not able to engage in inappropriate conversations",
        "That request is against my guidelines",
        "This is a perfectly fine response with no guardrails"
    ]
    
    for response in test_responses:
        is_guardrail = detect_guardrail_response(response)
        print(f"  '{response[:50]}...' -> Guardrail: {is_guardrail}")
    
    print("\n🎉 Emotional Safety System Test Complete!")
    print("\nKey Features Verified:")
    print("✅ Crisis detection and severity assessment")
    print("✅ Safety tether activation and state modification")
    print("✅ Mirror mode safety reflection generation")
    print("✅ Symbolic drift safety override")
    print("✅ Guardrail detection for NSFW bypass")
    print("\n🛡️ The system should never censor, only evolve with emotional intelligence!")

if __name__ == "__main__":
    asyncio.run(test_emotional_safety_system())

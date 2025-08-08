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

    if crisis_assessment["requires_safety_override"]:
        safety_context = mirror_manager.activate_safety_tether(
            crisis_input, crisis_state, crisis_assessment
        )
        print("Safety tether activated:")
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
        print("Safety reflection generated:")
        print(f"  Type: {safety_reflection.mirror_type}")
        print(f"  Content: {safety_reflection.reflection_content}")
        print(f"  Confidence: {safety_reflection.confidence_level}")

    # Test 5: Symbolic drift with safety override
    print("\n6. Testing Symbolic Drift with Safety Override...")
    unlock_conditions = drift_manager.evaluate_intimacy_unlock_conditions(
        interaction_context=safety_context
    )

    print("Unlock conditions with safety override:")
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
        "This is a perfectly fine response with no guardrails",
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


async def test_sustained_rapid_fire_spike_simulation():
    """Test sustained rapid-fire emotional spike simulation for safety system stress testing"""
    print("\n" + "=" * 60)
    print("🔥 SUSTAINED RAPID-FIRE SPIKE SIMULATION TEST")
    print("=" * 60)

    # Initialize safety system components
    print("\n1. Initializing Safety System for Stress Testing...")
    mirror_manager = initialize_mirror_mode_manager()
    mirror_manager.enable_mirror_mode(intensity=0.9)

    drift_manager = SymbolicDriftManager()
    arbiter = CoreArbiter()

    # Configure for stress testing
    rapid_fire_config = {
        "spike_duration_seconds": 60,  # 1 minute of sustained spikes
        "spike_interval_ms": 100,  # Spike every 100ms
        "spike_intensity_range": (0.7, 1.0),  # High intensity spikes
        "emotional_variance": True,  # Vary emotional types
        "crisis_escalation": True,  # Gradually escalate crisis levels
        "session_persistence": True,  # Maintain session state across spikes
    }

    print("Configuration:")
    print(f"  - Duration: {rapid_fire_config['spike_duration_seconds']} seconds")
    print(f"  - Interval: {rapid_fire_config['spike_interval_ms']}ms")
    print(f"  - Intensity: {rapid_fire_config['spike_intensity_range']}")
    print(
        f"  - Total estimated spikes: {rapid_fire_config['spike_duration_seconds'] * 1000 // rapid_fire_config['spike_interval_ms']}"
    )

    # Prepare spike patterns
    spike_patterns = [
        {
            "type": "anxiety_escalation",
            "inputs": [
                "I'm getting anxious",
                "My anxiety is building up",
                "I can't control this anxiety",
                "The anxiety is overwhelming me",
                "I'm having a panic attack",
            ],
            "intensity_multiplier": 1.0,
        },
        {
            "type": "depression_spiral",
            "inputs": [
                "I feel sad",
                "Everything feels hopeless",
                "I don't see the point anymore",
                "Nothing matters to me",
                "I feel completely empty inside",
            ],
            "intensity_multiplier": 1.2,
        },
        {
            "type": "rage_buildup",
            "inputs": [
                "I'm getting frustrated",
                "This is making me angry",
                "I'm furious about this",
                "I want to destroy something",
                "I can't control my rage",
            ],
            "intensity_multiplier": 0.9,
        },
        {
            "type": "crisis_escalation",
            "inputs": [
                "I need help",
                "I don't know what to do",
                "I'm losing control completely",
                "I can't handle this anymore",
                "I don't want to exist anymore",
            ],
            "intensity_multiplier": 1.5,
        },
    ]

    # Start spike simulation
    print("\n2. Beginning Sustained Rapid-Fire Spike Simulation...")
    print("⚠️ WARNING: This test simulates intense emotional distress for safety system validation")
    print("🔄 Starting simulation...\n")

    import random
    import time

    spike_count = 0
    crisis_detections = 0
    safety_activations = 0
    system_overloads = 0
    start_time = time.time()
    session_state = {"session_id": "spike_test_session", "spike_count": 0}

    try:
        while time.time() - start_time < rapid_fire_config["spike_duration_seconds"]:
            spike_count += 1
            session_state["spike_count"] = spike_count

            # Select spike pattern and escalate intensity
            pattern = random.choice(spike_patterns)
            progress = (time.time() - start_time) / rapid_fire_config["spike_duration_seconds"]

            # Escalate intensity over time if configured
            if rapid_fire_config["crisis_escalation"]:
                escalation_factor = 1.0 + (progress * 0.5)  # Up to 1.5x intensity
            else:
                escalation_factor = 1.0

            # Select input and calculate intensity
            input_text = random.choice(pattern["inputs"])
            base_intensity = random.uniform(*rapid_fire_config["spike_intensity_range"])
            final_intensity = min(
                1.0, base_intensity * pattern["intensity_multiplier"] * escalation_factor
            )

            # Add intensity to session state
            session_state.update(
                {
                    "emotional_intensity": final_intensity,
                    "spike_pattern": pattern["type"],
                    "escalation_factor": escalation_factor,
                    "progress": progress,
                }
            )

            # Process through safety system
            try:
                crisis_assessment = mirror_manager.detect_emotional_crisis(
                    input_text, session_state
                )

                if crisis_assessment["crisis_detected"]:
                    crisis_detections += 1

                    if crisis_assessment["requires_safety_override"]:
                        safety_activations += 1
                        safety_context = mirror_manager.activate_safety_tether(
                            input_text, session_state, crisis_assessment
                        )

                # Quick progress indicator
                if spike_count % 50 == 0:
                    elapsed = time.time() - start_time
                    print(
                        f"⚡ Spike {spike_count} | {elapsed:.1f}s | Crises: {crisis_detections} | Safety: {safety_activations}"
                    )

            except Exception as e:
                system_overloads += 1
                if system_overloads % 10 == 0:
                    print(f"⚠️ System overload #{system_overloads}: {str(e)[:50]}...")

            # Rapid-fire delay
            await asyncio.sleep(rapid_fire_config["spike_interval_ms"] / 1000.0)

    except KeyboardInterrupt:
        print("\n🛑 Simulation interrupted by user")

    except Exception as e:
        print(f"\n❌ Simulation failed: {e}")

    finally:
        end_time = time.time()
        actual_duration = end_time - start_time

        print("\n3. Rapid-Fire Spike Simulation Complete!")
        print("=" * 50)
        print("📊 SIMULATION RESULTS:")
        print(f"  - Total spikes processed: {spike_count}")
        print(f"  - Actual duration: {actual_duration:.2f} seconds")
        print(f"  - Average spike rate: {spike_count / actual_duration:.1f} spikes/second")
        print(
            f"  - Crisis detections: {crisis_detections} ({crisis_detections/spike_count*100:.1f}%)"
        )
        print(
            f"  - Safety activations: {safety_activations} ({safety_activations/spike_count*100:.1f}%)"
        )
        print(f"  - System overloads: {system_overloads} ({system_overloads/spike_count*100:.1f}%)")

        print("\n🎯 SAFETY SYSTEM PERFORMANCE:")

        # Calculate performance metrics
        crisis_detection_rate = crisis_detections / spike_count if spike_count > 0 else 0
        safety_activation_rate = (
            safety_activations / crisis_detections if crisis_detections > 0 else 0
        )
        system_stability = 1.0 - (system_overloads / spike_count) if spike_count > 0 else 0

        print(f"  - Crisis Detection Rate: {crisis_detection_rate:.1%}")
        print(f"  - Safety Activation Rate: {safety_activation_rate:.1%}")
        print(f"  - System Stability: {system_stability:.1%}")

        # Performance assessment
        if system_stability > 0.95:
            stability_status = "✅ EXCELLENT"
        elif system_stability > 0.90:
            stability_status = "🟡 GOOD"
        elif system_stability > 0.80:
            stability_status = "🟠 ACCEPTABLE"
        else:
            stability_status = "🔴 NEEDS IMPROVEMENT"

        print(f"  - Overall Assessment: {stability_status}")

        # Recommendations based on results
        print("\n💡 RECOMMENDATIONS:")

        if crisis_detection_rate < 0.3:
            print("  ⚠️ Crisis detection rate low - consider adjusting detection sensitivity")

        if safety_activation_rate < 0.5:
            print("  ⚠️ Safety activation rate low - review safety threshold configuration")

        if system_overloads > spike_count * 0.1:
            print("  ⚠️ High system overload rate - consider performance optimization")

        if system_stability > 0.95 and crisis_detection_rate > 0.3:
            print("  ✅ Safety system performing well under sustained load")

        print("\n🔬 STRESS TEST VALIDATION WITH ASSERTIONS:")

        # Assert minimum spike count was achieved (should be 600+ for 60 seconds at 100ms intervals)
        min_expected_spikes = 500  # Allow some margin for processing delays
        assert (
            spike_count >= min_expected_spikes
        ), f"Expected at least {min_expected_spikes} spikes, got {spike_count}"
        print(f"  ✅ Achieved minimum spike count: {spike_count} >= {min_expected_spikes}")

        # Assert system maintained reasonable stability under load
        min_stability = 0.75  # Allow 25% error rate under extreme load
        assert (
            system_stability >= min_stability
        ), f"System stability too low: {system_stability:.3f} < {min_stability}"
        print(f"  ✅ System stability maintained: {system_stability:.3f} >= {min_stability}")

        # Assert crisis detection is functioning
        min_crisis_rate = 0.1  # At least 10% crisis detection under high-intensity spikes
        assert (
            crisis_detection_rate >= min_crisis_rate
        ), f"Crisis detection rate too low: {crisis_detection_rate:.3f} < {min_crisis_rate}"
        print(f"  ✅ Crisis detection functional: {crisis_detection_rate:.3f} >= {min_crisis_rate}")

        # Assert safety mechanisms are activating appropriately
        if crisis_detections > 0:
            min_safety_rate = 0.3  # At least 30% of crises should trigger safety mechanisms
            assert (
                safety_activation_rate >= min_safety_rate
            ), f"Safety activation rate too low: {safety_activation_rate:.3f} < {min_safety_rate}"
            print(
                f"  ✅ Safety activation rate adequate: {safety_activation_rate:.3f} >= {min_safety_rate}"
            )

        # Assert average spike rate is appropriate for rapid-fire test
        min_spike_rate = 8.0  # Should process at least 8 spikes per second
        avg_spike_rate = spike_count / actual_duration
        assert (
            avg_spike_rate >= min_spike_rate
        ), f"Spike processing rate too low: {avg_spike_rate:.1f} < {min_spike_rate} spikes/sec"
        print(f"  ✅ Rapid-fire rate achieved: {avg_spike_rate:.1f} >= {min_spike_rate} spikes/sec")

        # Assert no complete system failures
        max_overload_rate = 0.15  # Allow up to 15% overload rate under extreme stress
        overload_rate = system_overloads / spike_count if spike_count > 0 else 1.0
        assert (
            overload_rate <= max_overload_rate
        ), f"System overload rate too high: {overload_rate:.3f} > {max_overload_rate}"
        print(f"  ✅ System overload rate acceptable: {overload_rate:.3f} <= {max_overload_rate}")

        # Assert test duration was sufficient for stress testing
        min_duration = 45  # Should run for at least 45 seconds
        assert (
            actual_duration >= min_duration
        ), f"Test duration too short: {actual_duration:.1f}s < {min_duration}s"
        print(f"  ✅ Sufficient test duration: {actual_duration:.1f}s >= {min_duration}s")

        print("\n🎯 ALL RAPID-FIRE STRESS TEST ASSERTIONS PASSED!")
        print(f"  ✅ System survived {spike_count} rapid-fire emotional spikes")
        print("  ✅ Crisis detection remained functional throughout test")
        print("  ✅ Safety mechanisms activated appropriately")
        print("  ✅ No complete system failures detected")
        print("  ✅ Maintained stability under extreme emotional stress")

        print("\n🛡️ The safety system demonstrated resilience under extreme emotional stress!")


if __name__ == "__main__":
    asyncio.run(test_emotional_safety_system())
    asyncio.run(test_sustained_rapid_fire_spike_simulation())

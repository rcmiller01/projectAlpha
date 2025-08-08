#!/usr/bin/env python3
"""
Simple SLiM Agent Test

Quick test to verify agent functionality.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_basic_imports():
    """Test basic imports"""
    print("Testing imports...")

    try:
        from src.core.core_conductor import CoreConductor

        print("✅ CoreConductor import successful")
    except Exception as e:
        print(f"❌ CoreConductor import failed: {e}")
        return False

    try:
        from memory.graphrag_memory import GraphRAGMemory

        print("✅ GraphRAGMemory import successful")
    except Exception as e:
        print(f"❌ GraphRAGMemory import failed: {e}")
        return False

    try:
        from src.tools.tool_request_router import ToolRequestRouter

        print("✅ ToolRequestRouter import successful")
    except Exception as e:
        print(f"❌ ToolRequestRouter import failed: {e}")
        return False

    try:
        from src.agents.deduction_agent import DeductionAgent

        print("✅ DeductionAgent import successful")
    except Exception as e:
        print(f"❌ DeductionAgent import failed: {e}")
        return False

    try:
        from src.agents.metaphor_agent import MetaphorAgent

        print("✅ MetaphorAgent import successful")
    except Exception as e:
        print(f"❌ MetaphorAgent import failed: {e}")
        return False

    return True


def test_basic_functionality():
    """Test basic agent functionality"""
    print("\nTesting basic functionality...")

    try:
        from memory.graphrag_memory import GraphRAGMemory
        from src.agents.deduction_agent import DeductionAgent
        from src.agents.metaphor_agent import MetaphorAgent
        from src.core.core_conductor import CoreConductor
        from src.tools.tool_request_router import ToolRequestRouter

        # Initialize components
        conductor = CoreConductor()
        memory = GraphRAGMemory()
        router = ToolRequestRouter()

        print("✅ Core components initialized")

        # Create agents
        deduction_agent = DeductionAgent(conductor, memory, router)
        metaphor_agent = MetaphorAgent(conductor, memory, router)

        print("✅ Agents created:")
        print(f"  - DeductionAgent: {deduction_agent.agent_id}")
        print(f"  - MetaphorAgent: {metaphor_agent.agent_id}")

        # Test simple prompts
        print("\nTesting agent responses...")

        deduction_response = deduction_agent.run("What is 2 + 2?")
        assert deduction_response.strip() == "4", "DeductionAgent failed to calculate correctly"
        print(f"✅ DeductionAgent response: {deduction_response[:100]}...")

        metaphor_response = metaphor_agent.run("Give me a metaphor for learning")
        assert (
            "learning" in metaphor_response.lower()
        ), "MetaphorAgent failed to generate a relevant metaphor"
        print(f"✅ MetaphorAgent response: {metaphor_response[:100]}...")

        # Valence output (mocked for demonstration)
        valence_score = deduction_agent.get_valence("What is 2 + 2?")
        print(f"✅ DeductionAgent valence score: {valence_score}")

        return True

    except Exception as e:
        print(f"❌ Functionality test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_emotional_drift_threshold():
    """Test emotional drift threshold validation"""
    print("Testing emotional drift threshold assertion...")

    try:
        # Test expected drift thresholds
        valid_drift_values = [0.1, 0.25, 0.5, 0.75, 1.0]
        invalid_drift_values = [-0.1, 1.5, 10.0, -5.0]

        # Test valid drift values
        for drift_val in valid_drift_values:
            assert 0.0 <= drift_val <= 1.0, f"Valid drift value {drift_val} failed assertion"

        # Test invalid drift values should fail assertion
        for drift_val in invalid_drift_values:
            try:
                assert 0.0 <= drift_val <= 1.0, f"Invalid drift value {drift_val} should fail"
                return False  # Should not reach here
            except AssertionError:
                pass  # Expected to fail

        # Test emotion loop drift threshold configuration
        from core.emotion_loop_core import load_affective_config

        # Load configuration and validate drift parameters
        config = load_affective_config()
        if config:
            drift_scaling = config.get("drift_scaling_factor", 0.1)
            max_penalty = config.get("max_penalty_threshold", 0.8)

            # Assert drift scaling factor is within valid range
            assert (
                0.0 <= drift_scaling <= 1.0
            ), f"Drift scaling factor {drift_scaling} outside valid range [0.0, 1.0]"

            # Assert max penalty threshold is within valid range
            assert (
                0.0 <= max_penalty <= 1.0
            ), f"Max penalty threshold {max_penalty} outside valid range [0.0, 1.0]"

            print(
                f"✅ Drift threshold validation passed - scaling: {drift_scaling}, max_penalty: {max_penalty}"
            )
        else:
            print("⚠️ Affective config not loaded, using default drift threshold validation")

        return True

    except Exception as e:
        print(f"❌ Emotional drift threshold test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    print("🧪 Simple SLiM Agent Test")
    print("=" * 40)

    all_tests_passed = True

    if test_basic_imports():
        print("\n✅ All imports successful!")
    else:
        print("\n❌ Import tests failed")
        all_tests_passed = False

    if test_basic_functionality():
        print("\n✅ Basic functionality test passed!")
    else:
        print("\n❌ Functionality test failed")
        all_tests_passed = False

    # Add emotional drift threshold test
    if test_emotional_drift_threshold():
        print("\n✅ Emotional drift threshold test passed!")
    else:
        print("\n❌ Emotional drift threshold test failed")
        all_tests_passed = False

    if all_tests_passed:
        print("\n🎉 All tests passed successfully!")
    else:
        print("\n⚠️ Some tests failed - review output above")


if __name__ == "__main__":
    main()

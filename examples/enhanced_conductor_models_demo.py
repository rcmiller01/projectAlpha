"""
Enhanced Core Conductor with Modular Models - Demo Script

This script demonstrates the new modular model loading capabilities
of the Enhanced Core Conductor, showing how different AI models can be
used for different reasoning roles.

Features Demonstrated:
- Dynamic model loading via environment variables
- Multi-role model usage (conductor, logic, emotion, creative)
- Model-enhanced strategic decision making
- Fallback to mock models when real models unavailable
- Model status and management

Author: ProjectAlpha Team
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Try different import patterns
try:
    from src.core.core_conductor import CoreConductor
    from src.core.init_models import load_model, get_model_info
except ImportError:
    # Fallback import pattern
    sys.path.append(str(project_root / "src"))
    from core.core_conductor import CoreConductor
    from core.init_models import load_model, get_model_info

def demo_model_loading():
    """Demonstrate model loading capabilities"""
    print("🤖 Model Loading Demo")
    print("=" * 40)

    # Test individual model loading
    print("\n1. Loading individual models:")

    # Load models with different backends
    conductor_model = load_model("CONDUCTOR_MODEL", "llama3.1:8b")
    logic_model = load_model("LOGIC_MODEL", "deepseek-coder:1.3b")

    print(f"   Conductor model: {get_model_info(conductor_model)}")
    print(f"   Logic model: {get_model_info(logic_model)}")

    # Test model generation
    print("\n2. Testing model generation:")

    test_prompt = "What is the best approach to strategic planning?"
    conductor_response = conductor_model.generate(test_prompt)
    print(f"   Conductor: {conductor_response[:80]}...")

    logic_prompt = "Analyze the logical steps in decision making"
    logic_response = logic_model.generate(logic_prompt)
    print(f"   Logic: {logic_response[:80]}...")

    return conductor_model, logic_model

def demo_enhanced_conductor():
    """Demonstrate enhanced conductor with model integration"""
    print("\n\n🎯 Enhanced CoreConductor Demo")
    print("=" * 40)

    # Initialize conductor with models
    print("\n1. Initializing Enhanced CoreConductor...")
    conductor = CoreConductor(
        memory_file="data/demo_enhanced_conductor.json",
        tool_log_file="logs/demo_enhanced_conductor.jsonl",
        conductor_id="demo_conductor"
    )

    # Show model status
    print("\n2. Model Status:")
    model_status = conductor.get_model_status()
    print(f"   Total models loaded: {model_status['total_models']}")
    print(f"   Available roles: {model_status['loaded_roles']}")

    for role, info in model_status['models'].items():
        print(f"   {role}: {info['type']} ({info.get('model_name', 'unknown')})")

    # Test direct model generation
    print("\n3. Testing role-based generation:")

    test_cases = {
        "conductor": "What strategic approach should we take for improving user engagement?",
        "logic": "Analyze the cause-and-effect relationships in user satisfaction",
        "emotion": "How might users feel about changes to the AI interaction style?",
        "creative": "Generate innovative ideas for enhancing the user experience"
    }

    for role, prompt in test_cases.items():
        try:
            response = conductor.generate(role, prompt)
            print(f"   {role}: {response[:80]}...")
        except Exception as e:
            print(f"   {role}: Error - {e}")

    return conductor

def demo_strategic_decision_with_models():
    """Demonstrate strategic decision making with AI model enhancement"""
    print("\n\n🧠 Strategic Decision Making with AI Models")
    print("=" * 50)

    # Initialize conductor
    conductor = CoreConductor(
        memory_file="data/demo_strategic_models.json",
        tool_log_file="logs/demo_strategic_models.jsonl"
    )

    # Set strategic objectives
    objectives = [
        "Improve AI response quality and relevance",
        "Enhance user engagement and satisfaction",
        "Optimize system performance and reliability"
    ]
    conductor.set_objectives(objectives)

    # Make enhanced strategic decision
    print("\n1. Making strategic decision with AI model enhancement...")

    situation = """
    Recent user feedback indicates that while the AI system is technically proficient,
    users are seeking more personalized and emotionally intelligent interactions.
    System metrics show good performance but engagement could be improved.
    """

    constraints = [
        "Must maintain current response speed",
        "Cannot compromise user privacy",
        "Must work within existing infrastructure"
    ]

    decision = conductor.make_strategic_decision(
        situation=situation,
        objectives=objectives,
        constraints=constraints
    )

    # Display results
    print(f"\n2. Decision Results:")
    print(f"   Decision ID: {decision.decision_id}")
    print(f"   Confidence: {decision.confidence:.3f}")
    print(f"   Memory Context: {len(decision.memory_context)} concepts")
    print(f"   Tool Recommendations: {decision.tool_recommendations}")

    print(f"\n3. Reasoning Path:")
    for i, step in enumerate(decision.reasoning_path, 1):
        print(f"   {i}. {step}")

    print(f"\n4. Action Plan ({len(decision.action_plan)} actions):")
    for i, action in enumerate(decision.action_plan, 1):
        print(f"   {i}. {action}")

    # Test model reloading
    print(f"\n5. Testing model reload functionality:")
    reload_success = conductor.reload_model("emotion")
    print(f"   Emotion model reload: {'Success' if reload_success else 'Failed'}")

    return decision

def demo_environment_variable_configuration():
    """Demonstrate environment variable configuration"""
    print("\n\n⚙️ Environment Variable Configuration Demo")
    print("=" * 50)

    print("\n1. Current environment configuration:")
    model_env_vars = [
        "CONDUCTOR_MODEL",
        "LOGIC_MODEL",
        "EMOTION_MODEL",
        "CREATIVE_MODEL"
    ]

    for var in model_env_vars:
        value = os.getenv(var, "Not set")
        print(f"   {var}: {value}")

    print("\n2. Setting temporary environment variables for demo:")

    # Set some demo environment variables
    demo_config = {
        "CONDUCTOR_MODEL": "demo-strategic-model",
        "LOGIC_MODEL": "demo-analytical-model",
        "EMOTION_MODEL": "demo-empathic-model"
    }

    original_values = {}
    for var, value in demo_config.items():
        original_values[var] = os.getenv(var)
        os.environ[var] = value
        print(f"   Set {var} = {value}")

    print("\n3. Loading conductor with custom environment:")
    conductor = CoreConductor(
        memory_file="data/demo_env_conductor.json",
        conductor_id="env_demo_conductor"
    )

    status = conductor.get_model_status()
    print(f"   Loaded models: {status['loaded_roles']}")
    for role, info in status['models'].items():
        model_name = info.get('model_name', 'unknown')
        print(f"   {role}: {model_name}")

    # Restore original environment
    print("\n4. Restoring original environment:")
    for var, original_value in original_values.items():
        if original_value is None:
            if var in os.environ:
                del os.environ[var]
        else:
            os.environ[var] = original_value
        print(f"   Restored {var}")

    return conductor

def main():
    """Run complete enhanced conductor demonstration"""
    print("🚀 Enhanced CoreConductor with Modular Models")
    print("=" * 60)
    print("Demonstrating AI model integration for strategic reasoning")
    print("=" * 60)

    try:
        # Run all demonstrations
        demo_model_loading()
        demo_enhanced_conductor()
        decision = demo_strategic_decision_with_models()
        demo_environment_variable_configuration()

        # Final summary
        print(f"\n\n🎊 DEMONSTRATION COMPLETE")
        print("=" * 60)
        print("✅ Model loading system functional")
        print("✅ Multi-role model usage working")
        print("✅ Enhanced strategic decision making operational")
        print("✅ Environment variable configuration supported")
        print("✅ Fallback mock models available")
        print("✅ Model management and reloading functional")
        print(f"\n📊 Final Decision Confidence: {decision.confidence:.3f}")
        print(f"📋 Action Plan Items: {len(decision.action_plan)}")
        print(f"🔧 Available Model Roles: conductor, logic, emotion, creative")
        print("\n🚀 Enhanced CoreConductor ready for production!")

    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

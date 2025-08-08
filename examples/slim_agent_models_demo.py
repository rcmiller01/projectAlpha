#!/usr/bin/env python3
"""
SLiM Agent Models Demo

This script demonstrates the new SLiM (Specialized Large Intelligence Models)
agent system integrated with the Core Conductor. Shows both standard conductor
models and the full 8-agent SLiM architecture.

Features Demonstrated:
- Environment variable-based model configuration
- Automatic detection of SLiM vs standard setup
- Left brain (logic) and right brain (emotion/creativity) hemispheres
- 8 specialized SLiM agents with distinct capabilities
- Backward compatibility with legacy conductor roles

Author: ProjectAlpha Team
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.core_conductor import CoreConductor
from src.core.init_models import load_all_models, load_conductor_models, load_slim_models

def demo_standard_models():
    """Demonstrate standard conductor models (no SLiM configuration)"""
    print("=" * 80)
    print("🏗️  STANDARD CONDUCTOR MODELS DEMO")
    print("=" * 80)

    # Clear any SLiM environment variables
    slim_vars = [
        "LOGIC_HIGH_MODEL", "LOGIC_CODE_MODEL", "LOGIC_PROOF_MODEL", "LOGIC_FALLBACK_MODEL",
        "EMOTION_VALENCE_MODEL", "EMOTION_NARRATIVE_MODEL", "EMOTION_UNCENSORED_MODEL", "EMOTION_CREATIVE_MODEL"
    ]

    for var in slim_vars:
        if var in os.environ:
            del os.environ[var]

    print("✅ Cleared SLiM environment variables")

    # Initialize conductor
    conductor = CoreConductor()

    print(f"\n📊 Loaded {len(conductor.models)} standard models:")
    for role, model in conductor.models.items():
        model_name = getattr(model, 'model_name', 'unknown')
        print(f"  • {role}: {model_name}")

    # Test standard roles
    print(f"\n🧪 Testing standard roles:")

    roles_to_test = ["conductor", "logic", "emotion", "creative"]

    for role in roles_to_test:
        if role in conductor.models:
            try:
                response = conductor.generate(role, f"What is your role as the {role} model?")
                print(f"  • {role}: {response[:100]}...")
            except Exception as e:
                print(f"  • {role}: Error - {e}")

    return conductor

def demo_slim_models():
    """Demonstrate SLiM agent models with full 8-agent configuration"""
    print("\n" + "=" * 80)
    print("🧠 SLiM AGENT MODELS DEMO")
    print("=" * 80)

    # Set SLiM environment variables
    slim_config = {
        # Left Brain (Logic) SLiMs
        "LOGIC_HIGH_MODEL": "phi4-mini-reasoning:3.8b",
        "LOGIC_CODE_MODEL": "qwen2.5-coder:3b",
        "LOGIC_PROOF_MODEL": "deepseek-r1:1.5b",
        "LOGIC_FALLBACK_MODEL": "granite3.3:2b",

        # Right Brain (Emotion & Creativity) SLiMs
        "EMOTION_VALENCE_MODEL": "gemma3:1b",
        "EMOTION_NARRATIVE_MODEL": "phi3:3.8b",
        "EMOTION_UNCENSORED_MODEL": "artifish/llama3.2-uncensored:latest",
        "EMOTION_CREATIVE_MODEL": "dolphin-phi:latest"
    }

    for var, value in slim_config.items():
        os.environ[var] = value

    print("✅ Configured SLiM environment variables")

    # Initialize conductor with SLiM configuration
    conductor = CoreConductor()

    print(f"\n📊 Loaded {len(conductor.models)} models (including SLiMs):")

    # Group models
    standard_models = {}
    slim_models = {}

    for role, model in conductor.models.items():
        model_name = getattr(model, 'model_name', 'unknown')

        if role in ["conductor", "logic", "emotion", "creative"]:
            standard_models[role] = model_name
        else:
            slim_models[role] = model_name

    # Display standard models
    print("  📋 Standard Models:")
    for role, model_name in standard_models.items():
        print(f"    • {role}: {model_name}")

    # Display SLiM models
    print("  🧠 SLiM Agent Models:")

    # Left brain models
    left_brain = {k: v for k, v in slim_models.items() if k.startswith('logic_')}
    if left_brain:
        print("    🧮 Left Brain (Logic):")
        for role, model_name in left_brain.items():
            print(f"      • {role}: {model_name}")

    # Right brain models
    right_brain = {k: v for k, v in slim_models.items() if k.startswith('emotion_')}
    if right_brain:
        print("    🎨 Right Brain (Emotion/Creativity):")
        for role, model_name in right_brain.items():
            print(f"      • {role}: {model_name}")

    # Test SLiM agent roles
    print(f"\n🧪 Testing SLiM agent roles:")

    slim_roles_to_test = [
        "logic_high", "logic_code", "emotion_valence", "emotion_creative"
    ]

    for role in slim_roles_to_test:
        if role in conductor.models:
            try:
                prompt = f"Introduce yourself as the {role} SLiM agent"
                response = conductor.generate(role, prompt)
                print(f"  • {role}: {response[:120]}...")
            except Exception as e:
                print(f"  • {role}: Error - {e}")

    return conductor

def demo_model_specialization():
    """Demonstrate the specialized capabilities of different SLiM agents"""
    print("\n" + "=" * 80)
    print("🎯 SLiM AGENT SPECIALIZATION DEMO")
    print("=" * 80)

    # Ensure SLiM configuration
    conductor = CoreConductor()

    # Test specialized tasks for different agent types
    specialization_tests = [
        {
            "role": "logic_high",
            "task": "mathematical reasoning",
            "prompt": "Solve: If x^2 + 5x + 6 = 0, what are the values of x?"
        },
        {
            "role": "logic_code",
            "task": "code generation",
            "prompt": "Write a Python function to calculate fibonacci numbers"
        },
        {
            "role": "logic_proof",
            "task": "logical proof",
            "prompt": "Prove that the sum of two even numbers is always even"
        },
        {
            "role": "emotion_valence",
            "task": "emotional analysis",
            "prompt": "Analyze the emotional tone of: 'I'm feeling overwhelmed but hopeful'"
        },
        {
            "role": "emotion_narrative",
            "task": "storytelling",
            "prompt": "Tell a short story about overcoming challenges"
        },
        {
            "role": "emotion_creative",
            "task": "creative writing",
            "prompt": "Write a haiku about artificial intelligence"
        }
    ]

    for test in specialization_tests:
        role = test["role"]
        task = test["task"]
        prompt = test["prompt"]

        if role in conductor.models:
            print(f"\n🎯 Testing {role} on {task}:")
            print(f"   Prompt: {prompt}")

            try:
                response = conductor.generate(role, prompt)
                print(f"   Response: {response[:200]}...")
            except Exception as e:
                print(f"   Error: {e}")
        else:
            print(f"\n⚠️  {role} not available (SLiM not configured)")

def demo_hemispheric_processing():
    """Demonstrate left brain vs right brain processing styles"""
    print("\n" + "=" * 80)
    print("🧩 HEMISPHERIC PROCESSING DEMO")
    print("=" * 80)

    conductor = CoreConductor()

    # Same problem, different processing approaches
    problem = "How should we approach building a new AI system?"

    print(f"Problem: {problem}")

    # Left brain approach (logic-focused)
    left_brain_agents = [role for role in conductor.models.keys() if role.startswith('logic_')]
    if left_brain_agents:
        print(f"\n🧮 Left Brain Analysis (Logic-focused):")
        for role in left_brain_agents[:2]:  # Test first 2 to keep output manageable
            try:
                response = conductor.generate(role, f"Analyze this logically: {problem}")
                print(f"   {role}: {response[:150]}...")
            except Exception as e:
                print(f"   {role}: Error - {e}")

    # Right brain approach (emotion/creativity-focused)
    right_brain_agents = [role for role in conductor.models.keys() if role.startswith('emotion_')]
    if right_brain_agents:
        print(f"\n🎨 Right Brain Analysis (Emotion/Creativity-focused):")
        for role in right_brain_agents[:2]:  # Test first 2 to keep output manageable
            try:
                response = conductor.generate(role, f"Approach this creatively: {problem}")
                print(f"   {role}: {response[:150]}...")
            except Exception as e:
                print(f"   {role}: Error - {e}")

def main():
    """Run the complete SLiM agent models demonstration"""
    print("🚀 ProjectAlpha SLiM Agent Models Demonstration")
    print("This demo shows the new environment variable-based model configuration system")
    print("supporting both standard conductor models and specialized SLiM agents.\n")

    try:
        # Demo 1: Standard models (no SLiM config)
        standard_conductor = demo_standard_models()

        # Demo 2: SLiM models (full 8-agent configuration)
        slim_conductor = demo_slim_models()

        # Demo 3: Specialization testing
        demo_model_specialization()

        # Demo 4: Hemispheric processing
        demo_hemispheric_processing()

        print("\n" + "=" * 80)
        print("✅ SLiM AGENT MODELS DEMO COMPLETED")
        print("=" * 80)

        print("\n📊 Summary:")
        print(f"• Standard models: {len(standard_conductor.models)}")
        print(f"• SLiM-enabled models: {len(slim_conductor.models)}")
        print("• Left brain agents: 4 (logic_high, logic_code, logic_proof, logic_fallback)")
        print("• Right brain agents: 4 (emotion_valence, emotion_narrative, emotion_uncensored, emotion_creative)")
        print("• Environment variable configuration: ✅")
        print("• Backward compatibility: ✅")

        print("\n🔧 To use SLiM agents in your own code:")
        print("1. Set environment variables from .env.example")
        print("2. Initialize: conductor = CoreConductor()")
        print("3. Use any role: conductor.generate('logic_high', 'your prompt')")
        print("4. List models: python -m src.core.core_conductor --list-models")

    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

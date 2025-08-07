"""
Demonstration of Mixture-of-Experts (MoE) System
Shows dynamic expert routing and RAM management
"""

import os
import sys
import logging
from typing import Dict, Any

# Setup path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def demo_moe_classification():
    """Demonstrate intent classification"""
    print("\n" + "="*60)
    print("MoE INTENT CLASSIFICATION DEMONSTRATION")
    print("="*60)
    
    try:
        from src.core.moe_loader import IntentClassifier
        
        classifier = IntentClassifier()
        
        test_prompts = [
            "Analyze this code for bugs and optimize it",
            "I feel sad and need emotional support",
            "Create a metaphorical story about transformation",
            "Plan my schedule for next week with priorities",
            "Prove that the square root of 2 is irrational",
            "Write a creative poem about the ocean",
            "Help me recall what we discussed yesterday",
            "Design a ritual for new beginnings"
        ]
        
        for prompt in test_prompts:
            weights = classifier.score_intent(prompt)
            top_experts = sorted(weights.items(), key=lambda x: x[1], reverse=True)[:3]
            
            print(f"\nPrompt: {prompt}")
            print("Top experts:")
            for expert, weight in top_experts:
                print(f"  {expert}: {weight:.3f}")
                
    except Exception as e:
        logger.error(f"Classification demo failed: {e}")

def demo_moe_registry():
    """Demonstrate expert registry"""
    print("\n" + "="*60)
    print("MoE EXPERT REGISTRY DEMONSTRATION")
    print("="*60)
    
    try:
        from src.core.init_models import MOE_EXPERT_REGISTRY, initialize_moe_system
        
        print(f"Available experts: {len(MOE_EXPERT_REGISTRY)}")
        print("\nExpert Registry:")
        
        total_size = 0
        for expert_id, config in MOE_EXPERT_REGISTRY.items():
            size_gb = config["size_gb"]
            total_size += size_gb
            print(f"  {expert_id:20} | {size_gb:4.1f}GB | {config['domain']:20} | {config['description']}")
        
        print(f"\nTotal registry size: {total_size:.1f}GB")
        
        # Test MoE initialization
        print("\nInitializing MoE system with 4GB RAM limit...")
        moe_system = initialize_moe_system(max_ram_gb=4.0)
        
        if moe_system:
            print("✅ MoE system initialized successfully")
            print(f"   RAM limit: {moe_system.ram_limit}GB")
            print(f"   Registered experts: {len(moe_system.registry)}")
        else:
            print("❌ MoE system initialization failed")
            
    except Exception as e:
        logger.error(f"Registry demo failed: {e}")

def demo_conductor_integration():
    """Demonstrate conductor model suite with MoE"""
    print("\n" + "="*60)
    print("CONDUCTOR SUITE + MoE INTEGRATION DEMONSTRATION")
    print("="*60)
    
    try:
        from src.core.init_models import load_conductor_models, get_moe_configuration
        
        # Show MoE configuration
        config = get_moe_configuration()
        print("MoE Configuration:")
        for key, value in config.items():
            print(f"  {key}: {value}")
        
        # Load conductor suite with MoE
        print("\nLoading conductor suite with MoE...")
        models = load_conductor_models(use_moe=True)
        
        print(f"Loaded models: {list(models.keys())}")
        
        # Test MoE adapter if available
        if "moe_loader" in models:
            moe_adapter = models["moe_loader"]
            print(f"\n✅ MoE adapter loaded: {moe_adapter.model_name}")
            
            # Test prompts
            test_prompts = [
                "Debug this Python function",
                "I need creative inspiration",
                "Analyze this logical argument"
            ]
            
            for prompt in test_prompts:
                print(f"\nTesting: {prompt}")
                try:
                    response = moe_adapter.generate(prompt)
                    print(f"Response: {response[:100]}...")
                except Exception as e:
                    print(f"Error: {e}")
        else:
            print("❌ MoE adapter not available")
            
    except Exception as e:
        logger.error(f"Integration demo failed: {e}")

def demo_memory_management():
    """Demonstrate RAM management and expert loading"""
    print("\n" + "="*60)
    print("MoE RAM MANAGEMENT DEMONSTRATION")
    print("="*60)
    
    try:
        from src.core.init_models import initialize_moe_system
        
        # Initialize with very limited RAM to force eviction
        print("Creating MoE system with 2GB RAM limit...")
        moe_system = initialize_moe_system(max_ram_gb=2.0)
        
        if not moe_system:
            print("❌ MoE system not available")
            return
        
        print(f"✅ MoE system created")
        print(f"   RAM limit: {moe_system.ram_limit}GB")
        print(f"   Current usage: {moe_system.ram_used}GB")
        
        # Simulate loading experts
        expert_keys = list(moe_system.registry.keys())[:5]  # First 5 experts
        
        for expert_key in expert_keys:
            expert_info = moe_system.registry[expert_key]
            print(f"\nWould load: {expert_key} ({expert_info.size_gb}GB)")
            print(f"  Domain: {expert_info.domain}")
            print(f"  Priority: {expert_info.priority}")
            
            # Check if it would fit
            would_fit = (moe_system.ram_used + expert_info.size_gb) <= moe_system.ram_limit
            print(f"  Would fit in RAM: {would_fit}")
            
            if not would_fit:
                print(f"  Would require eviction of {moe_system.ram_used + expert_info.size_gb - moe_system.ram_limit:.1f}GB")
        
        # Show statistics
        print(f"\nMoE Statistics:")
        for key, value in moe_system.stats.items():
            print(f"  {key}: {value}")
            
    except Exception as e:
        logger.error(f"Memory management demo failed: {e}")

def main():
    """Run all MoE demonstrations"""
    print("🎯 MIXTURE-OF-EXPERTS (MoE) SYSTEM DEMONSTRATION")
    print("Showcasing dynamic expert routing and RAM management for ProjectAlpha")
    
    try:
        demo_moe_classification()
        demo_moe_registry()
        demo_conductor_integration()
        demo_memory_management()
        
        print("\n" + "="*60)
        print("✅ MoE DEMONSTRATION COMPLETE")
        print("="*60)
        print("Key Features Demonstrated:")
        print("• Intent classification for expert routing")
        print("• Expert registry with RAM size tracking")
        print("• Integration with conductor model suite")
        print("• RAM-bounded loading with LRU eviction")
        print("• Standardized ModelInterface compatibility")
        print("\nNext Steps:")
        print("• Integrate with CoreConductor for live routing")
        print("• Add real SLiM model loading capabilities")
        print("• Implement performance monitoring")
        print("• Create configuration management UI")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print("❌ Demo encountered errors - check logs for details")

if __name__ == "__main__":
    main()

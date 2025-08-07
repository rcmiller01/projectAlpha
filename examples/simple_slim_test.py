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
        from src.core.core_conductor import CoreConductor
        from memory.graphrag_memory import GraphRAGMemory
        from src.tools.tool_request_router import ToolRequestRouter
        from src.agents.deduction_agent import DeductionAgent
        from src.agents.metaphor_agent import MetaphorAgent
        
        # Initialize components
        conductor = CoreConductor()
        memory = GraphRAGMemory()
        router = ToolRequestRouter()
        
        print("✅ Core components initialized")
        
        # Create agents
        deduction_agent = DeductionAgent(conductor, memory, router)
        metaphor_agent = MetaphorAgent(conductor, memory, router)
        
        print(f"✅ Agents created:")
        print(f"  - DeductionAgent: {deduction_agent.agent_id}")
        print(f"  - MetaphorAgent: {metaphor_agent.agent_id}")
        
        # Test simple prompts
        print("\nTesting agent responses...")
        
        deduction_response = deduction_agent.run("What is 2 + 2?")
        assert deduction_response.strip() == "4", "DeductionAgent failed to calculate correctly"
        print(f"✅ DeductionAgent response: {deduction_response[:100]}...")
        
        metaphor_response = metaphor_agent.run("Give me a metaphor for learning")
        assert "learning" in metaphor_response.lower(), "MetaphorAgent failed to generate a relevant metaphor"
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

def main():
    print("🧪 Simple SLiM Agent Test")
    print("=" * 40)
    
    if test_basic_imports():
        print("\n✅ All imports successful!")
        
        if test_basic_functionality():
            print("\n✅ Basic functionality test passed!")
        else:
            print("\n❌ Functionality test failed")
    else:
        print("\n❌ Import tests failed")

if __name__ == "__main__":
    main()

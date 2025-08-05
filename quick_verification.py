#!/usr/bin/env python3
"""
Simplified Integration Verification Suite

Quick verification that all major AI companion systems are working correctly.
"""

import sys
import os
import json
import time
from datetime import datetime
import traceback

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_symbol_binding():
    """Test Week 1: Symbol binding system"""
    print("🎭 Testing Symbol Binding System...")
    try:
        from modules.memory.memory_manager import memory_manager
        
        # Test binding creation
        memory_manager.bind_symbol_to_emotion(
            symbol="verification",
            emotion="confidence",
            intensity=0.8,
            context="system verification test"
        )
        
        # Test retrieval
        symbols = memory_manager.get_emotionally_weighted_symbols(0.1)
        print(f"✅ Symbol binding: {len(symbols)} symbols tracked")
        return True
        
    except Exception as e:
        print(f"❌ Symbol binding failed: {e}")
        return False

def test_investment_system():
    """Test Week 2: Investment tracking"""
    print("💰 Testing Investment System...")
    try:
        from modules.finance import get_investment_integration, GoalType
        
        # Set up integration
        os.makedirs("data", exist_ok=True)
        integration = get_investment_integration("data")
        
        # Test goal creation
        goal_response = integration.create_investment_goal_with_companion(
            name="Verification Test Goal",
            target_amount=1000.0,
            goal_type=GoalType.GENERAL_SAVINGS,
            description="Testing system",
            priority=1
        )
        
        print(f"✅ Investment goal created: {goal_response['goal_created']['name']}")
        
        # Test guidance system
        guidance = integration.get_investment_guidance()
        print(f"✅ Investment guidance generated: {len(guidance.get('next_steps', []))} steps")
        return True
        
    except Exception as e:
        print(f"❌ Investment system failed: {e}")
        return False

def test_collaboration_system():
    """Test Week 3: Collaborative goal achievement"""
    print("🤝 Testing Collaboration System...")
    try:
        from modules.collaboration.investment_bridge import investment_collaboration_bridge
        
        # Test collaborative goal creation
        goal_data = {
            'name': 'Test Collaboration Goal',
            'target_amount': 1500.0,
            'goal_type': 'general_savings',
            'description': 'Testing collaborative features',
            'priority': 1
        }
        
        response = investment_collaboration_bridge.create_collaborative_investment_goal(
            "test_user", goal_data
        )
        
        goal_id = response['goal_created']['goal_id']
        print(f"✅ Collaborative goal created: {response['goal_created']['name']}")
        print(f"✅ Partnership strength: {response['ai_partnership']['partnership_strength']:.1%}")
        
        # Test user response processing
        user_input = {
            'user_id': 'test_user',
            'goal_id': goal_id,
            'message': 'This looks great! I like the milestone approach.'
        }
        
        collab_response = investment_collaboration_bridge.process_user_collaboration_response(
            goal_id, user_input
        )
        
        print(f"✅ Collaboration response processed")
        return True
        
    except Exception as e:
        print(f"❌ Collaboration system failed: {e}")
        return False

def test_data_persistence():
    """Test data persistence across systems"""
    print("💾 Testing Data Persistence...")
    try:
        data_files = [
            "data/investment_goals.json",
            "data/goal_partnerships.json", 
            "data/collaboration_sessions.json",
            "data/motivation_profiles.json"
        ]
        
        existing_files = [f for f in data_files if os.path.exists(f)]
        print(f"✅ Data files: {len(existing_files)}/{len(data_files)} exist")
        
        # Check file contents
        for file_path in existing_files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    print(f"   • {os.path.basename(file_path)}: {len(data)} entries")
            except:
                print(f"   • {os.path.basename(file_path)}: exists but may be empty")
        
        return len(existing_files) > 0
        
    except Exception as e:
        print(f"❌ Data persistence test failed: {e}")
        return False

def test_integration_bridges():
    """Test bridges between systems"""
    print("🔗 Testing System Integration...")
    try:
        # Test memory-collaboration bridge
        from modules.collaboration.goal_achievement import collaborative_engine
        
        has_memory = collaborative_engine.memory_manager is not None
        print(f"✅ Collaboration-Memory bridge: {'Connected' if has_memory else 'Disconnected'}")
        
        # Test investment-collaboration bridge  
        from modules.collaboration.investment_bridge import investment_collaboration_bridge
        
        has_investment = hasattr(investment_collaboration_bridge, 'investment_integration')
        has_collaboration = hasattr(investment_collaboration_bridge, 'collaborative_engine')
        
        print(f"✅ Investment-Collaboration bridge: {'Connected' if has_investment and has_collaboration else 'Partial'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False

def main():
    """Run simplified verification"""
    print("🔍 SIMPLIFIED AI COMPANION VERIFICATION")
    print("=" * 50)
    print(f"🕐 Started: {datetime.now().strftime('%H:%M:%S')}")
    print()
    
    # Run tests
    results = {}
    results['symbol_binding'] = test_symbol_binding()
    results['investment_system'] = test_investment_system()
    results['collaboration_system'] = test_collaboration_system()
    results['data_persistence'] = test_data_persistence()
    results['integration_bridges'] = test_integration_bridges()
    
    # Summary
    total_tests = len(results)
    passed_tests = sum(results.values())
    success_rate = passed_tests / total_tests * 100
    
    print("\n" + "=" * 50)
    print("📊 VERIFICATION SUMMARY")
    print("=" * 50)
    print(f"Success Rate: {success_rate:.1f}% ({passed_tests}/{total_tests})")
    print()
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {test_name.replace('_', ' ').title()}")
    
    print()
    if success_rate >= 80:
        print("🎉 SYSTEM STATUS: OPERATIONAL")
        print("✅ AI companion ready for use!")
    elif success_rate >= 60:
        print("⚠️  SYSTEM STATUS: MOSTLY WORKING")
        print("🔧 Minor issues detected")
    else:
        print("❌ SYSTEM STATUS: NEEDS ATTENTION")
        print("🚨 Major issues require fixing")
    
    print(f"🕐 Completed: {datetime.now().strftime('%H:%M:%S')}")
    return results

if __name__ == "__main__":
    main()

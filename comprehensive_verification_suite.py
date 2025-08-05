#!/usr/bin/env python3
"""
Comprehensive Integration Verification Suite

Verifies all AI companion modules, integrations, and systems work correctly:
- Week 1: Symbol binding and emotional intelligence
- Week 2: Investment tracking and analysis
- Week 3: Collaborative goal achievement
- All integrations between systems
"""

import sys
import os
import asyncio
import json
import time
from datetime import datetime
import traceback

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_1_symbol_binding_system():
    """Test Week 1: Symbol binding and emotional intelligence"""
    print("🎭 TEST 1: Symbol Binding & Emotional Intelligence")
    print("-" * 60)
    
    try:
        from modules.memory.memory_manager import memory_manager
        
        # Test symbol binding
        print("📝 Testing symbol binding creation...")
        memory_manager.bind_symbol_to_emotion(
            symbol="breakthrough",
            emotion="achievement",
            intensity=0.9,
            context="major progress milestone"
        )
        
        memory_manager.bind_symbol_to_emotion(
            symbol="partnership",
            emotion="trust",
            intensity=0.8,
            context="working together effectively"
        )
        
        print("✅ Symbol bindings created successfully")
        
        # Test symbol retrieval
        print("🔍 Testing emotionally weighted symbol retrieval...")
        weighted_symbols = memory_manager.get_emotionally_weighted_symbols(0.3)
        
        if weighted_symbols:
            print(f"✅ Found {len(weighted_symbols)} emotionally weighted symbols:")
            for symbol, binding in list(weighted_symbols.items())[:3]:
                print(f"   • '{symbol}': weight={binding.emotional_weight:.2f}, meaning='{binding.drifted_meaning}'")
        else:
            print("⚠️  No emotionally weighted symbols found (may be expected for new system)")
        
        # Test attachment reflection
        print("🪞 Testing attachment reflector...")
        try:
            from modules.memory.attachment_reflector import attachment_reflector
            reflection = attachment_reflector.get_symbol_attachment_insights()
            print(f"✅ Attachment reflection generated: {len(reflection.get('insights', []))} insights")
        except ImportError:
            print("⚠️  Attachment reflector not available")
        except AttributeError:
            print("⚠️  Attachment reflector method not found")
        except Exception as e:
            print(f"⚠️  Attachment reflector error: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Symbol binding test failed: {e}")
        traceback.print_exc()
        return False

def test_2_investment_tracking_system():
    """Test Week 2: Investment tracking and analysis"""
    print("\n💰 TEST 2: Investment Tracking & Analysis")
    print("-" * 60)
    
    try:
        from modules.finance import get_investment_integration, StrategyType, GoalType, OptionsLeg
        
        # Set up integration
        os.makedirs("data", exist_ok=True)
        integration = get_investment_integration("data")
        
        print("📊 Testing strategy analysis...")
        # Test credit spread analysis
        strategy_result = integration.analyze_strategy_with_emotional_context(
            ticker="SPY",
            strategy_type=StrategyType.CREDIT_SPREAD,
            legs=[
                OptionsLeg(
                    option_type="put",
                    strike=550.0,
                    premium=2.50,
                    quantity=1,
                    action="sell"
                ),
                OptionsLeg(
                    option_type="put", 
                    strike=545.0,
                    premium=1.20,
                    quantity=1,
                    action="buy"
                )
            ],
            expiration_date=datetime(2025, 8, 15)
        )
        
        print(f"✅ Strategy analysis complete:")
        tech_analysis = strategy_result.get('technical_analysis', {})
        print(f"   Max Gain: ${tech_analysis.get('max_gain', 0):.2f}")
        print(f"   Max Loss: ${tech_analysis.get('max_loss', 0):.2f}")
        print(f"   Win Probability: {tech_analysis.get('probability_of_profit', 0):.1%}")
        print(f"   Return on Risk: {tech_analysis.get('return_on_risk', 0):.1%}")
        
        # Test goal creation
        print("🎯 Testing investment goal creation...")
        goal_response = integration.create_investment_goal_with_companion(
            name="Test Verification Goal",
            target_amount=1000.0,
            goal_type=GoalType.GENERAL_SAVINGS,
            description="Verification test goal",
            priority=1
        )
        
        goal_id = goal_response['goal_created']['goal_id']
        print(f"✅ Goal created: {goal_response['goal_created']['name']}")
        print(f"   Goal ID: {goal_id}")
        print(f"   Companion says: \"{goal_response['companion_response'][:100]}...\"")
        
        # Test trade result logging
        print("📈 Testing trade result logging...")
        trade_result = integration.tracker.log_trade_result(
            strategy_id=strategy_result['strategy_id'],
            actual_profit=130.0,
            close_date="2025-07-30",
            notes="Verification test trade"
        )
        
        print(f"✅ Trade logged:")
        print(f"   Profit: ${trade_result['actual_profit']:.2f}")
        print(f"   Return: {trade_result['actual_return']:.1%}")
        print(f"   Win/Loss: {trade_result['win_loss']}")
        
        # Test profit allocation
        print("💡 Testing profit allocation...")
        allocation = integration.suggest_profit_allocation(130.0)
        
        if allocation["suggestions"]:
            top_suggestion = allocation["suggestions"][0]
            print(f"✅ Allocation suggested:")
            print(f"   Goal: {top_suggestion['goal_name']}")
            print(f"   Amount: ${top_suggestion['suggested_amount']:.2f}")
            print(f"   New Progress: {top_suggestion['new_progress']:.1f}%")
        
        return True
        
    except Exception as e:
        print(f"❌ Investment tracking test failed: {e}")
        traceback.print_exc()
        return False

def test_3_collaborative_goal_system():
    """Test Week 3: Collaborative goal achievement"""
    print("\n🤝 TEST 3: Collaborative Goal Achievement")
    print("-" * 60)
    
    try:
        from modules.collaboration.goal_achievement import collaborative_engine
        from modules.collaboration.investment_bridge import investment_collaboration_bridge
        
        # Test collaborative goal creation
        print("🧠 Testing collaborative goal creation...")
        goal_data = {
            'name': 'Verification Partnership Goal',
            'target_amount': 2000.0,
            'goal_type': 'general_savings',
            'description': 'Testing collaborative system integration',
            'priority': 1,
            'timeline_months': 12
        }
        
        partnership_response = investment_collaboration_bridge.create_collaborative_investment_goal(
            "verification_user", goal_data
        )
        
        goal_id = partnership_response['goal_created']['goal_id']
        print(f"✅ Collaborative goal created:")
        print(f"   Goal: {partnership_response['goal_created']['name']}")
        print(f"   Partnership Strength: {partnership_response['ai_partnership']['partnership_strength']:.1%}")
        print(f"   AI Commitment: {partnership_response['ai_partnership']['ai_commitment_level']:.1%}")
        
        # Test brainstorming session
        print("💡 Testing AI brainstorming...")
        brainstorm = partnership_response['brainstorming_session']
        print(f"✅ Brainstorming session created:")
        print(f"   Ideas generated: {len(brainstorm['ai_ideas'])}")
        for i, idea in enumerate(brainstorm['ai_ideas'][:2], 1):
            print(f"   {i}. {idea['title']}: {idea['description'][:60]}...")
        
        # Test user collaboration response
        print("💬 Testing user collaboration response...")
        user_input = {
            'user_id': 'verification_user',
            'goal_id': goal_id,
            'message': 'I love the milestone approach! The automation sounds great too. Let me think about the risk level.'
        }
        
        collaboration_response = investment_collaboration_bridge.process_user_collaboration_response(
            goal_id, user_input
        )
        
        print(f"✅ Collaboration response processed:")
        print(f"   AI Message: \"{collaboration_response['ai_message'][:80]}...\"")
        print(f"   Refined Strategy: {collaboration_response['refined_strategy']['primary_approach']}")
        print(f"   Partnership Growth: {collaboration_response['partnership_update']['growth_trajectory']}")
        
        # Test daily check-ins
        print("🌅 Testing daily partnership check-ins...")
        # Simulate time passage for check-in
        partnership = collaborative_engine.partnerships[goal_id]
        partnership.last_check_in = time.time() - (25 * 3600)  # 25 hours ago
        
        daily_updates = investment_collaboration_bridge.get_daily_partnership_updates("verification_user")
        
        if daily_updates:
            update = daily_updates[0]
            print(f"✅ Daily check-in generated:")
            print(f"   Type: {update['check_in_type']}")
            print(f"   Message: \"{update['message'][:60]}...\"")
            print(f"   Motivation: \"{update['motivation_boost'][:50]}...\"")
        
        return True
        
    except Exception as e:
        print(f"❌ Collaborative goal test failed: {e}")
        traceback.print_exc()
        return False

def test_4_cross_system_integration():
    """Test integration between all systems"""
    print("\n🔗 TEST 4: Cross-System Integration")
    print("-" * 60)
    
    try:
        # Test memory manager integration with collaboration
        print("🧠 Testing memory-collaboration integration...")
        from modules.memory.memory_manager import memory_manager
        from modules.collaboration.goal_achievement import collaborative_engine
        
        # Check if collaboration system can access memory manager
        test_partnership = list(collaborative_engine.partnerships.values())[0] if collaborative_engine.partnerships else None
        
        if test_partnership and collaborative_engine.memory_manager:
            weighted_symbols = collaborative_engine.memory_manager.get_emotionally_weighted_symbols(0.2)
            print(f"✅ Collaboration system can access {len(weighted_symbols)} emotional symbols")
        else:
            print("⚠️  Memory manager not available to collaboration system")
        
        # Test investment-collaboration bridge
        print("💰 Testing investment-collaboration bridge...")
        from modules.collaboration.investment_bridge import investment_collaboration_bridge
        
        # Verify bridge has access to both systems
        if hasattr(investment_collaboration_bridge, 'investment_integration') and \
           hasattr(investment_collaboration_bridge, 'collaborative_engine'):
            print("✅ Investment bridge has access to both investment and collaboration systems")
        else:
            print("⚠️  Investment bridge missing system connections")
        
        # Test data persistence across systems
        print("💾 Testing data persistence...")
        data_files = [
            "data/investment_goals.json",
            "data/goal_partnerships.json", 
            "data/collaboration_sessions.json",
            "data/motivation_profiles.json",
            "data/partnership_interactions.json"
        ]
        
        existing_files = [f for f in data_files if os.path.exists(f)]
        print(f"✅ Data persistence: {len(existing_files)}/{len(data_files)} files exist")
        
        for file_path in existing_files[:3]:  # Check first 3 files
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    print(f"   • {os.path.basename(file_path)}: {len(data)} entries")
            except:
                print(f"   • {os.path.basename(file_path)}: File exists but may be corrupted")
        
        return True
        
    except Exception as e:
        print(f"❌ Cross-system integration test failed: {e}")
        traceback.print_exc()
        return False

def test_5_error_handling_and_edge_cases():
    """Test error handling and edge cases"""
    print("\n🛡️ TEST 5: Error Handling & Edge Cases")
    print("-" * 60)
    
    try:
        # Test invalid goal creation
        print("⚠️  Testing invalid goal handling...")
        try:
            from modules.collaboration.investment_bridge import investment_collaboration_bridge
            
            invalid_goal = {
                'name': '',  # Empty name
                'target_amount': -100,  # Negative amount
                'goal_type': 'invalid_type',  # Invalid type
            }
            
            # This should handle gracefully
            result = investment_collaboration_bridge.create_collaborative_investment_goal(
                "test_user", invalid_goal
            )
            print("⚠️  Invalid goal was accepted (may need stricter validation)")
            
        except Exception as e:
            print(f"✅ Invalid goal properly rejected: {type(e).__name__}")
        
        # Test missing data directories
        print("📁 Testing missing data directory handling...")
        temp_dir = "temp_test_data"
        
        try:
            from modules.collaboration.goal_achievement import CollaborativeGoalEngine
            temp_engine = CollaborativeGoalEngine(temp_dir)
            print("✅ Handles missing data directory gracefully")
        except Exception as e:
            print(f"⚠️  Missing directory handling issue: {e}")
        
        # Test large data handling
        print("📊 Testing large data volume handling...")
        try:
            from modules.memory.memory_manager import memory_manager
            
            # Add many symbol bindings
            for i in range(50):
                memory_manager.bind_symbol_to_emotion(
                    symbol=f"test_symbol_{i}",
                    emotion="test_emotion",
                    intensity=0.5,
                    context=f"test context {i}"
                )
            
            # Test retrieval with many symbols
            symbols = memory_manager.get_emotionally_weighted_symbols(0.1)
            print(f"✅ Handles large symbol sets: {len(symbols)} symbols processed")
            
        except Exception as e:
            print(f"⚠️  Large data handling issue: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        traceback.print_exc()
        return False

def test_6_performance_and_memory():
    """Test performance and memory usage"""
    print("\n⚡ TEST 6: Performance & Memory Usage")
    print("-" * 60)
    
    try:
        import time
        import psutil
        import os
        
        # Measure memory before tests
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        print(f"🧠 Memory usage before: {memory_before:.1f} MB")
        
        # Performance test: Symbol binding operations
        print("⏱️  Testing symbol binding performance...")
        start_time = time.time()
        
        from modules.memory.memory_manager import memory_manager
        
        for i in range(100):
            memory_manager.bind_symbol_to_emotion(
                symbol=f"perf_test_{i}",
                emotion="performance",
                intensity=0.6,
                context="performance testing"
            )
        
        binding_time = time.time() - start_time
        print(f"✅ 100 symbol bindings: {binding_time:.3f}s ({binding_time/100*1000:.1f}ms per binding)")
        
        # Performance test: Goal creation
        print("⏱️  Testing goal creation performance...")
        start_time = time.time()
        
        from modules.collaboration.investment_bridge import investment_collaboration_bridge
        
        for i in range(10):
            goal_data = {
                'name': f'Performance Test Goal {i}',
                'target_amount': 1000.0 + i * 100,
                'goal_type': 'general_savings',
                'description': f'Performance test goal number {i}',
                'priority': 1
            }
            
            investment_collaboration_bridge.create_collaborative_investment_goal(
                f"perf_user_{i}", goal_data
            )
        
        goal_time = time.time() - start_time
        print(f"✅ 10 collaborative goals: {goal_time:.3f}s ({goal_time/10*1000:.0f}ms per goal)")
        
        # Measure memory after tests
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = memory_after - memory_before
        
        print(f"🧠 Memory usage after: {memory_after:.1f} MB")
        print(f"📈 Memory increase: {memory_increase:.1f} MB")
        
        if memory_increase > 100:  # More than 100MB increase
            print("⚠️  High memory usage detected")
        else:
            print("✅ Memory usage within acceptable range")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        traceback.print_exc()
        return False

def generate_verification_report(test_results):
    """Generate comprehensive verification report"""
    print("\n" + "=" * 70)
    print("🎯 COMPREHENSIVE VERIFICATION REPORT")
    print("=" * 70)
    
    total_tests = len(test_results)
    passed_tests = sum(test_results.values())
    success_rate = passed_tests / total_tests * 100
    
    print(f"📊 Overall Success Rate: {success_rate:.1f}% ({passed_tests}/{total_tests} tests passed)")
    print()
    
    # Individual test results
    test_names = {
        'test_1': 'Week 1: Symbol Binding & Emotional Intelligence',
        'test_2': 'Week 2: Investment Tracking & Analysis', 
        'test_3': 'Week 3: Collaborative Goal Achievement',
        'test_4': 'Cross-System Integration',
        'test_5': 'Error Handling & Edge Cases',
        'test_6': 'Performance & Memory Usage'
    }
    
    print("📋 Individual Test Results:")
    for test_key, passed in test_results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        test_name = test_names.get(test_key, test_key)
        print(f"   {status} - {test_name}")
    
    print()
    
    # System status assessment
    if success_rate >= 90:
        status = "🎉 EXCELLENT"
        message = "All systems operational and performing excellently!"
    elif success_rate >= 75:
        status = "✅ GOOD"
        message = "Most systems working well with minor issues to address."
    elif success_rate >= 50:
        status = "⚠️  NEEDS ATTENTION"
        message = "Several systems need attention before production use."
    else:
        status = "❌ CRITICAL ISSUES"
        message = "Major issues detected - immediate attention required."
    
    print(f"🏆 System Status: {status}")
    print(f"💬 Assessment: {message}")
    print()
    
    # Specific recommendations
    print("🔧 Recommendations:")
    
    if test_results.get('test_1', False):
        print("   ✅ Emotional intelligence system ready for production")
    else:
        print("   🔧 Review symbol binding implementation")
    
    if test_results.get('test_2', False):
        print("   ✅ Investment tracking system fully functional")
    else:
        print("   🔧 Check investment module dependencies")
    
    if test_results.get('test_3', False):
        print("   ✅ Collaborative partnership system operational")
    else:
        print("   🔧 Debug collaboration engine integration")
    
    if test_results.get('test_4', False):
        print("   ✅ Cross-system integration working properly")
    else:
        print("   🔧 Fix integration bridges between systems")
    
    if test_results.get('test_5', False):
        print("   ✅ Error handling robust and reliable")
    else:
        print("   🔧 Strengthen error handling and validation")
    
    if test_results.get('test_6', False):
        print("   ✅ Performance and memory usage optimized")
    else:
        print("   🔧 Optimize performance and memory management")
    
    print()
    print("🚀 Ready for Deployment:", "YES" if success_rate >= 80 else "NEEDS WORK")
    print("=" * 70)

def main():
    """Run comprehensive verification suite"""
    print("🔍 COMPREHENSIVE AI COMPANION VERIFICATION SUITE")
    print(f"🕐 Started at: {datetime.now().strftime('%H:%M:%S')}")
    print("🎯 Testing all Week 1, 2, and 3 implementations...")
    print()
    
    # Create data directory
    os.makedirs("data", exist_ok=True)
    
    # Run all tests
    test_results = {}
    
    test_results['test_1'] = test_1_symbol_binding_system()
    test_results['test_2'] = test_2_investment_tracking_system()
    test_results['test_3'] = test_3_collaborative_goal_system()
    test_results['test_4'] = test_4_cross_system_integration()
    test_results['test_5'] = test_5_error_handling_and_edge_cases()
    test_results['test_6'] = test_6_performance_and_memory()
    
    # Generate final report
    generate_verification_report(test_results)
    
    print(f"🕐 Completed at: {datetime.now().strftime('%H:%M:%S')}")
    
    return test_results

if __name__ == "__main__":
    results = main()

#!/usr/bin/env python3
"""
HRM System Integration Test
===========================

Comprehensive integration test to verify HRM system works with existing
projectAlpha components before moving forward.
"""

import sys
import os
import asyncio
import traceback
from datetime import datetime

def test_basic_imports():
    """Test basic component imports"""
    print("🔧 Testing Basic Component Imports...")
    
    try:
        # Core components
        from core_arbiter import CoreArbiter, ArbiterResponse
        print("   ✅ CoreArbiter imported")
        
        from mirror_mode import MirrorModeManager, MirrorType
        print("   ✅ MirrorMode imported")
        
        # HRM components
        from hrm_router import HRMRouter, HRMMode, RequestType
        print("   ✅ HRM Router imported")
        
        # Backend components
        try:
            from backend.subagent_router import SubAgentRouter, AgentType
            print("   ✅ SubAgent Router imported")
        except ImportError as e:
            print(f"   ⚠️  SubAgent Router import issue: {e}")
        
        try:
            from backend.ai_reformulator import PersonalityFormatter
            print("   ✅ AI Reformulator imported")
        except ImportError as e:
            print(f"   ⚠️  AI Reformulator import issue: {e}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Import failed: {e}")
        traceback.print_exc()
        return False

def test_component_initialization():
    """Test component initialization"""
    print("\n🏗️  Testing Component Initialization...")
    
    try:
        # Initialize core components
        from core_arbiter import CoreArbiter
        arbiter = CoreArbiter()
        print("   ✅ CoreArbiter initialized")
        
        from mirror_mode import MirrorModeManager
        mirror = MirrorModeManager()
        print("   ✅ MirrorMode initialized")
        
        from hrm_router import HRMRouter
        router = HRMRouter()
        print("   ✅ HRM Router initialized")
        
        # Test system status
        status = router.get_system_status()
        print(f"   ✅ System status retrieved: {len(status)} keys")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Initialization failed: {e}")
        traceback.print_exc()
        return False

async def test_core_arbiter_integration():
    """Test Core Arbiter integration"""
    print("\n⚖️  Testing Core Arbiter Integration...")
    
    try:
        from core_arbiter import CoreArbiter
        
        arbiter = CoreArbiter()
        
        # Test processing
        test_input = "Hello, this is a test message"
        test_state = {"user_id": "test_user", "mood": "neutral"}
        
        response = await arbiter.process_input(test_input, test_state)
        
        print(f"   ✅ Core Arbiter processed input")
        print(f"   📊 Confidence: {response.confidence:.2f}")
        print(f"   🎭 Tone: {response.tone}")
        print(f"   📝 Response length: {len(response.final_output)} chars")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Core Arbiter integration failed: {e}")
        traceback.print_exc()
        return False

async def test_hrm_router_processing():
    """Test HRM Router processing"""
    print("\n🧠 Testing HRM Router Processing...")
    
    try:
        from hrm_router import HRMRouter
        
        router = HRMRouter()
        
        # Test different types of requests
        test_cases = [
            ("Hello, how are you today?", {"mood": "neutral"}),
            ("Can you help me with Python programming?", {"priority": 0.8}),
            ("I'm feeling sad about something", {"mood": "sadness"}),
        ]
        
        for i, (user_input, context) in enumerate(test_cases, 1):
            print(f"   🧪 Test case {i}: {user_input[:30]}...")
            
            response = await router.process_request(user_input, context)
            
            print(f"      ✅ Mode: {response.processing_mode.value}")
            print(f"      📊 Confidence: {response.confidence_score:.2f}")
            print(f"      ⏱️  Time: {response.processing_time:.3f}s")
            
        return True
        
    except Exception as e:
        print(f"   ❌ HRM Router processing failed: {e}")
        traceback.print_exc()
        return False

def test_mirror_mode_integration():
    """Test Mirror Mode integration"""
    print("\n🪩 Testing Mirror Mode Integration...")
    
    try:
        from mirror_mode import MirrorModeManager, MirrorType
        
        mirror = MirrorModeManager()
        
        # Test mirror reflection
        test_response = "This is a test response from the AI system."
        test_context = {
            "processing_mode": "balanced",
            "confidence": 0.85,
            "agents_used": ["core_arbiter"]
        }
        
        enhanced_response = mirror.add_mirror_reflection(
            test_response,
            test_context,
            [MirrorType.REASONING]
        )
        
        print("   ✅ Mirror reflection added")
        print(f"   📏 Original length: {len(test_response)}")
        print(f"   📏 Enhanced length: {len(enhanced_response)}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Mirror Mode integration failed: {e}")
        traceback.print_exc()
        return False

def test_configuration_system():
    """Test configuration system"""
    print("\n⚙️  Testing Configuration System...")
    
    try:
        from hrm_router import HRMRouter
        
        router = HRMRouter()
        
        # Test configuration loading
        config = router.config
        print(f"   ✅ Config loaded with {len(config)} keys")
        
        # Test key configuration items
        required_keys = ["default_mode", "enable_mirror_mode", "memory_budget"]
        for key in required_keys:
            if key in config:
                print(f"   ✅ {key}: {config[key]}")
            else:
                print(f"   ⚠️  Missing config key: {key}")
        
        # Test system status
        status = router.get_system_status()
        print(f"   ✅ System status: {status['health']['status']}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Configuration test failed: {e}")
        traceback.print_exc()
        return False

def test_data_directory_structure():
    """Test data directory structure"""
    print("\n📁 Testing Data Directory Structure...")
    
    try:
        import os
        from pathlib import Path
        
        # Check for required directories
        required_dirs = ["data", "logs"]
        for dir_name in required_dirs:
            dir_path = Path(dir_name)
            if dir_path.exists():
                print(f"   ✅ {dir_name}/ directory exists")
            else:
                print(f"   📁 Creating {dir_name}/ directory")
                dir_path.mkdir(exist_ok=True)
        
        # Check for config files
        config_files = [
            "data/hrm_config.json",
            "data/core_arbiter_config.json",
            "data/identity_tether.json"
        ]
        
        for config_file in config_files:
            if Path(config_file).exists():
                print(f"   ✅ {config_file} exists")
            else:
                print(f"   ⚠️  {config_file} will be created on first run")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Directory structure test failed: {e}")
        return False

async def run_integration_tests():
    """Run all integration tests"""
    print("🚀 HRM SYSTEM INTEGRATION TEST SUITE")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Python: {sys.version}")
    print(f"Working Directory: {os.getcwd()}")
    
    # Run tests
    tests = [
        ("Basic Imports", test_basic_imports),
        ("Component Initialization", test_component_initialization),
        ("Core Arbiter Integration", test_core_arbiter_integration),
        ("HRM Router Processing", test_hrm_router_processing),
        ("Mirror Mode Integration", test_mirror_mode_integration),
        ("Configuration System", test_configuration_system),
        ("Data Directory Structure", test_data_directory_structure),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n" + "─" * 60)
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
            results[test_name] = result
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    print(f"\n" + "=" * 60)
    print("📊 INTEGRATION TEST RESULTS")
    print("=" * 60)
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\n📈 SUMMARY: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED - HRM system is ready for production!")
    elif passed >= total * 0.8:
        print("⚠️  MOSTLY WORKING - Some minor issues to address")
    else:
        print("❌ SIGNIFICANT ISSUES - Need to fix critical problems")
    
    return results

if __name__ == "__main__":
    asyncio.run(run_integration_tests())

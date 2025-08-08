#!/usr/bin/env python3
"""
Test script for ProjectAlpha centralized configuration system.
Verifies that settings load correctly and validation works.
"""

import os
import sys
import tempfile
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_configuration_loading():
    """Test configuration loading with various scenarios."""
    print("🧪 Testing ProjectAlpha Configuration System")
    print("=" * 50)
    
    # Test 1: Load with minimal required env vars
    print("\n📋 Test 1: Minimal Configuration")
    os.environ.update({
        'MONGO_ROOT_USERNAME': 'test_user',
        'MONGO_ROOT_PASSWORD': 'test_pass',
        'SERVER_ROLE': 'primary',
        'SERVER_ID': 'test-server-001'
    })
    
    try:
        from config.settings import load_settings, log_configuration_summary
        settings = load_settings()
        print("✅ Minimal configuration loaded successfully")
        
        # Verify defaults are applied
        assert settings.DRIFT_SCALING_FACTOR == 0.35, f"Expected 0.35, got {settings.DRIFT_SCALING_FACTOR}"
        assert settings.MAX_PENALTY_THRESHOLD == 0.85, f"Expected 0.85, got {settings.MAX_PENALTY_THRESHOLD}"
        assert settings.EMOTION_LOOP_ENABLED == True, f"Expected True, got {settings.EMOTION_LOOP_ENABLED}"
        assert settings.AUTOPILOT_ENABLED == True, f"Expected True, got {settings.AUTOPILOT_ENABLED}"
        assert settings.SAFE_MODE_FORCE == False, f"Expected False, got {settings.SAFE_MODE_FORCE}"
        print("✅ Default values applied correctly")
        
        log_configuration_summary(settings)
        
    except Exception as e:
        print(f"❌ Minimal configuration test failed: {e}")
        return False
    
    # Test 2: Override with environment variables
    print("\n📋 Test 2: Environment Variable Overrides")
    os.environ.update({
        'DRIFT_SCALING_FACTOR': '0.7',
        'MAX_PENALTY_THRESHOLD': '0.9',
        'RATE_LIMIT_WINDOW': '30',
        'RATE_LIMIT_MAX': '200',
        'EMOTION_LOOP_ENABLED': 'false',
        'SAFE_MODE_FORCE': 'true',
        'LOG_LEVEL': 'DEBUG'
    })
    
    try:
        # Clear cached settings
        if hasattr(load_settings, '_settings'):
            delattr(load_settings, '_settings')
        
        settings = load_settings()
        print("✅ Environment override configuration loaded successfully")
        
        # Verify overrides are applied
        assert settings.DRIFT_SCALING_FACTOR == 0.7, f"Expected 0.7, got {settings.DRIFT_SCALING_FACTOR}"
        assert settings.MAX_PENALTY_THRESHOLD == 0.9, f"Expected 0.9, got {settings.MAX_PENALTY_THRESHOLD}"
        assert settings.RATE_LIMIT_WINDOW == 30, f"Expected 30, got {settings.RATE_LIMIT_WINDOW}"
        assert settings.RATE_LIMIT_MAX == 200, f"Expected 200, got {settings.RATE_LIMIT_MAX}"
        assert settings.EMOTION_LOOP_ENABLED == False, f"Expected False, got {settings.EMOTION_LOOP_ENABLED}"
        assert settings.SAFE_MODE_FORCE == True, f"Expected True, got {settings.SAFE_MODE_FORCE}"
        assert settings.LOG_LEVEL == 'DEBUG', f"Expected DEBUG, got {settings.LOG_LEVEL}"
        print("✅ Environment variable overrides applied correctly")
        
    except Exception as e:
        print(f"❌ Environment override test failed: {e}")
        return False
    
    # Test 3: Validation errors
    print("\n📋 Test 3: Validation Error Handling")
    
    # Test invalid drift scaling factor
    os.environ['DRIFT_SCALING_FACTOR'] = '1.5'  # Invalid: > 1.0
    try:
        settings = load_settings()
        print("❌ Should have failed with invalid DRIFT_SCALING_FACTOR")
        return False
    except ValueError as e:
        if "DRIFT_SCALING_FACTOR" in str(e):
            print("✅ Correctly caught invalid DRIFT_SCALING_FACTOR")
        else:
            print(f"❌ Unexpected validation error: {e}")
            return False
    except Exception as e:
        print(f"❌ Unexpected error type: {e}")
        return False
    
    # Reset to valid value
    os.environ['DRIFT_SCALING_FACTOR'] = '0.35'
    
    # Test invalid server role
    os.environ['SERVER_ROLE'] = 'invalid_role'
    try:
        settings = load_settings()
        print("❌ Should have failed with invalid SERVER_ROLE")
        return False
    except ValueError as e:
        if "SERVER_ROLE" in str(e):
            print("✅ Correctly caught invalid SERVER_ROLE")
        else:
            print(f"❌ Unexpected validation error: {e}")
            return False
    except Exception as e:
        print(f"❌ Unexpected error type: {e}")
        return False
    
    # Reset to valid value
    os.environ['SERVER_ROLE'] = 'primary'
    
    # Test 4: Feature flag helpers
    print("\n📋 Test 4: Feature Flag Helpers")
    try:
        # Import fresh to get updated settings
        import importlib
        import config.settings
        importlib.reload(config.settings)
        
        from config.settings import is_feature_enabled, is_safe_mode_active, get_rate_limit_config
        
        # These should work with current env vars
        safe_mode_status = is_safe_mode_active()
        print(f"   Safe mode status: {safe_mode_status}")
        assert safe_mode_status == True, f"Safe mode should be active, got {safe_mode_status}"
        
        emotion_loop_status = is_feature_enabled('emotion_loop')
        print(f"   Emotion loop status: {emotion_loop_status}")
        assert emotion_loop_status == False, f"Emotion loop should be disabled, got {emotion_loop_status}"
        
        safe_mode_feature = is_feature_enabled('safe_mode')
        print(f"   Safe mode feature: {safe_mode_feature}")
        assert safe_mode_feature == True, f"Safe mode feature should be enabled, got {safe_mode_feature}"
        
        rate_config = get_rate_limit_config()
        assert rate_config['window'] == 30, f"Expected window 30, got {rate_config['window']}"
        assert rate_config['max_requests'] == 200, f"Expected max 200, got {rate_config['max_requests']}"
        
        print("✅ Feature flag helpers working correctly")
        
    except Exception as e:
        print(f"❌ Feature flag helper test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 5: Missing required fields
    print("\n📋 Test 5: Missing Required Fields")
    del os.environ['MONGO_ROOT_USERNAME']
    try:
        settings = load_settings()
        print("❌ Should have failed with missing MONGO_ROOT_USERNAME")
        return False
    except ValueError as e:
        if "MONGO_ROOT_USERNAME" in str(e):
            print("✅ Correctly caught missing required field")
        else:
            print(f"❌ Unexpected validation error: {e}")
            return False
    except Exception as e:
        print(f"❌ Unexpected error type: {e}")
        return False
    
    print("\n🎉 All configuration tests passed!")
    return True

def test_schema_validation():
    """Test that .env.schema.json contains required properties."""
    print("\n📋 Testing Schema Validation")
    
    try:
        import json
        schema_path = Path(".env.schema.json")
        if not schema_path.exists():
            print("❌ .env.schema.json not found")
            return False
        
        with open(schema_path) as f:
            schema = json.load(f)
        
        required_props = [
            'DRIFT_SCALING_FACTOR',
            'MAX_PENALTY_THRESHOLD',
            'RATE_LIMIT_WINDOW',
            'RATE_LIMIT_MAX',
            'EMOTION_LOOP_ENABLED',
            'AUTOPILOT_ENABLED',
            'SAFE_MODE_FORCE'
        ]
        
        for prop in required_props:
            if prop not in schema.get('properties', {}):
                print(f"❌ Missing property in schema: {prop}")
                return False
            print(f"✅ Found {prop} in schema")
        
        # Check that DRIFT_SCALING_FACTOR and MAX_PENALTY_THRESHOLD have correct defaults
        drift_prop = schema['properties']['DRIFT_SCALING_FACTOR']
        penalty_prop = schema['properties']['MAX_PENALTY_THRESHOLD']
        
        assert drift_prop.get('default') == 0.35, f"DRIFT_SCALING_FACTOR default should be 0.35, got {drift_prop.get('default')}"
        assert penalty_prop.get('default') == 0.85, f"MAX_PENALTY_THRESHOLD default should be 0.85, got {penalty_prop.get('default')}"
        
        print("✅ Schema validation passed")
        return True
        
    except Exception as e:
        print(f"❌ Schema validation failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 ProjectAlpha Configuration System Tests")
    print("=" * 60)
    
    # Store original env to restore later
    original_env = dict(os.environ)
    
    try:
        # Run tests
        config_test = test_configuration_loading()
        schema_test = test_schema_validation()
        
        # Summary
        print("\n" + "=" * 60)
        print("📊 TEST SUMMARY")
        print("=" * 60)
        
        tests = [
            ("Configuration Loading", config_test),
            ("Schema Validation", schema_test),
        ]
        
        passed = sum(1 for _, result in tests if result)
        total = len(tests)
        
        for test_name, result in tests:
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{status} {test_name}")
        
        print(f"\n📈 SUMMARY: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
        
        if passed == total:
            print("🎉 ALL TESTS PASSED - Configuration system is ready!")
        else:
            print("❌ SOME TESTS FAILED - Please fix issues before proceeding")
            
        return passed == total
        
    finally:
        # Restore original environment
        os.environ.clear()
        os.environ.update(original_env)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

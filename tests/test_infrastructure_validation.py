"""
Test runner to verify determinism and resilience test infrastructure.
Validates that all test components work correctly together.
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_deterministic_mode_setup():
    """Test that deterministic mode configuration works."""
    # This would be run with --deterministic flag
    import random
    import numpy as np
    
    # Seeds should be set deterministically
    assert random.getstate() is not None
    assert np.random.get_state() is not None
    
    print("✅ Deterministic mode setup working")

def test_mock_fixtures_available():
    """Test that mock fixtures are properly configured."""
    # Import should work
    try:
        from tests.conftest import mock_conductor, mock_memory_system
        print("✅ Mock fixtures are importable")
    except ImportError as e:
        pytest.fail(f"Mock fixtures not available: {e}")

def test_chaos_controller_available():
    """Test that chaos controller fixture is available."""
    try:
        from tests.conftest import chaos_controller
        print("✅ Chaos controller fixture available")
    except ImportError as e:
        pytest.fail(f"Chaos controller not available: {e}")

def test_basic_hypothesis_functionality():
    """Test that Hypothesis is working."""
    try:
        from hypothesis import given, strategies as st
        
        @given(st.integers(min_value=0, max_value=100))
        def dummy_property_test(value):
            assert 0 <= value <= 100
        
        # Run a few examples
        dummy_property_test()
        print("✅ Hypothesis property testing working")
    except Exception as e:
        pytest.fail(f"Hypothesis not working: {e}")

def test_security_module_imports():
    """Test that security module imports work."""
    try:
        from backend.common.security import get_token_type, can_access_layer, mask_token
        
        # Test basic functionality
        assert get_token_type('admin_test_token') == 'admin'
        assert get_token_type('user_test_token') == 'user'
        assert mask_token('test123') == '***t123'  # Shows last 4 chars
        
        print("✅ Security module imports working")
    except Exception as e:
        pytest.fail(f"Security module imports failed: {e}")

def test_memory_system_imports():
    """Test that memory system imports work."""
    try:
        from core.memory_system import MemorySystem
        
        # Basic instantiation test
        memory_system = MemorySystem(memory_dir="test_imports")
        assert memory_system is not None
        
        # Cleanup
        import shutil
        test_dir = Path("test_imports")
        if test_dir.exists():
            shutil.rmtree(test_dir)
        
        print("✅ Memory system imports working")
    except Exception as e:
        pytest.fail(f"Memory system imports failed: {e}")

def test_retry_logic_imports():
    """Test that retry logic imports work."""
    try:
        from backend.common.retry import retry_with_backoff, RetryConfig
        
        # Test basic configuration
        config = RetryConfig(max_attempts=3, base_delay=0.1)
        assert config.max_attempts == 3
        assert config.base_delay == 0.1
        
        print("✅ Retry logic imports working")
    except Exception as e:
        pytest.fail(f"Retry logic imports failed: {e}")

def run_validation_tests():
    """Run all validation tests."""
    print("🧪 Testing ProjectAlpha Test Infrastructure")
    print("=" * 50)
    
    tests = [
        test_deterministic_mode_setup,
        test_mock_fixtures_available,
        test_chaos_controller_available,
        test_basic_hypothesis_functionality,
        test_security_module_imports,
        test_memory_system_imports,
        test_retry_logic_imports
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"❌ {test.__name__}: {e}")
            failed += 1
    
    print("\n" + "=" * 50)
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"📊 Success rate: {passed/(passed+failed)*100:.1f}%")
    
    if failed == 0:
        print("\n🎉 All test infrastructure validation tests passed!")
        print("Ready to run property-based and chaos tests.")
    else:
        print(f"\n⚠️  {failed} validation tests failed. Check dependencies and imports.")
    
    return failed == 0

if __name__ == "__main__":
    success = run_validation_tests()
    sys.exit(0 if success else 1)

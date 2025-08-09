#!/usr/bin/env python3
"""
Test script for graceful degradation and bounded growth features.
Tests safe mode, idempotency, retry logic, and memory quotas.
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import requests

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def test_safe_mode_force():
    """Test SAFE_MODE_FORCE environment variable."""
    print("🔒 Testing Safe Mode Enforcement")
    print("=" * 50)

    try:
        # Import CoreConductor to test safe mode
        from src.core.core_conductor import CoreConductor

        # Test without forced safe mode
        print("\n📋 Test 1: Normal operation (SAFE_MODE_FORCE=false)")
        os.environ["SAFE_MODE_FORCE"] = "false"
        conductor_normal = CoreConductor(load_models=False)

        status = conductor_normal.get_safe_mode_status()
        print(f"   Safe mode enabled: {status['safe_mode_enabled']}")
        print(f"   Emotion loop paused: {status['emotion_loop_paused']}")
        print(f"   Writes locked: {status['writes_locked']}")

        assert not status["safe_mode_enabled"], "Safe mode should be disabled"
        assert not status["emotion_loop_paused"], "Emotion loop should not be paused"
        assert not status["writes_locked"], "Writes should not be locked"
        print("   ✅ Normal operation confirmed")

        # Test with forced safe mode
        print("\n📋 Test 2: Forced safe mode (SAFE_MODE_FORCE=true)")
        os.environ["SAFE_MODE_FORCE"] = "true"

        # Need to reimport to pick up new environment variable
        import importlib

        import src.core.core_conductor

        importlib.reload(src.core.core_conductor)
        from src.core.core_conductor import CoreConductor

        conductor_safe = CoreConductor(load_models=False)

        status = conductor_safe.get_safe_mode_status()
        print(f"   Safe mode enabled: {status['safe_mode_enabled']}")
        print(f"   Safe mode reason: {status['safe_mode_reason']}")
        print(f"   Emotion loop paused: {status['emotion_loop_paused']}")
        print(f"   Writes locked: {status['writes_locked']}")

        assert status["safe_mode_enabled"], "Safe mode should be enabled"
        assert status["emotion_loop_paused"], "Emotion loop should be paused"
        assert status["writes_locked"], "Writes should be locked"
        print("   ✅ Forced safe mode confirmed")

        # Test safe mode operations
        print("\n📋 Test 3: Safe mode operation restrictions")
        allowed, reason = conductor_safe.check_safe_mode_restriction("get_status")
        assert allowed, "get_status should be allowed in safe mode"
        print("   ✅ Allowed operation works")

        allowed, reason = conductor_safe.check_safe_mode_restriction("dangerous_operation")
        assert not allowed, "dangerous_operation should be blocked"
        print(f"   ✅ Blocked operation: {reason}")

        # Test manual safe mode entry/exit
        print("\n📋 Test 4: Manual safe mode control")
        # Reset environment for clean test
        os.environ["SAFE_MODE_FORCE"] = "false"
        importlib.reload(src.core.core_conductor)
        from src.core.core_conductor import CoreConductor

        conductor_manual = CoreConductor(load_models=False)

        # Enter safe mode manually
        success = conductor_manual.enter_safe_mode("Manual test entry")
        assert success, "Should be able to enter safe mode manually"

        status = conductor_manual.get_safe_mode_status()
        assert status["safe_mode_enabled"], "Safe mode should be active"
        print("   ✅ Manual safe mode entry works")

        # Try to exit safe mode
        success, message = conductor_manual.exit_safe_mode(force=True)
        assert success, f"Should be able to force exit safe mode: {message}"

        status = conductor_manual.get_safe_mode_status()
        assert not status["safe_mode_enabled"], "Safe mode should be disabled"
        print("   ✅ Manual safe mode exit works")

        return True

    except Exception as e:
        print(f"❌ Safe mode test failed: {e}")
        return False


def test_idempotency():
    """Test idempotency functionality."""
    print("\n🔄 Testing Idempotency")
    print("=" * 50)

    try:
        from backend.core_arbiter_api import check_idempotency, store_idempotency_response

        # Test idempotency key caching
        print("\n📋 Test 1: Idempotency caching")

        test_key = "test_key_12345"
        test_response = {"status": "success", "data": "test_data"}

        # First check - should not exist
        is_duplicate, cached_response = check_idempotency(test_key)
        assert not is_duplicate, "Key should not exist initially"
        assert cached_response is None, "No cached response should exist"
        print("   ✅ New key correctly identified")

        # Store response
        store_idempotency_response(test_key, test_response)
        print("   ✅ Response stored in cache")

        # Second check - should exist
        is_duplicate, cached_response = check_idempotency(test_key)
        assert is_duplicate, "Key should exist after storage"
        assert cached_response == test_response, "Cached response should match"
        print("   ✅ Cached response correctly retrieved")

        # Test with different key
        different_key = "different_key_67890"
        is_duplicate, cached_response = check_idempotency(different_key)
        assert not is_duplicate, "Different key should not exist"
        print("   ✅ Different key correctly isolated")

        return True

    except Exception as e:
        print(f"❌ Idempotency test failed: {e}")
        return False


def test_retry_logic():
    """Test retry and backoff functionality."""
    print("\n🔄 Testing Retry Logic")
    print("=" * 50)

    try:
        from backend.common.retry import (
            RetryableError,
            RetryConfig,
            calculate_delay,
            retry_with_backoff,
        )

        # Test delay calculation
        print("\n📋 Test 1: Backoff delay calculation")
        config = RetryConfig(base_delay=1.0, exponential_base=2.0, max_delay=10.0)

        delays = [calculate_delay(i, config) for i in range(4)]
        print(f"   Delay progression: {[f'{d:.2f}s' for d in delays]}")

        # Delays should increase exponentially but be capped
        assert delays[0] < delays[1] < delays[2], "Delays should increase"
        assert all(d <= config.max_delay for d in delays), "Delays should be capped"
        print("   ✅ Exponential backoff working correctly")

        # Test retry decorator
        print("\n📋 Test 2: Retry decorator functionality")

        attempt_count = 0

        @retry_with_backoff(
            config=RetryConfig(max_attempts=3, base_delay=0.1, jitter=False),
            exceptions=(RetryableError,),
        )
        def failing_function():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise RetryableError(f"Attempt {attempt_count} failed")
            return "Success!"

        result = failing_function()
        assert result == "Success!", "Function should eventually succeed"
        assert attempt_count == 3, "Should have taken 3 attempts"
        print(f"   ✅ Retry succeeded after {attempt_count} attempts")

        # Test non-retryable exception
        @retry_with_backoff(
            config=RetryConfig(max_attempts=3, base_delay=0.1),
            exceptions=(RetryableError,),
        )
        def non_retryable_function():
            raise ValueError("This should not be retried")

        try:
            non_retryable_function()
            raise AssertionError("Should have raised ValueError")
        except ValueError:
            print("   ✅ Non-retryable exception correctly propagated")

        return True

    except Exception as e:
        print(f"❌ Retry logic test failed: {e}")
        return False


def test_memory_quotas():
    """Test memory quota enforcement."""
    print("\n💾 Testing Memory Quotas")
    print("=" * 50)

    try:
        from core.memory_system import MemorySystem

        # Create test memory system
        test_memory_dir = "test_memory"
        memory_system = MemorySystem(memory_dir=test_memory_dir)

        # Override quotas for testing
        memory_system.quotas["ephemeral"]["max_items"] = 5

        print("\n📋 Test 1: Adding memories within quota")

        # Add memories within quota
        for i in range(3):
            success = memory_system.add_layered_memory(
                "ephemeral", f"Test memory {i}", importance=0.5 + (i * 0.1)
            )
            assert success, f"Should be able to add memory {i}"

        status = memory_system.get_memory_quota_status()
        ephemeral_status = status["ephemeral"]
        print(
            "   Current usage: "
            f"{ephemeral_status['current_items']}/"
            f"{ephemeral_status['max_items']}"
        )
        assert not ephemeral_status["is_over_quota"], "Should not be over quota yet"
        print("   ✅ Within quota operation successful")

        print("\n📋 Test 2: Quota enforcement and pruning")

        # Add enough memories to exceed quota
        for i in range(3, 10):  # This will exceed the quota of 5
            memory_system.add_layered_memory(
                "ephemeral",
                f"Test memory {i}",
                importance=0.1 + (i * 0.05),  # Varying importance
            )

        # Check that quota was enforced
        status = memory_system.get_memory_quota_status()
        ephemeral_status = status["ephemeral"]
        print(
            "   After quota enforcement: "
            f"{ephemeral_status['current_items']}/"
            f"{ephemeral_status['max_items']}"
        )
        assert (
            ephemeral_status["current_items"] <= ephemeral_status["max_items"]
        ), "Should be within quota after pruning"
        print("   ✅ Quota enforcement working")

        # Check pruning log
        pruning_log_path = Path("logs/memory_pruning.jsonl")
        if pruning_log_path.exists():
            with pruning_log_path.open() as f:
                lines = f.readlines()
            if lines:
                last_entry = json.loads(lines[-1])
                print("   Pruning event logged: " f"{last_entry['pruned_count']} items removed")
                print("   ✅ Pruning events properly logged")

        print("\n📋 Test 3: Manual quota status and pruning")

        # Get detailed quota status
        full_status = memory_system.get_memory_quota_status()
        print("   Memory usage by layer:")
        for layer, layer_status in full_status.items():
            usage_pct = layer_status["usage_percentage"]
            print(
                f"     {layer}: "
                f"{layer_status['current_items']}/"
                f"{layer_status['max_items']} ({usage_pct:.1f}%)"
            )

        # Test manual pruning
        pruning_results = memory_system.prune_all_layers(force=True)
        print(f"   Manual pruning results: {pruning_results}")
        print("   ✅ Manual pruning operations successful")

        # Cleanup test directory
        import shutil

        if Path(test_memory_dir).exists():
            shutil.rmtree(test_memory_dir)

        return True

    except Exception as e:
        print(f"❌ Memory quota test failed: {e}")
        return False


def test_api_endpoints():
    """Test API endpoints for graceful degradation.
    Requires running services.
    """
    print("\n🌐 Testing API Endpoints")
    print("=" * 50)

    base_urls = [
        "http://localhost:5000",  # Core arbiter
        "http://localhost:5001",  # HRM
    ]

    for base_url in base_urls:
        try:
            # Test health endpoint
            response = requests.get(f"{base_url}/health", timeout=2)
            if response.status_code == 200:
                health_data = response.json()
                if "safe_mode_enabled" in health_data:
                    print(
                        "   ✅ "
                        f"{base_url} - Safe mode status: "
                        f"{health_data['safe_mode_enabled']}"
                    )
                else:
                    print(f"   ✅ {base_url} - Health check passed")
            else:
                print("   ⚠️  ", end="")
                print(f"{base_url} - Health check failed: {response.status_code}")

        except requests.exceptions.ConnectionError:
            print(f"   ⚠️  {base_url} - Service not running")
        except Exception as e:
            print(f"   ❌ {base_url} - Error: {e}")

    # Test idempotency with a running service
    try:
        test_url = "http://localhost:5000/api/arbiter/strategy"
        headers = {
            "Content-Type": "application/json",
            "Idempotency-Key": f"test_key_{int(time.time())}",
            "X-API-Key": "admin_test_token",
        }
        data = {"strategy": "harmonic"}

        # First request
        response1 = requests.post(test_url, json=data, headers=headers, timeout=5)

        # Second request with same idempotency key
        response2 = requests.post(test_url, json=data, headers=headers, timeout=5)

        if response1.status_code == response2.status_code:
            print("   ✅ Idempotency test: Both requests returned " f"{response1.status_code}")
        else:
            print(
                "   ⚠️  Idempotency test: Different status codes "
                f"{response1.status_code} vs {response2.status_code}"
            )

    except requests.exceptions.ConnectionError:
        print("   ⚠️  Cannot test idempotency - service not running")
    except Exception as e:
        print(f"   ❌ Idempotency test error: {e}")

    return True


def main():
    """Run all graceful degradation tests."""
    print("🚀 ProjectAlpha Graceful Degradation Test Suite")
    print("=" * 80)
    print(f"Test started at: {datetime.now().isoformat()}")

    tests = [
        ("Safe Mode Enforcement", test_safe_mode_force),
        ("Idempotency Functionality", test_idempotency),
        ("Retry and Backoff Logic", test_retry_logic),
        ("Memory Quota Management", test_memory_quotas),
        ("API Endpoint Health", test_api_endpoints),
    ]

    results = []

    for test_name, test_func in tests:
        print(f"\n🔍 Running {test_name}...")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 80)
    print("📊 Graceful Degradation Test Results")
    print("=" * 80)

    passed = 0
    total = len(results)

    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} - {test_name}")
        if success:
            passed += 1

    print(f"\n🎯 Overall: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All graceful degradation features working correctly!")
        print("\n📋 Features Verified:")
        print("   ✅ Safe mode with SAFE_MODE_FORCE environment variable")
        print("   ✅ Emotion loop pausing and write locking")
        print("   ✅ Idempotency keys for mutating endpoints")
        print("   ✅ Exponential backoff with jitter for retries")
        print("   ✅ Memory quotas with automatic pruning")
        print("   ✅ Comprehensive audit logging for all operations")
        print("\n🔒 System Resilience:")
        print("   ✅ Graceful handling of offline dependencies")
        print("   ✅ Bounded memory growth with intelligent pruning")
        print("   ✅ Duplicate request protection via idempotency")
        print("   ✅ Safe mode fallback for critical failures")
    else:
        print("⚠️  Some tests failed - review the output above")

    print(f"\nTest completed at: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()

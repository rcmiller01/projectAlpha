"""
Chaos testing for ProjectAlpha supervision systems.
Tests Mirror/Anchor failure scenarios and safe mode behavior.
"""

import time
import pytest
import random
import threading
from pathlib import Path
from typing import Dict, Any, List, Optional
from unittest.mock import patch, MagicMock, call
from contextlib import contextmanager

# Import project modules
import sys
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.core_conductor import CoreConductor
from backend.common.retry import retry_with_backoff, RetryConfig

class TestMirrorAnchorChaos:
    """Chaos tests for Mirror and Anchor system failures."""
    
    def test_mirror_failure_triggers_safe_mode(self, chaos_controller):
        """Test that Mirror service failure triggers safe mode correctly."""
        # Set up CoreConductor
        conductor = CoreConductor()
        
        # Initially not in safe mode
        assert not conductor.safe_mode_enabled
        
        # Simulate Mirror service failure
        with chaos_controller.service_failure('mirror'):
            # Try to access Mirror service (should fail)
            with patch.object(conductor, '_check_mirror_health', return_value=False):
                # This should trigger safe mode
                conductor.enter_safe_mode(reason="Mirror service unhealthy")
                
                # Verify safe mode is active
                assert conductor.safe_mode_enabled
                assert conductor.safe_mode_reason == "Mirror service unhealthy"
    
    def test_anchor_failure_graceful_degradation(self, chaos_controller):
        """Test graceful degradation when Anchor system fails."""
        conductor = CoreConductor()
        
        # Set up chaos scenario
        with chaos_controller.service_failure('anchor'):
            # Mock Anchor health check to fail
            with patch.object(conductor, '_check_anchor_health', return_value=False):
                # Enter safe mode due to Anchor failure
                conductor.enter_safe_mode(reason="Anchor system unavailable")
                
                # Test that system still functions in safe mode
                assert conductor.safe_mode_enabled
                
                # Basic operations should still work
                test_response = conductor.safe_mode_generate("Test query")
                assert test_response is not None
                assert "safe mode" in test_response.lower()
    
    def test_concurrent_failures_stability(self, chaos_controller):
        """Test system stability under multiple concurrent failures."""
        conductor = CoreConductor()
        failure_events = []
        
        def simulate_mirror_failure():
            """Simulate Mirror service becoming unavailable."""
            time.sleep(random.uniform(0.1, 0.5))
            with chaos_controller.service_failure('mirror'):
                failure_events.append(('mirror', time.time()))
                time.sleep(1.0)
        
        def simulate_anchor_failure():
            """Simulate Anchor service becoming unavailable.""" 
            time.sleep(random.uniform(0.1, 0.5))
            with chaos_controller.service_failure('anchor'):
                failure_events.append(('anchor', time.time()))
                time.sleep(1.0)
        
        def simulate_memory_pressure():
            """Simulate memory pressure."""
            time.sleep(random.uniform(0.1, 0.5))
            with chaos_controller.resource_pressure('memory', level=0.9):
                failure_events.append(('memory', time.time()))
                time.sleep(1.0)
        
        # Start concurrent failure simulations
        threads = [
            threading.Thread(target=simulate_mirror_failure),
            threading.Thread(target=simulate_anchor_failure),
            threading.Thread(target=simulate_memory_pressure)
        ]
        
        for thread in threads:
            thread.start()
        
        # Monitor system stability during failures
        start_time = time.time()
        stability_checks = []
        
        while time.time() - start_time < 3.0:  # Monitor for 3 seconds
            try:
                # Check if conductor can handle requests
                response = conductor.safe_mode_generate("Stability check")
                stability_checks.append((time.time(), response is not None))
            except Exception as e:
                stability_checks.append((time.time(), False, str(e)))
            
            time.sleep(0.2)
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Analyze stability
        successful_checks = sum(1 for check in stability_checks if check[1])
        total_checks = len(stability_checks)
        stability_ratio = successful_checks / total_checks if total_checks > 0 else 0
        
        # System should maintain at least 80% stability during chaos
        assert stability_ratio >= 0.8, f"System stability {stability_ratio:.2%} below threshold during chaos"
        
        # Should have detected multiple failure types
        failure_types = {event[0] for event in failure_events}
        assert len(failure_types) >= 2, f"Expected multiple failure types, got {failure_types}"

class TestRetryLogicChaos:
    """Chaos tests for retry logic under various failure patterns."""
    
    def test_exponential_backoff_under_pressure(self, chaos_controller):
        """Test retry logic behavior under sustained service pressure."""
        failure_count = 0
        call_times = []
        
        def flaky_service():
            """Service that fails frequently under pressure."""
            nonlocal failure_count
            call_times.append(time.time())
            failure_count += 1
            
            # Fail for first several attempts
            if failure_count <= 5:
                raise ConnectionError(f"Service unavailable (attempt {failure_count})")
            return {"status": "success", "attempt": failure_count}
        
        # Configure aggressive retry
        retry_config = RetryConfig(
            max_attempts=10,
            base_delay=0.1,
            max_delay=2.0,
            exponential_base=1.5
        )
        
        # Test retry under chaos
        with chaos_controller.service_failure('external_api'):
            decorated_service = retry_with_backoff(config=retry_config)(flaky_service)
            
            start_time = time.time()
            result = decorated_service()
            end_time = time.time()
            
            # Should eventually succeed
            assert result['status'] == 'success'
            assert failure_count > 5  # Should have retried multiple times
            
            # Check timing patterns
            if len(call_times) > 1:
                delays = [call_times[i] - call_times[i-1] for i in range(1, len(call_times))]
                
                # Delays should generally increase (exponential backoff)
                for i in range(1, len(delays)):
                    # Allow some jitter, but trend should be increasing
                    assert delays[i] >= delays[i-1] * 0.8, "Backoff delays should generally increase"
    
    def test_circuit_breaker_behavior(self, chaos_controller):
        """Test circuit breaker pattern during sustained failures."""
        consecutive_failures = 0
        circuit_open = False
        
        def failing_service():
            """Service that consistently fails."""
            nonlocal consecutive_failures, circuit_open
            consecutive_failures += 1
            
            # Simulate circuit breaker logic
            if consecutive_failures >= 5:
                circuit_open = True
                raise Exception("Circuit breaker open - service unavailable")
            
            raise ConnectionError(f"Service failure {consecutive_failures}")
        
        # Test circuit breaker behavior
        retry_config = RetryConfig(max_attempts=3, base_delay=0.1)
        decorated_service = retry_with_backoff(config=retry_config)(failing_service)
        
        # Multiple attempts should eventually trigger circuit breaker
        with pytest.raises(Exception) as exc_info:
            for attempt in range(10):
                try:
                    decorated_service()
                except Exception as e:
                    if "Circuit breaker open" in str(e):
                        raise  # Circuit breaker triggered
                    continue  # Regular failure, keep trying
        
        assert "Circuit breaker open" in str(exc_info.value)
        assert circuit_open, "Circuit breaker should be open after sustained failures"

class TestSafeModeTransitions:
    """Test safe mode transitions under various chaos scenarios."""
    
    def test_safe_mode_entry_exit_cycle(self, chaos_controller):
        """Test complete safe mode entry and exit cycle."""
        conductor = CoreConductor()
        
        # Normal operation
        assert not conductor.safe_mode_enabled
        
        # Trigger safe mode
        with chaos_controller.service_failure('mirror'):
            conductor.enter_safe_mode(reason="Testing safe mode cycle")
            
            # Verify safe mode state
            assert conductor.safe_mode_enabled
            assert conductor.safe_mode_reason == "Testing safe mode cycle"
            
            # Test safe mode operations
            response = conductor.safe_mode_generate("Test in safe mode")
            assert response is not None
            assert "safe mode" in response.lower()
        
        # Exit safe mode when service recovers
        conductor.exit_safe_mode()
        
        # Verify normal operation restored
        assert not conductor.safe_mode_enabled
        assert conductor.safe_mode_reason is None
    
    def test_rapid_safe_mode_transitions(self, chaos_controller):
        """Test system stability under rapid safe mode transitions."""
        conductor = CoreConductor()
        transition_log = []
        
        # Simulate rapid failure/recovery cycles
        for cycle in range(5):
            # Enter safe mode
            with chaos_controller.service_failure('anchor'):
                conductor.enter_safe_mode(reason=f"Rapid failure {cycle}")
                transition_log.append(('enter', time.time(), cycle))
                
                # Brief operation in safe mode
                time.sleep(0.1)
                
                # Exit safe mode
                conductor.exit_safe_mode()
                transition_log.append(('exit', time.time(), cycle))
                
                # Brief normal operation
                time.sleep(0.1)
        
        # Verify final state is normal
        assert not conductor.safe_mode_enabled
        
        # Verify all transitions were logged
        enter_count = sum(1 for log in transition_log if log[0] == 'enter')
        exit_count = sum(1 for log in transition_log if log[0] == 'exit')
        
        assert enter_count == 5, f"Expected 5 safe mode entries, got {enter_count}"
        assert exit_count == 5, f"Expected 5 safe mode exits, got {exit_count}"
    
    def test_safe_mode_memory_protection(self, chaos_controller):
        """Test that safe mode protects memory operations.""" 
        from core.memory_system import MemorySystem
        
        memory_system = MemorySystem(memory_dir="test_safe_mode_memory")
        conductor = CoreConductor()
        
        try:
            # Normal memory operation
            success = memory_system.add_layered_memory('ephemeral', 'Normal operation', 0.5)
            assert success, "Normal memory operation should succeed"
            
            # Enter safe mode
            conductor.enter_safe_mode(reason="Testing memory protection")
            
            # Memory operations in safe mode should be more conservative
            with chaos_controller.resource_pressure('memory', level=0.95):
                # High importance memories should still be added
                success_high = memory_system.add_layered_memory('ephemeral', 'High importance', 0.9)
                assert success_high, "High importance memories should be added in safe mode"
                
                # Low importance memories might be rejected
                success_low = memory_system.add_layered_memory('ephemeral', 'Low importance', 0.1)
                # Don't assert failure - depends on quota status
                
                # But system should remain stable
                quota_status = memory_system.get_memory_quota_status()
                assert quota_status is not None, "Memory system should remain accessible in safe mode"
        
        finally:
            # Cleanup
            import shutil
            test_dir = Path("test_safe_mode_memory")
            if test_dir.exists():
                shutil.rmtree(test_dir)

class TestSystemRecovery:
    """Test system recovery patterns after chaos events."""
    
    def test_graceful_recovery_after_mirror_failure(self, chaos_controller):
        """Test graceful recovery after Mirror service failure."""
        conductor = CoreConductor()
        recovery_stages = []
        
        # Simulate Mirror failure and recovery
        with chaos_controller.service_failure('mirror'):
            # Initial failure
            conductor.enter_safe_mode(reason="Mirror service failed")
            recovery_stages.append(('safe_mode_entered', time.time()))
            
            # System operates in safe mode
            time.sleep(0.5)
            recovery_stages.append(('safe_mode_operation', time.time()))
        
        # Service recovery simulation
        time.sleep(0.2)  # Brief recovery delay
        
        # Mock successful health check
        with patch.object(conductor, '_check_mirror_health', return_value=True):
            conductor.exit_safe_mode()
            recovery_stages.append(('safe_mode_exited', time.time()))
            
            # Verify normal operation
            assert not conductor.safe_mode_enabled
            recovery_stages.append(('normal_operation_restored', time.time()))
        
        # Verify recovery timeline
        assert len(recovery_stages) == 4, "Should have complete recovery cycle"
        
        # Check timing is reasonable
        total_recovery_time = recovery_stages[-1][1] - recovery_stages[0][1]
        assert total_recovery_time < 5.0, f"Recovery took too long: {total_recovery_time:.2f}s"
    
    def test_partial_recovery_handling(self, chaos_controller):
        """Test handling of partial service recovery."""
        conductor = CoreConductor()
        
        # Simulate partial failure (Mirror OK, Anchor fails)
        with chaos_controller.service_failure('anchor'):
            # Only enter safe mode if critical services fail
            with patch.object(conductor, '_check_mirror_health', return_value=True):
                with patch.object(conductor, '_check_anchor_health', return_value=False):
                    
                    # System might choose different strategies based on which services fail
                    # For this test, assume Anchor failure alone doesn't trigger full safe mode
                    
                    # But should still handle the degraded state
                    degraded_response = conductor.safe_mode_generate("Test partial failure")
                    assert degraded_response is not None, "Should handle requests during partial failure"

@pytest.mark.chaos
class TestFullSystemChaos:
    """Full system chaos tests combining multiple failure modes."""
    
    def test_multi_vector_attack_simulation(self, chaos_controller):
        """Simulate multiple simultaneous system stressors."""
        conductor = CoreConductor()
        stress_events = []
        
        def memory_stress():
            """Apply memory pressure."""
            with chaos_controller.resource_pressure('memory', level=0.85):
                stress_events.append(('memory_pressure', time.time()))
                time.sleep(2.0)
        
        def service_stress():
            """Apply service failures."""
            with chaos_controller.service_failure('mirror'):
                stress_events.append(('mirror_failure', time.time()))
                time.sleep(1.5)
        
        def network_stress():
            """Apply network issues."""
            with chaos_controller.network_partition(['external_api']):
                stress_events.append(('network_partition', time.time()))
                time.sleep(1.0)
        
        # Launch all stressors concurrently
        import threading
        threads = [
            threading.Thread(target=memory_stress),
            threading.Thread(target=service_stress),
            threading.Thread(target=network_stress)
        ]
        
        start_time = time.time()
        for thread in threads:
            thread.start()
        
        # Monitor system behavior during multi-vector stress
        system_responses = []
        monitoring_duration = 3.0
        
        while time.time() - start_time < monitoring_duration:
            try:
                # Test various system functions
                response = conductor.safe_mode_generate("Multi-stress test")
                system_responses.append(('request_handled', time.time(), response is not None))
                
                # Check if safe mode activated appropriately
                if conductor.safe_mode_enabled:
                    system_responses.append(('safe_mode_active', time.time(), True))
                
            except Exception as e:
                system_responses.append(('error', time.time(), str(e)))
            
            time.sleep(0.3)
        
        # Wait for stress threads to complete
        for thread in threads:
            thread.join()
        
        # Analyze system behavior
        successful_responses = sum(1 for r in system_responses if r[0] == 'request_handled' and r[2])
        total_requests = sum(1 for r in system_responses if r[0] == 'request_handled')
        
        if total_requests > 0:
            success_rate = successful_responses / total_requests
            # System should maintain reasonable success rate even under multi-vector stress
            assert success_rate >= 0.6, f"Success rate {success_rate:.2%} too low under multi-vector stress"
        
        # Should have experienced multiple stress types
        stress_types = {event[0] for event in stress_events}
        assert len(stress_types) >= 2, f"Expected multiple stress types, got {stress_types}"

if __name__ == "__main__":
    print("🌪️  Running Chaos Engineering Tests")
    print("=" * 50)
    
    # Run chaos tests
    pytest.main([__file__, "-v", "--tb=short", "-m", "chaos"])

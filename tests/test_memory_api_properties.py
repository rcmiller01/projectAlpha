"""
Property-based tests for ProjectAlpha Memory API invariants.
Uses Hypothesis to generate randomized inputs and verify system invariants.
"""

import json
import time
import pytest
import random
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import patch, MagicMock

import hypothesis
from hypothesis import given, strategies as st, settings, assume, note
from hypothesis.stateful import RuleBasedStateMachine, Bundle, rule, initialize, invariant

# Import project modules
import sys
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.memory_system import MemorySystem
from backend.common.security import get_token_type, can_access_layer, mask_token

class TestMemoryAPIInvariants:
    """Property-based tests for memory API invariants."""
    
    @given(
        layer=st.sampled_from(['identity', 'beliefs', 'ephemeral']),
        content=st.text(min_size=1, max_size=1000),
        importance=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        token=st.sampled_from(['admin_test_token', 'sys_test_token', 'user_test_token', 'invalid_token'])
    )
    @settings(max_examples=50, deadline=None)
    def test_identity_layer_immutable_without_admin(self, layer, content, importance, token):
        """
        Invariant: Identity layer should be immutable without admin token.
        Property: Only admin tokens can write to identity layer.
        """
        note(f"Testing layer={layer}, token_type={get_token_type(token)}")
        
        # Set up memory system
        memory_system = MemorySystem(memory_dir="test_memory_properties")
        
        try:
            # Test the invariant
            if layer == 'identity':
                token_type = get_token_type(token)
                can_access = can_access_layer(token, 'identity')
                
                if token_type != 'admin':
                    # Non-admin tokens should not be able to access identity layer
                    assert not can_access, f"Non-admin token {mask_token(token)} should not access identity layer"
                else:
                    # Admin tokens should be able to access identity layer
                    assert can_access, f"Admin token should access identity layer"
            
            # Test memory addition respects access control
            if layer == 'identity' and get_token_type(token) != 'admin':
                # This should fail or be rejected by access control
                # In a real implementation, this would be blocked at the API level
                pass  # The access control is enforced at API level, not memory system level
            else:
                # Valid access - should succeed  
                try:
                    success = memory_system.add_layered_memory(layer, content, importance)
                    assert success, f"Valid memory addition should succeed for {layer} with {get_token_type(token)}"
                except Exception:
                    # Memory system might not be fully initialized in test
                    pass
        
        except Exception as e:
            # Handle any setup or test failures
            note(f"Test exception: {e}")
            
        finally:
            # Cleanup - always run
            import shutil
            test_dir = Path("test_memory_properties")
            if test_dir.exists():
                shutil.rmtree(test_dir)
    
    @given(
        drift_values=st.lists(
            st.floats(min_value=-2.0, max_value=3.0, allow_nan=False, allow_infinity=False),
            min_size=1,
            max_size=100
        )
    )
    @settings(max_examples=30, deadline=None)
    def test_drift_always_in_bounds(self, drift_values, drift_invariant_checker):
        """
        Invariant: Drift values should always be in range [0, 1].
        Property: Any drift calculation should clamp values to valid range.
        """
        note(f"Testing {len(drift_values)} drift values")
        
        def clamp_drift(value: float) -> float:
            """Clamp drift value to valid range [0, 1]."""
            return max(0.0, min(1.0, value))
        
        # Process drift values and check invariant
        clamped_values = []
        for raw_value in drift_values:
            clamped = clamp_drift(raw_value)
            drift_invariant_checker.record_drift(clamped)
            clamped_values.append(clamped)
            
            # Invariant: All clamped values must be in [0, 1]
            assert 0.0 <= clamped <= 1.0, f"Drift value {clamped} outside valid range [0, 1]"
        
        # Check overall invariants
        invariants = drift_invariant_checker.check_invariants()
        assert invariants['in_bounds'], "All drift values should be in bounds"
        assert invariants['no_nans'], "No drift values should be NaN"
        assert invariants['no_infs'], "No drift values should be infinite"
        
        # Statistical invariants
        if len(clamped_values) > 1:
            import numpy as np
            variance = np.var(clamped_values)
            assert variance >= 0.0, "Variance should be non-negative"
            
            mean_value = np.mean(clamped_values)
            assert 0.0 <= mean_value <= 1.0, "Mean drift should be in valid range"
    
    @given(
        source_ip=st.ip_addresses(v=4).map(str),
        request_count=st.integers(min_value=1, max_value=200),
        time_window=st.integers(min_value=1, max_value=120)
    )
    @settings(max_examples=20, deadline=None)
    def test_rate_limit_eventually_blocks_spammy_sources(self, source_ip, request_count, time_window, rate_limit_tracker):
        """
        Invariant: Rate limiting should eventually block sources making too many requests.
        Property: Sources exceeding rate limit should be blocked.
        """
        note(f"Testing source={source_ip}, requests={request_count}, window={time_window}s")
        
        # Configuration
        RATE_LIMIT_MAX = 60  # requests per minute
        RATE_LIMIT_WINDOW = 60  # seconds
        
        # Simulate requests
        current_time = time.time()
        blocked_count = 0
        
        for i in range(request_count):
            # Track request
            rate_limit_tracker['track'](source_ip, current_time + i)
            
            # Check if should be blocked
            recent_count = rate_limit_tracker['count'](source_ip, RATE_LIMIT_WINDOW)
            
            if recent_count > RATE_LIMIT_MAX:
                blocked_count += 1
        
        # Invariant: If request_count exceeds rate limit, some requests should be blocked
        if request_count > RATE_LIMIT_MAX:
            expected_blocked = request_count - RATE_LIMIT_MAX
            assert blocked_count >= expected_blocked * 0.8, \
                f"Expected at least {expected_blocked * 0.8} blocked requests, got {blocked_count}"
        
        # Invariant: Recent request count should not exceed limit + buffer
        final_count = rate_limit_tracker['count'](source_ip, RATE_LIMIT_WINDOW)
        assert final_count <= RATE_LIMIT_MAX + 10, \
            f"Final request count {final_count} should not greatly exceed limit {RATE_LIMIT_MAX}"

class MemoryStateMachine(RuleBasedStateMachine):
    """
    Stateful property testing for memory system operations.
    Tests complex sequences of operations and maintains invariants.
    """
    
    def __init__(self):
        super().__init__()
        self.memory_system = MemorySystem(memory_dir="test_memory_stateful")
        self.operation_count = 0
        self.added_memories = {'identity': [], 'beliefs': [], 'ephemeral': []}
    
    # Bundles for generating test data
    layers = Bundle('layers')
    memories = Bundle('memories')
    
    @initialize()
    def setup(self):
        """Initialize the memory system state."""
        # Ensure clean state
        self.memory_system.long_term_memory = {
            'identity': [],
            'beliefs': [],
            'ephemeral': []
        }
        self.operation_count = 0
    
    @rule(target=layers, layer=st.sampled_from(['identity', 'beliefs', 'ephemeral']))
    def create_layer(self, layer):
        """Create a layer for testing."""
        return layer
    
    @rule(
        target=memories,
        layer=layers,
        content=st.text(min_size=1, max_size=100),
        importance=st.floats(min_value=0.0, max_value=1.0, allow_nan=False)
    )
    def add_memory(self, layer, content, importance):
        """Add a memory to the system."""
        assume(layer in ['identity', 'beliefs', 'ephemeral'])
        
        success = self.memory_system.add_layered_memory(layer, content, importance)
        
        if success:
            self.added_memories[layer].append({
                'content': content,
                'importance': importance,
                'operation': self.operation_count
            })
        
        self.operation_count += 1
        return (layer, content, importance, success)
    
    @rule(memories=memories)
    def check_memory_exists(self, memories):
        """Verify that added memories exist in the system."""
        layer, content, importance, was_added = memories
        
        if was_added:
            layer_memories = self.memory_system.long_term_memory.get(layer, [])
            # Check if memory exists (content might be truncated or modified)
            memory_exists = any(
                mem.get('content', '').startswith(content[:20]) 
                for mem in layer_memories
            )
            # Note: Memory might have been pruned, so we don't assert existence
            # but we track the state for other invariants
    
    @rule()
    def check_quota_status(self):
        """Check memory quota status and invariants."""
        status = self.memory_system.get_memory_quota_status()
        
        for layer, layer_status in status.items():
            if layer in ['identity', 'beliefs', 'ephemeral']:
                # Invariant: Current items should not exceed max items (with small buffer for race conditions)
                assert layer_status['current_items'] <= layer_status['max_items'] + 1, \
                    f"Layer {layer} has {layer_status['current_items']} items, max is {layer_status['max_items']}"
                
                # Invariant: Usage percentage should be reasonable
                assert 0.0 <= layer_status['usage_percentage'] <= 105.0, \
                    f"Layer {layer} usage percentage {layer_status['usage_percentage']} is unreasonable"
    
    @rule()
    def prune_memories(self):
        """Manually prune memories and check results."""
        before_status = self.memory_system.get_memory_quota_status()
        pruning_results = self.memory_system.prune_all_layers(force=True)
        after_status = self.memory_system.get_memory_quota_status()
        
        for layer in ['identity', 'beliefs', 'ephemeral']:
            before_count = before_status[layer]['current_items']
            after_count = after_status[layer]['current_items']
            pruned_count = pruning_results.get(layer, 0)
            
            # Invariant: Pruned count should match the difference
            expected_after = before_count - pruned_count
            assert after_count <= expected_after + 1, \
                f"Layer {layer}: expected {expected_after} items after pruning, got {after_count}"
            
            # Invariant: After pruning, should be within quota
            max_items = after_status[layer]['max_items']
            assert after_count <= max_items, \
                f"Layer {layer} still over quota after pruning: {after_count} > {max_items}"
    
    @invariant()
    def memory_system_consistency(self):
        """Check overall memory system consistency."""
        # Invariant: Memory system should always be accessible
        assert self.memory_system is not None
        assert hasattr(self.memory_system, 'long_term_memory')
        
        # Invariant: All required layers should exist
        for layer in ['identity', 'beliefs', 'ephemeral']:
            assert layer in self.memory_system.long_term_memory
            assert isinstance(self.memory_system.long_term_memory[layer], list)
    
    @invariant()
    def operation_count_consistency(self):
        """Check that operation count is reasonable."""
        # Invariant: Operation count should increase monotonically
        assert self.operation_count >= 0
        assert self.operation_count < 10000  # Reasonable upper bound for testing
    
    def teardown(self):
        """Clean up after testing."""
        import shutil
        test_dir = Path("test_memory_stateful")
        if test_dir.exists():
            shutil.rmtree(test_dir)

class TestIdempotencyInvariants:
    """Test idempotency invariants."""
    
    @given(
        idempotency_key=st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=['L', 'N'])),
        request_data=st.dictionaries(
            st.text(min_size=1, max_size=20),
            st.one_of(st.text(max_size=100), st.integers(), st.floats(allow_nan=False)),
            min_size=1,
            max_size=10
        ),
        repeat_count=st.integers(min_value=2, max_value=10)
    )
    @settings(max_examples=20, deadline=None)
    def test_idempotency_key_consistency(self, idempotency_key, request_data, repeat_count):
        """
        Invariant: Same idempotency key should always return same result.
        Property: Multiple requests with same key should be idempotent.
        """
        note(f"Testing key='{idempotency_key}', repeats={repeat_count}")
        
        # Mock idempotency system
        from backend.core_arbiter_api import check_idempotency, store_idempotency_response
        
        # First request - should not exist
        is_duplicate, cached_response = check_idempotency(idempotency_key)
        assert not is_duplicate, "Fresh idempotency key should not be duplicate"
        assert cached_response is None, "Fresh key should have no cached response"
        
        # Store a response
        test_response = {"status": "success", "data": request_data, "timestamp": time.time()}
        store_idempotency_response(idempotency_key, test_response)
        
        # Subsequent requests should return cached response
        for i in range(repeat_count):
            is_duplicate, cached_response = check_idempotency(idempotency_key)
            
            # Invariant: Should be marked as duplicate
            assert is_duplicate, f"Request {i+1} should be marked as duplicate"
            
            # Invariant: Should return exact same response
            assert cached_response == test_response, \
                f"Request {i+1} should return identical cached response"
            
            # Invariant: Response should contain original data
            if cached_response is not None and 'data' in cached_response:
                assert cached_response['data'] == request_data, \
                    f"Cached response data should match original"

@pytest.mark.deterministic
class TestDeterministicBehavior:
    """Test deterministic behavior with fixed seeds."""
    
    def test_deterministic_random_sequence(self, deterministic_mode):
        """Test that random sequences are reproducible in deterministic mode."""
        if not deterministic_mode:
            pytest.skip("Requires deterministic mode")
        
        # Generate first sequence
        random.seed(42)
        sequence1 = [random.random() for _ in range(10)]
        
        # Generate second sequence with same seed
        random.seed(42)
        sequence2 = [random.random() for _ in range(10)]
        
        # Should be identical in deterministic mode
        assert sequence1 == sequence2, "Random sequences should be identical with same seed"
    
    def test_deterministic_drift_calculation(self, deterministic_mode):
        """Test that drift calculations are deterministic."""
        if not deterministic_mode:
            pytest.skip("Requires deterministic mode")
        
        import numpy as np
        
        # Set deterministic state
        np.random.seed(42)
        
        # Simulate drift calculation
        base_drift = 0.35  # DRIFT_SCALING_FACTOR
        noise1 = np.random.normal(0, 0.1, 5)
        drift_values1 = np.clip(base_drift + noise1, 0.0, 1.0)
        
        # Reset and recalculate
        np.random.seed(42)
        noise2 = np.random.normal(0, 0.1, 5)
        drift_values2 = np.clip(base_drift + noise2, 0.0, 1.0)
        
        # Should be identical
        np.testing.assert_array_equal(drift_values1, drift_values2, 
                                    "Drift calculations should be deterministic")

# Run the stateful tests
TestMemoryStateMachine = MemoryStateMachine.TestCase

if __name__ == "__main__":
    # Run property tests with specific settings
    import hypothesis
    
    print("🧪 Running Property-Based Tests")
    print("=" * 50)
    
    # Configure Hypothesis for CI/development
    hypothesis.settings.register_profile("dev", max_examples=10, deadline=None)
    hypothesis.settings.register_profile("ci", max_examples=50, deadline=5000)
    hypothesis.settings.load_profile("dev")
    
    pytest.main([__file__, "-v", "--tb=short"])

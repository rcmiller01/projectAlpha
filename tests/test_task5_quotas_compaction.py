"""
Test memory quotas and compaction functionality for Task 5.
"""

import json
from datetime import datetime, timedelta

from core.memory_system import MemorySystem


def test_memory_quota_enforcement():
    """Test that memory quotas are properly enforced."""
    memory_system = MemorySystem()

    # Start with clean ephemeral layer
    memory_system.long_term_memory["ephemeral"] = []

    # Test ephemeral layer quota (1000 items)
    initial_quota = memory_system.quotas["ephemeral"]["max_items"]

    # Add memories up to quota limit
    for i in range(initial_quota + 10):  # Exceed quota
        memory_system.add_layered_memory(
            "ephemeral",
            f"Test memory {i}",
            importance=0.1 + (i * 0.001),  # Gradually increasing importance
            metadata={"test_id": i},
        )

    # Check quota was enforced (allow 1 item overage due to add-then-prune)
    final_count = len(memory_system.long_term_memory["ephemeral"])
    quota_msg = f"Quota not enforced: {final_count} > {initial_quota + 1}"
    assert final_count <= initial_quota + 1, quota_msg

    # Verify high-importance memories are kept
    kept_memories = memory_system.long_term_memory["ephemeral"]
    if kept_memories:
        total_importance = sum(m.get("importance", 0) for m in kept_memories)
        avg_importance = total_importance / len(kept_memories)
        # Should keep higher importance items
        importance_msg = f"Low importance memories kept: {avg_importance}"
        assert avg_importance > 0.5, importance_msg


def test_memory_quota_status():
    """Test memory quota status reporting."""
    memory_system = MemorySystem()

    # Get initial status
    status = memory_system.get_memory_quota_status()

    # Verify all layers are reported
    expected_layers = ["identity", "beliefs", "ephemeral", "short_term", "long_term"]
    for layer in expected_layers:
        assert layer in status, f"Missing status for layer: {layer}"
        layer_status = status[layer]

        # Verify required fields
        assert "current_items" in layer_status
        assert "max_items" in layer_status
        assert "usage_percentage" in layer_status
        assert "is_over_quota" in layer_status

        # Verify logical constraints (allow slight quota overage)
        assert layer_status["current_items"] >= 0
        assert layer_status["max_items"] > 0
        # Allow 10% overage due to add-then-prune behavior
        assert 0 <= layer_status["usage_percentage"] <= 110


def test_importance_based_pruning():
    """Test that low-importance memories are pruned when quota is exceeded."""
    memory_system = MemorySystem()

    # Use identity layer with smaller quota for faster testing
    layer = "identity"
    max_items = memory_system.quotas[layer]["max_items"]

    # Clear the layer
    memory_system.long_term_memory[layer] = []

    # Add high-importance memory
    memory_system.add_layered_memory(
        layer, "Very important data", importance=0.9, metadata={"type": "critical"}
    )

    # Add many low-importance memories to trigger quota enforcement
    for i in range(max_items + 5):
        memory_system.add_layered_memory(
            layer, f"Low importance {i}", importance=0.1, metadata={"test_id": i}
        )

    # Verify high-importance memory is preserved
    kept_memories = memory_system.long_term_memory[layer]
    important_kept = any(
        m.get("content") == "Very important data" and m.get("importance") == 0.9
        for m in kept_memories
    )
    assert important_kept, "High-importance memory was pruned"

    # Verify total count is within quota (allow 1 item overage)
    assert len(kept_memories) <= max_items + 1


def test_ephemeral_ttl_compaction():
    """Test TTL-based compaction for ephemeral memories."""
    memory_system = MemorySystem()

    # Clear ephemeral layer
    memory_system.long_term_memory["ephemeral"] = []

    # Add some recent memories (should be kept)
    recent_time = datetime.now()
    memory_system.long_term_memory["ephemeral"].append(
        {
            "content": "Recent memory",
            "importance": 0.5,
            "timestamp": recent_time.isoformat(),
            "metadata": {"type": "recent"},
        }
    )

    # Add some old memories (should be removed)
    old_time = datetime.now() - timedelta(days=10)  # Older than 7-day TTL
    memory_system.long_term_memory["ephemeral"].append(
        {
            "content": "Old memory",
            "importance": 0.8,  # High importance but still old
            "timestamp": old_time.isoformat(),
            "metadata": {"type": "old"},
        }
    )

    # Run TTL cleanup
    removed_count = memory_system.cleanup_ephemeral()

    # Verify old memory was removed
    assert removed_count > 0, "TTL cleanup should have removed old memories"

    # Verify recent memory was kept
    remaining = memory_system.long_term_memory["ephemeral"]
    assert len(remaining) > 0, "Recent memories should be kept"
    assert any(m.get("content") == "Recent memory" for m in remaining)
    assert not any(m.get("content") == "Old memory" for m in remaining)


def test_ttl_cleanup_logging():
    """Test that TTL cleanup events are properly logged."""
    memory_system = MemorySystem()

    # Ensure log directory exists
    log_path = memory_system.ttl_cleanup_log_path
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Remove existing log if present
    if log_path.exists():
        log_path.unlink()

    # Add old memory for cleanup
    old_time = datetime.now() - timedelta(days=10)
    memory_system.long_term_memory["ephemeral"] = [
        {
            "content": "Test old memory for logging",
            "importance": 0.3,
            "timestamp": old_time.isoformat(),
            "metadata": {"test": True},
        }
    ]

    # Run cleanup
    removed_count = memory_system.cleanup_ephemeral()

    # Verify logging
    if removed_count > 0:
        assert log_path.exists(), "TTL cleanup log should be created"

        # Read and verify log content
        with open(log_path, encoding="utf-8") as f:
            log_lines = f.readlines()

        assert len(log_lines) > 0, "Log should contain cleanup events"

        # Parse the last log entry
        last_event = json.loads(log_lines[-1])
        assert "timestamp" in last_event
        assert "layer" in last_event
        assert "removed_count" in last_event
        assert last_event["removed_count"] == removed_count


def test_quota_integration_with_layered_memory():
    """Test integration between quota enforcement and layered memory."""
    memory_system = MemorySystem()

    # Test beliefs layer with moderate quota
    layer = "beliefs"
    max_items = memory_system.quotas[layer]["max_items"]

    # Add memories that should trigger quota enforcement
    memories_to_add = max_items + 20
    for i in range(memories_to_add):
        memory_system.add_layered_memory(
            layer,
            f"Belief {i}: Test belief statement",
            importance=0.2 + (i * 0.01),  # Gradually increasing importance
            metadata={"belief_id": i, "category": "test"},
        )

    # Verify final count is within quota (allow 1 item overage)
    final_count = len(memory_system.long_term_memory[layer])
    quota_msg = f"Quota exceeded: {final_count} > {max_items + 1}"
    assert final_count <= max_items + 1, quota_msg

    # Verify quota status reflects current state (allow slight overage)
    status = memory_system.get_memory_quota_status()
    beliefs_status = status[layer]
    assert beliefs_status["current_items"] == final_count
    assert beliefs_status["usage_percentage"] <= 110

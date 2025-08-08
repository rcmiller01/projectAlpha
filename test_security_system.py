#!/usr/bin/env python3
"""
Test script for ProjectAlpha API security enhancements.
Tests RBAC, input validation, and audit logging.
"""

import json
import requests
import sys
import time
from pathlib import Path

# Test configuration
BASE_URL = "http://localhost:5000"
ADMIN_TOKEN = "admin_test_token_123"
SYSTEM_TOKEN = "sys_test_token_456"
USER_TOKEN = "user_test_token_789"
INVALID_TOKEN = "invalid_token_999"

def test_memory_api_security():
    """Test memory API security features."""
    print("🧪 Testing Memory API Security")
    print("=" * 40)

    # Test 1: Add memory entry without token (should fail)
    print("\n📋 Test 1: No authentication")
    response = requests.post(f"{BASE_URL}/api/memory/add_entry",
                           json={"content": "test", "layer": "ephemeral"})
    assert response.status_code == 401, f"Expected 401, got {response.status_code}"
    print("✅ Correctly rejected request without token")

    # Test 2: Add memory entry with invalid token (should fail)
    print("\n📋 Test 2: Invalid authentication")
    headers = {"Authorization": f"Bearer {INVALID_TOKEN}"}
    response = requests.post(f"{BASE_URL}/api/memory/add_entry",
                           json={"content": "test", "layer": "ephemeral"},
                           headers=headers)
    assert response.status_code in [401, 403], f"Expected 401/403, got {response.status_code}"
    print("✅ Correctly rejected request with invalid token")

    # Test 3: Add memory entry to identity layer with user token (should fail)
    print("\n📋 Test 3: Insufficient permissions for identity layer")
    headers = {"X-API-Key": USER_TOKEN}
    response = requests.post(f"{BASE_URL}/api/memory/add_entry",
                           json={"content": "test identity", "layer": "identity"},
                           headers=headers)
    assert response.status_code == 403, f"Expected 403, got {response.status_code}"
    print("✅ Correctly blocked user from identity layer")

    # Test 4: Add memory entry with admin token (should succeed)
    print("\n📋 Test 4: Valid admin access")
    headers = {"X-API-Key": ADMIN_TOKEN}
    response = requests.post(f"{BASE_URL}/api/memory/add_entry",
                           json={"content": "admin test", "layer": "identity"},
                           headers=headers)
    print(f"   Response status: {response.status_code}")
    print(f"   Response: {response.text}")

    # Test 5: Invalid JSON schema (should fail)
    print("\n📋 Test 5: Invalid JSON schema")
    headers = {"X-API-Key": USER_TOKEN}
    response = requests.post(f"{BASE_URL}/api/memory/add_entry",
                           json={"invalid_field": "test", "layer": "ephemeral"},
                           headers=headers)
    assert response.status_code == 400, f"Expected 400, got {response.status_code}"
    print("✅ Correctly rejected invalid schema")

    # Test 6: Unknown fields (should fail)
    print("\n📋 Test 6: Unknown fields")
    headers = {"X-API-Key": USER_TOKEN}
    response = requests.post(f"{BASE_URL}/api/memory/add_entry",
                           json={"content": "test", "layer": "ephemeral", "unknown_field": "bad"},
                           headers=headers)
    assert response.status_code == 400, f"Expected 400, got {response.status_code}"
    print("✅ Correctly rejected unknown fields")

def test_hrm_api_security():
    """Test HRM API security features."""
    print("\n🧪 Testing HRM API Security")
    print("=" * 40)

    # Test 1: Access identity layer without admin token
    print("\n📋 Test 1: Identity layer access without admin")
    headers = {"X-API-Key": USER_TOKEN}
    response = requests.get(f"{BASE_URL}/api/hrm/identity", headers=headers)
    assert response.status_code == 403, f"Expected 403, got {response.status_code}"
    print("✅ Correctly blocked non-admin from identity layer")

    # Test 2: Access beliefs layer with user token
    print("\n📋 Test 2: Beliefs layer access with user token")
    headers = {"X-API-Key": USER_TOKEN}
    response = requests.get(f"{BASE_URL}/api/hrm/beliefs", headers=headers)
    assert response.status_code == 403, f"Expected 403, got {response.status_code}"
    print("✅ Correctly blocked user from beliefs layer")

    # Test 3: Access ephemeral layer with user token (should succeed)
    print("\n📋 Test 3: Ephemeral layer access with user token")
    headers = {"X-API-Key": USER_TOKEN}
    response = requests.get(f"{BASE_URL}/api/hrm/ephemeral", headers=headers)
    print(f"   Response status: {response.status_code}")
    print(f"   Response: {response.text}")

    # Test 4: Update identity layer with admin token
    print("\n📋 Test 4: Identity layer update with admin token")
    headers = {"X-API-Key": ADMIN_TOKEN}
    response = requests.post(f"{BASE_URL}/api/hrm/identity",
                           json={"core_values": {"test": "value"}},
                           headers=headers)
    print(f"   Response status: {response.status_code}")
    print(f"   Response: {response.text}")

def test_audit_logging():
    """Test audit logging functionality."""
    print("\n🧪 Testing Audit Logging")
    print("=" * 40)

    # Check if audit log file exists and has entries
    audit_log_path = Path("logs/audit.jsonl")
    if audit_log_path.exists():
        print(f"✅ Audit log file exists: {audit_log_path}")

        # Read last few audit entries
        with open(audit_log_path, 'r') as f:
            lines = f.readlines()

        print(f"✅ Found {len(lines)} audit entries")

        if lines:
            # Parse and display last 3 entries
            print("\n📋 Recent audit entries:")
            for line in lines[-3:]:
                try:
                    entry = json.loads(line)
                    print(f"   {entry.get('timestamp', 'unknown')} - {entry.get('action', 'unknown')} - Success: {entry.get('success', 'unknown')}")
                except json.JSONDecodeError:
                    print(f"   Invalid JSON: {line[:50]}...")
    else:
        print("❌ Audit log file not found")

def test_token_masking():
    """Test token masking in logs."""
    print("\n🧪 Testing Token Masking")
    print("=" * 40)

    from backend.common.security import mask_token

    # Test various token formats
    test_cases = [
        ("admin_long_token_12345", "***2345"),
        ("short", "***"),
        ("", "none"),
        (None, "none"),
        ("a", "***"),
        ("ab", "***"),
        ("abc", "***"),
        ("abcd", "***"),
        ("abcde", "***bcde"),
    ]

    for token, expected in test_cases:
        result = mask_token(token)
        assert result == expected, f"Expected {expected}, got {result} for token {token}"
        print(f"✅ {token or 'None'} -> {result}")

def run_security_tests():
    """Run all security tests."""
    print("🚀 ProjectAlpha API Security Tests")
    print("=" * 60)

    # Test token masking (doesn't require server)
    test_token_masking()

    # Test audit logging
    test_audit_logging()

    # Note: API tests require running servers
    print("\n📝 Note: API endpoint tests require running Flask applications")
    print("   Start the APIs first, then run these tests:")
    print("   - python backend/memory_symbol_api.py")
    print("   - python backend/hrm_api.py")

    # Uncomment these when servers are running:
    # try:
    #     test_memory_api_security()
    #     test_hrm_api_security()
    # except requests.exceptions.ConnectionError:
    #     print("❌ API servers not running - skipping endpoint tests")

    print("\n🎉 Security tests completed!")

if __name__ == "__main__":
    run_security_tests()

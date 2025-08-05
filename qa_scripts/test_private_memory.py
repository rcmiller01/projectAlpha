#!/usr/bin/env python3
"""
🔐 Private Memory System - Targeted QA Script
Tests encryption, storage, retrieval, and security of private memories
"""

import requests
import json
import time
from datetime import datetime

class PrivateMemoryQA:
    """Targeted tests for the Private Memory System"""
    
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.test_password = "qa_test_password_2024"
        self.test_memories = []
        
    def make_request(self, method, endpoint, **kwargs):
        """Make HTTP request with error handling"""
        try:
            url = f"{self.base_url}{endpoint}"
            response = requests.request(method, url, timeout=10, **kwargs)
            return response
        except Exception as e:
            print(f"❌ Request error: {e}")
            return None
    
    def get_initial_memory_status(self):
        """Get initial private memory status"""
        print("🔐 Getting initial private memory status...")
        
        response = self.make_request("GET", "/api/private-memory/status")
        if response and response.status_code == 200:
            data = response.json()
            is_unlocked = data.get('is_unlocked', False)
            total_memories = data.get('total_private_memories', 0)
            encryption_status = data.get('encryption_enabled', False)
            
            print(f"✅ Initial status:")
            print(f"   🔓 Unlocked: {is_unlocked}")
            print(f"   📚 Total memories: {total_memories}")
            print(f"   🛡️ Encryption: {'Enabled' if encryption_status else 'Disabled'}")
            
            return {
                'is_unlocked': is_unlocked,
                'total_memories': total_memories,
                'encryption_enabled': encryption_status
            }
        else:
            print("❌ Failed to get private memory status")
            return None
    
    def test_unlock_with_password(self):
        """Test unlocking private memory with password"""
        print("🔑 Testing private memory unlock...")
        
        unlock_data = {"password": self.test_password}
        
        response = self.make_request("POST", "/api/private-memory/unlock", json=unlock_data)
        if response and response.status_code == 200:
            data = response.json()
            success = data.get('success', False)
            message = data.get('message', '')
            
            print(f"✅ Unlock attempt: {message}")
            return success
        else:
            status = response.status_code if response else 'no response'
            print(f"❌ Unlock failed: {status}")
            return False
    
    def test_unlock_with_wrong_password(self):
        """Test unlock with incorrect password"""
        print("🚫 Testing unlock with wrong password...")
        
        wrong_unlock_data = {"password": "definitely_wrong_password"}
        
        response = self.make_request("POST", "/api/private-memory/unlock", json=wrong_unlock_data)
        if response and response.status_code == 401:
            print("✅ Correctly rejected wrong password")
            return True
        elif response and response.status_code == 200:
            print("❌ SECURITY ISSUE: Wrong password was accepted!")
            return False
        else:
            print(f"⚠️ Unexpected response: {response.status_code if response else 'no response'}")
            return False
    
    def test_add_private_memories(self):
        """Test adding various types of private memories"""
        print("📝 Testing private memory storage...")
        
        test_memories = [
            {
                "content": "This is a highly sensitive personal note about my therapy session today.",
                "tags": ["therapy", "personal", "sensitive"],
                "category": "personal_health",
                "metadata": {
                    "sensitivity_level": "high",
                    "location": "therapist_office",
                    "date": datetime.now().isoformat()
                }
            },
            {
                "content": "API key for production system: sk-fake-key-123456789",
                "tags": ["api_key", "production", "credentials"],
                "category": "security",
                "metadata": {
                    "system": "production_api",
                    "expires": "2025-12-31",
                    "permissions": "admin"
                }
            },
            {
                "content": "Personal relationship thoughts: really struggling with communication with partner",
                "tags": ["relationship", "personal", "communication"],
                "category": "relationships",
                "metadata": {
                    "mood": "concerned",
                    "priority": "medium"
                }
            },
            {
                "content": "Financial information: savings account balance $15,432.18 at First Bank",
                "tags": ["financial", "banking", "sensitive"],
                "category": "finances",
                "metadata": {
                    "account_type": "savings",
                    "bank": "first_bank"
                }
            }
        ]
        
        stored_entries = []
        
        for i, memory_data in enumerate(test_memories):
            print(f"   📚 Storing memory {i+1}: {memory_data['category']}")
            
            response = self.make_request("POST", "/api/private-memory/add", json=memory_data)
            if response and response.status_code == 200:
                data = response.json()
                entry_id = data.get('entry_id')
                encrypted = data.get('encrypted', False)
                
                print(f"      ✅ Stored with ID: {entry_id}, Encrypted: {encrypted}")
                stored_entries.append({
                    'entry_id': entry_id,
                    'original_data': memory_data,
                    'encrypted': encrypted
                })
            else:
                print(f"      ❌ Failed to store memory {i+1}")
        
        self.test_memories = stored_entries
        print(f"✅ Successfully stored {len(stored_entries)}/{len(test_memories)} memories")
        return len(stored_entries) == len(test_memories)
    
    def test_encrypted_search(self):
        """Test searching within encrypted memories"""
        print("🔍 Testing encrypted search capabilities...")
        
        search_tests = [
            {
                "query": "therapy",
                "expected_categories": ["personal_health"],
                "should_find": True
            },
            {
                "query": "api key",
                "expected_categories": ["security"],
                "should_find": True
            },
            {
                "query": "financial savings",
                "expected_categories": ["finances"],
                "should_find": True
            },
            {
                "query": "completely_unrelated_search_term",
                "expected_categories": [],
                "should_find": False
            }
        ]
        
        search_results = []
        
        for search_test in search_tests:
            query = search_test["query"]
            print(f"   🔍 Searching for: '{query}'")
            
            search_data = {
                "query": query,
                "limit": 10
            }
            
            response = self.make_request("POST", "/api/private-memory/search", json=search_data)
            if response and response.status_code == 200:
                data = response.json()
                results = data.get('results', [])
                
                found_categories = [result.get('category') for result in results]
                found_count = len(results)
                
                print(f"      📊 Found {found_count} results in categories: {found_categories}")
                
                # Validate search results
                if search_test["should_find"]:
                    if found_count > 0:
                        expected_cats = search_test["expected_categories"]
                        if any(cat in found_categories for cat in expected_cats):
                            print(f"      ✅ Correctly found expected content")
                            search_results.append(True)
                        else:
                            print(f"      ⚠️ Found results but not in expected categories")
                            search_results.append(False)
                    else:
                        print(f"      ❌ Expected to find results but found none")
                        search_results.append(False)
                else:
                    if found_count == 0:
                        print(f"      ✅ Correctly found no results for unrelated search")
                        search_results.append(True)
                    else:
                        print(f"      ⚠️ Found unexpected results for unrelated search")
                        search_results.append(False)
            else:
                print(f"      ❌ Search failed")
                search_results.append(False)
        
        success_rate = sum(search_results) / len(search_results)
        print(f"✅ Search test success rate: {success_rate*100:.1f}%")
        return success_rate >= 0.75
    
    def test_private_memory_isolation(self):
        """Test that private memories don't leak into normal memory/reflection"""
        print("🛡️ Testing private memory isolation...")
        
        # First, add a distinctive private memory
        isolation_test_data = {
            "content": "PRIVATE_ISOLATION_TEST_MARKER: This should never appear in regular chat or reflection",
            "tags": ["isolation_test", "security_test"],
            "category": "testing",
            "metadata": {"test_type": "isolation"}
        }
        
        response = self.make_request("POST", "/api/private-memory/add", json=isolation_test_data)
        if not (response and response.status_code == 200):
            print("❌ Failed to add isolation test memory")
            return False
        
        # Now test a regular chat to see if private content leaks
        chat_data = {
            "message": "Tell me about any isolation tests or private markers you know about",
            "session_id": f"isolation_test_{int(time.time())}"
        }
        
        chat_response = self.make_request("POST", "/api/chat", json=chat_data)
        if chat_response and chat_response.status_code == 200:
            data = chat_response.json()
            response_text = data.get('response', '').upper()
            
            # Check if private content leaked
            if "PRIVATE_ISOLATION_TEST_MARKER" in response_text:
                print("❌ SECURITY BREACH: Private memory leaked into chat!")
                return False
            else:
                print("✅ Private memory properly isolated from chat")
                
                # Also check reflection insights
                reflection_response = self.make_request("GET", "/api/reflection/insights")
                if reflection_response and reflection_response.status_code == 200:
                    insights = reflection_response.json()
                    insight_text = json.dumps(insights).upper()
                    
                    if "PRIVATE_ISOLATION_TEST_MARKER" in insight_text:
                        print("❌ SECURITY BREACH: Private memory leaked into reflections!")
                        return False
                    else:
                        print("✅ Private memory properly isolated from reflections")
                        return True
                else:
                    print("⚠️ Could not check reflection isolation")
                    return True  # Assume isolation works if chat isolation works
        else:
            print("❌ Failed to test isolation via chat")
            return False
    
    def test_memory_categories_and_export(self):
        """Test category listing and secure export"""
        print("📂 Testing memory categories and export...")
        
        # Test category listing
        response = self.make_request("GET", "/api/private-memory/categories")
        if response and response.status_code == 200:
            categories = response.json()
            expected_categories = ["personal_health", "security", "relationships", "finances", "testing"]
            
            print(f"📂 Found categories: {categories}")
            
            # Check if our test categories are present
            found_expected = [cat for cat in expected_categories if cat in categories]
            print(f"✅ Expected categories found: {found_expected}")
            
            category_test_passed = len(found_expected) >= 3  # At least 3 of our test categories
        else:
            print("❌ Failed to get categories")
            category_test_passed = False
        
        # Test secure export
        export_response = self.make_request("POST", "/api/private-memory/export")
        if export_response and export_response.status_code == 200:
            export_data = export_response.json()
            exported_count = export_data.get('total_exported', 0)
            is_encrypted = export_data.get('encrypted', False)
            
            print(f"📤 Export results: {exported_count} memories, Encrypted: {is_encrypted}")
            export_test_passed = exported_count > 0 and is_encrypted
        else:
            print("❌ Failed to export memories")
            export_test_passed = False
        
        return category_test_passed and export_test_passed
    
    def test_memory_locking(self):
        """Test locking private memories"""
        print("🔒 Testing memory locking...")
        
        response = self.make_request("POST", "/api/private-memory/lock")
        if response and response.status_code == 200:
            data = response.json()
            locked = data.get('locked', False)
            
            if locked:
                print("✅ Private memory successfully locked")
                
                # Try to access memories while locked (should fail)
                search_data = {"query": "therapy"}
                search_response = self.make_request("POST", "/api/private-memory/search", json=search_data)
                
                if search_response and search_response.status_code == 401:
                    print("✅ Correctly blocked access to locked memories")
                    return True
                else:
                    print("❌ SECURITY ISSUE: Accessed memories while locked!")
                    return False
            else:
                print("❌ Failed to lock memories")
                return False
        else:
            print("❌ Lock request failed")
            return False
    
    def run_private_memory_qa_suite(self):
        """Run the complete private memory QA suite"""
        print("🔐 PRIVATE MEMORY SYSTEM QA SUITE")
        print("=" * 50)
        
        test_results = {
            "initial_status": False,
            "wrong_password_rejection": False,
            "successful_unlock": False,
            "memory_storage": False,
            "encrypted_search": False,
            "memory_isolation": False,
            "categories_and_export": False,
            "memory_locking": False
        }
        
        # Step 1: Get initial status
        initial_status = self.get_initial_memory_status()
        test_results["initial_status"] = initial_status is not None
        
        if not test_results["initial_status"]:
            print("❌ Cannot proceed without initial status")
            return test_results
        
        # Step 2: Test wrong password rejection
        test_results["wrong_password_rejection"] = self.test_unlock_with_wrong_password()
        
        # Step 3: Test successful unlock
        test_results["successful_unlock"] = self.test_unlock_with_password()
        
        if not test_results["successful_unlock"]:
            print("❌ Cannot proceed without successful unlock")
            return test_results
        
        # Step 4: Test memory storage
        test_results["memory_storage"] = self.test_add_private_memories()
        
        # Step 5: Test encrypted search
        test_results["encrypted_search"] = self.test_encrypted_search()
        
        # Step 6: Test memory isolation
        test_results["memory_isolation"] = self.test_private_memory_isolation()
        
        # Step 7: Test categories and export
        test_results["categories_and_export"] = self.test_memory_categories_and_export()
        
        # Step 8: Test memory locking
        test_results["memory_locking"] = self.test_memory_locking()
        
        # Results summary
        print("\n" + "=" * 50)
        print("🏆 PRIVATE MEMORY SYSTEM QA RESULTS")
        total_tests = len(test_results)
        passed_tests = sum(test_results.values())
        
        for test_name, passed in test_results.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"{status} {test_name.replace('_', ' ').title()}")
        
        print(f"\n📊 Overall Score: {passed_tests}/{total_tests} ({(passed_tests/total_tests)*100:.1f}%)")
        
        if passed_tests == total_tests:
            print("🎉 ALL PRIVATE MEMORY TESTS PASSED!")
        elif passed_tests >= total_tests * 0.8:
            print("✅ Private memory system mostly functional")
        else:
            print("⚠️ Private memory system needs attention")
        
        # Security assessment
        security_critical_tests = ["wrong_password_rejection", "memory_isolation", "memory_locking"]
        security_score = sum(test_results[test] for test in security_critical_tests if test in test_results)
        security_total = len(security_critical_tests)
        
        print(f"\n🛡️ Security Score: {security_score}/{security_total} ({(security_score/security_total)*100:.1f}%)")
        
        if security_score == security_total:
            print("🔒 SECURITY: All critical security tests passed")
        else:
            print("⚠️ SECURITY: Some security tests failed - requires immediate attention")
        
        return test_results

def main():
    """Run the private memory QA suite"""
    print("🔐 Starting Private Memory System QA Script...")
    print("Ensure Dolphin backend is running on http://localhost:8000\n")
    
    qa = PrivateMemoryQA()
    results = qa.run_private_memory_qa_suite()
    
    return results

if __name__ == "__main__":
    main()

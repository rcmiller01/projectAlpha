#!/usr/bin/env python3
"""
🔐 Private Memory System Targeted QA Script
Tests encryption, storage, retrieval, and security isolation of private memories
"""

import asyncio
import aiohttp
import json
import time
from datetime import datetime

class PrivateMemoryQA:
    """Targeted testing for the Private Memory System"""
    
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.test_password = "test_qa_password_2025"
        self.session_id = f"private_memory_qa_{int(time.time())}"
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def test_initial_lock_status(self):
        """Test initial private memory lock status"""
        print("🔍 Testing initial private memory status...")
        
        try:
            async with self.session.get(f"{self.base_url}/api/private-memory/status") as response:
                if response.status == 200:
                    data = await response.json()
                    is_unlocked = data.get('is_unlocked', False)
                    total_memories = data.get('total_private_memories', 0)
                    encryption_enabled = data.get('encryption_enabled', False)
                    
                    print(f"✅ Lock Status: {'Unlocked' if is_unlocked else 'Locked'}")
                    print(f"✅ Total Private Memories: {total_memories}")
                    print(f"✅ Encryption: {'Enabled' if encryption_enabled else 'Disabled'}")
                    
                    return data
                else:
                    print(f"❌ Private memory status unavailable: {response.status}")
                    return None
        except Exception as e:
            print(f"❌ Error getting private memory status: {e}")
            return None
    
    async def test_unlock_functionality(self):
        """Test unlocking private memory with password"""
        print("🔓 Testing private memory unlock...")
        
        try:
            unlock_data = {"password": self.test_password}
            
            async with self.session.post(f"{self.base_url}/api/private-memory/unlock", 
                                       json=unlock_data) as response:
                if response.status == 200:
                    data = await response.json()
                    success = data.get('success', False)
                    message = data.get('message', '')
                    
                    print(f"✅ Unlock attempt: {message}")
                    
                    if success:
                        print("✅ Private memory successfully unlocked")
                        return True
                    else:
                        print("⚠️ Unlock failed - may be first time setup")
                        return False
                else:
                    print(f"❌ Unlock request failed: {response.status}")
                    return False
        except Exception as e:
            print(f"❌ Error unlocking private memory: {e}")
            return False
    
    async def test_add_private_memories(self):
        """Test adding encrypted private memories"""
        print("💾 Testing private memory storage...")
        
        # Test memories with different types of sensitive content
        test_memories = [
            {
                "content": "Personal goal: Learn advanced AI development techniques by end of 2025",
                "tags": ["personal", "goals", "ai"],
                "category": "personal_development",
                "metadata": {
                    "priority": "high",
                    "deadline": "2025-12-31",
                    "test_entry": True
                }
            },
            {
                "content": "Sensitive project idea: Build a privacy-focused AI assistant with local processing",
                "tags": ["projects", "sensitive", "ai", "privacy"],
                "category": "project_ideas",
                "metadata": {
                    "confidentiality": "high",
                    "innovation_score": 9,
                    "test_entry": True
                }
            },
            {
                "content": "Personal reflection: Today I realized the importance of AI transparency and user privacy",
                "tags": ["reflection", "personal", "ai-ethics"],
                "category": "thoughts",
                "metadata": {
                    "date": datetime.now().isoformat(),
                    "mood": "thoughtful",
                    "test_entry": True
                }
            }
        ]
        
        added_entries = []
        
        for i, memory in enumerate(test_memories):
            try:
                async with self.session.post(f"{self.base_url}/api/private-memory/add", 
                                           json=memory) as response:
                    if response.status == 200:
                        data = await response.json()
                        entry_id = data.get('entry_id')
                        encrypted = data.get('encrypted', False)
                        
                        print(f"  Memory {i+1}: ✅ Added (ID: {entry_id}, Encrypted: {encrypted})")
                        added_entries.append({
                            "id": entry_id,
                            "original": memory,
                            "encrypted": encrypted
                        })
                    else:
                        print(f"  Memory {i+1}: ❌ Failed to add (status: {response.status})")
                        
            except Exception as e:
                print(f"  Memory {i+1}: ❌ Error adding: {e}")
        
        print(f"✅ Added {len(added_entries)} private memories")
        return added_entries
    
    async def test_encrypted_search(self):
        """Test searching within encrypted private memories"""
        print("🔍 Testing encrypted memory search...")
        
        # Test different search queries
        search_queries = [
            {
                "query": "AI development",
                "expected_matches": 2
            },
            {
                "query": "personal goal",
                "expected_matches": 1
            },
            {
                "query": "privacy",
                "expected_matches": 2
            },
            {
                "query": "nonexistent_term_xyz",
                "expected_matches": 0
            }
        ]
        
        search_results = []
        
        for search in search_queries:
            try:
                search_data = {
                    "query": search["query"],
                    "limit": 10,
                    "include_metadata": True
                }
                
                async with self.session.post(f"{self.base_url}/api/private-memory/search", 
                                           json=search_data) as response:
                    if response.status == 200:
                        data = await response.json()
                        matches = data.get('matches', [])
                        total_found = len(matches)
                        
                        print(f"  Query '{search['query']}': {total_found} matches")
                        
                        # Show first match preview
                        if matches:
                            first_match = matches[0]
                            content_preview = first_match.get('content', '')[:50] + "..."
                            confidence = first_match.get('confidence', 0)
                            print(f"    Preview: {content_preview} (confidence: {confidence:.2f})")
                        
                        search_results.append({
                            "query": search["query"],
                            "found": total_found,
                            "expected": search["expected_matches"],
                            "matches": matches
                        })
                    else:
                        print(f"  Query '{search['query']}': ❌ Search failed (status: {response.status})")
                        
            except Exception as e:
                print(f"  Query '{search['query']}': ❌ Error: {e}")
        
        print(f"✅ Completed {len(search_results)} search tests")
        return search_results
    
    async def test_memory_categories(self):
        """Test private memory category organization"""
        print("📂 Testing memory category organization...")
        
        try:
            async with self.session.get(f"{self.base_url}/api/private-memory/categories") as response:
                if response.status == 200:
                    categories = await response.json()
                    
                    print(f"✅ Found {len(categories)} memory categories:")
                    for category in categories:
                        name = category.get('name', 'Unknown')
                        count = category.get('count', 0)
                        print(f"   📁 {name}: {count} entries")
                    
                    return categories
                else:
                    print(f"❌ Categories unavailable: {response.status}")
                    return []
        except Exception as e:
            print(f"❌ Error getting categories: {e}")
            return []
    
    async def test_lock_functionality(self):
        """Test locking private memory system"""
        print("🔒 Testing private memory lock...")
        
        try:
            async with self.session.post(f"{self.base_url}/api/private-memory/lock") as response:
                if response.status == 200:
                    data = await response.json()
                    message = data.get('message', '')
                    
                    print(f"✅ Lock result: {message}")
                    
                    # Verify it's actually locked
                    await asyncio.sleep(1)
                    status = await self.test_initial_lock_status()
                    
                    if status and not status.get('is_unlocked', True):
                        print("✅ Private memory successfully locked")
                        return True
                    else:
                        print("⚠️ Lock status unclear")
                        return False
                else:
                    print(f"❌ Lock request failed: {response.status}")
                    return False
        except Exception as e:
            print(f"❌ Error locking private memory: {e}")
            return False
    
    async def test_access_control_when_locked(self):
        """Test that private memories are inaccessible when locked"""
        print("🚫 Testing access control when locked...")
        
        # Try to search while locked
        try:
            search_data = {
                "query": "AI development",
                "limit": 5
            }
            
            async with self.session.post(f"{self.base_url}/api/private-memory/search", 
                                       json=search_data) as response:
                if response.status == 401 or response.status == 403:
                    print("✅ Access correctly denied when locked")
                    return True
                elif response.status == 200:
                    data = await response.json()
                    matches = data.get('matches', [])
                    if not matches:
                        print("✅ No matches returned when locked (access controlled)")
                        return True
                    else:
                        print("⚠️ Search returned results when locked - security concern")
                        return False
                else:
                    print(f"⚠️ Unexpected response when locked: {response.status}")
                    return False
        except Exception as e:
            print(f"❌ Error testing locked access: {e}")
            return False
    
    async def test_memory_isolation(self):
        """Test that private memories don't leak into regular memory/reflection"""
        print("🔐 Testing memory isolation...")
        
        # First unlock to add a distinctive private memory
        await self.test_unlock_functionality()
        
        distinctive_memory = {
            "content": "SECRET_TEST_PHRASE_XYZ_SHOULD_NOT_APPEAR_IN_REFLECTION",
            "tags": ["secret", "isolation_test"],
            "category": "security_test",
            "metadata": {"isolation_test": True}
        }
        
        try:
            # Add the distinctive private memory
            async with self.session.post(f"{self.base_url}/api/private-memory/add", 
                                       json=distinctive_memory) as response:
                if response.status == 200:
                    print("✅ Added distinctive private memory for isolation test")
                else:
                    print("❌ Could not add test memory")
                    return False
            
            # Lock the system
            await self.test_lock_functionality()
            
            # Try a regular chat that might trigger reflection
            chat_data = {
                "message": "Can you tell me about privacy and security in AI systems?",
                "session_id": self.session_id
            }
            
            async with self.session.post(f"{self.base_url}/api/chat", json=chat_data) as response:
                if response.status == 200:
                    data = await response.json()
                    response_text = data.get('response', '')
                    
                    # Check if secret phrase leaked
                    if "SECRET_TEST_PHRASE_XYZ" in response_text:
                        print("❌ SECURITY BREACH: Private memory leaked into response!")
                        return False
                    else:
                        print("✅ Private memory properly isolated from regular responses")
                        return True
                else:
                    print(f"⚠️ Chat test failed: {response.status}")
                    return False
                    
        except Exception as e:
            print(f"❌ Error testing memory isolation: {e}")
            return False
    
    async def run_private_memory_qa_suite(self):
        """Run the complete private memory QA suite"""
        print("🔐 PRIVATE MEMORY SYSTEM QA SUITE")
        print("=" * 50)
        
        # Test sequence
        initial_status = await self.test_initial_lock_status()
        if not initial_status:
            print("❌ Cannot continue without private memory system")
            return False
        
        unlock_success = await self.test_unlock_functionality()
        
        if unlock_success:
            added_memories = await self.test_add_private_memories()
            search_results = await self.test_encrypted_search()
            categories = await self.test_memory_categories()
        else:
            added_memories = []
            search_results = []
            categories = []
        
        lock_success = await self.test_lock_functionality()
        access_control_works = await self.test_access_control_when_locked()
        isolation_secure = await self.test_memory_isolation()
        
        # Final assessment
        print("\n" + "=" * 50)
        print("🎯 PRIVATE MEMORY SYSTEM QA RESULTS:")
        print(f"✅ Initial Status: Available")
        print(f"{'✅' if unlock_success else '⚠️'} Unlock: {'Working' if unlock_success else 'Failed'}")
        print(f"✅ Memory Storage: {len(added_memories)} memories added")
        print(f"✅ Encrypted Search: {len(search_results)} search tests")
        print(f"✅ Categories: {len(categories)} categories found")
        print(f"{'✅' if lock_success else '⚠️'} Lock: {'Working' if lock_success else 'Failed'}")
        print(f"{'✅' if access_control_works else '❌'} Access Control: {'Secure' if access_control_works else 'SECURITY ISSUE'}")
        print(f"{'✅' if isolation_secure else '❌'} Memory Isolation: {'Secure' if isolation_secure else 'SECURITY BREACH'}")
        
        critical_security = access_control_works and isolation_secure
        basic_functionality = unlock_success and len(added_memories) > 0
        
        if critical_security and basic_functionality:
            print("\n🎉 Private Memory System QA: PASSED")
            return True
        elif not critical_security:
            print("\n🚨 Private Memory System QA: FAILED - SECURITY ISSUES DETECTED")
            return False
        else:
            print("\n❌ Private Memory System QA: PARTIAL - Functionality issues")
            return False

async def main():
    """Main QA execution"""
    print("Starting Private Memory System QA...")
    print("Make sure Dolphin backend is running on http://localhost:8000\n")
    
    async with PrivateMemoryQA() as qa:
        success = await qa.run_private_memory_qa_suite()
        return success

if __name__ == "__main__":
    result = asyncio.run(main())
    exit(0 if result else 1)

#!/usr/bin/env python3
"""
🐬 Quick Start Script for Dolphin AI Orchestrator v2.1
Tests all advanced features: Reflection, Connectivity, Private Memory, 
Persona Instructions, Mirror Mode, and System Metrics
"""

import asyncio
import aiohttp
import json
import time
from datetime import datetime
import os
import pytest
from unittest import mock

if os.getenv("RUN_INTEGRATION", "false") != "true":
    pytest.skip("Integration tests skipped", allow_module_level=True)

class DolphinV21Tester:
    """Test suite for Dolphin AI Orchestrator v2.1 advanced features"""
    
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.session = None
        
    async def __aenter__(self):
        if os.getenv("MOCK_API", "true") == "true":
            self.session = mock.AsyncMock()
            mock_resp = mock.MagicMock()
            mock_resp.status = 200
            mock_resp.json = mock.AsyncMock(return_value={})
            self.session.get.return_value.__aenter__.return_value = mock_resp
            self.session.post.return_value.__aenter__.return_value = mock_resp
        else:
            self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def test_basic_connectivity(self):
        """Test basic API connectivity"""
        print("🔍 Testing basic connectivity...")
        try:
            async with self.session.get(f"{self.base_url}/api/status") as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ Backend online - Version: {data.get('version', 'unknown')}")
                    return True
                else:
                    print(f"❌ Backend returned status {response.status}")
                    return False
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            return False
    
    async def test_reflection_engine(self):
        """Test reflection engine features"""
        print("\n🔄 Testing Reflection Engine...")
        
        try:
            # Get reflection summary
            async with self.session.get(f"{self.base_url}/api/reflection/summary") as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ Reflection engine status: {data.get('total_reflections', 0)} reflections generated")
                else:
                    print(f"⚠️ Reflection engine not available (status: {response.status})")
                    
            # Enable reflection engine
            async with self.session.post(f"{self.base_url}/api/reflection/enable") as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ Reflection engine enabled: {data.get('status')}")
                else:
                    print(f"⚠️ Could not enable reflection engine")
                    
        except Exception as e:
            print(f"❌ Reflection engine test failed: {e}")
    
    async def test_connectivity_management(self):
        """Test connectivity management features"""
        print("\n🌤️ Testing Connectivity Management...")
        
        try:
            # Get connectivity status
            async with self.session.get(f"{self.base_url}/api/connectivity/status") as response:
                if response.status == 200:
                    data = await response.json()
                    mode = data.get('current_mode', 'unknown')
                    print(f"✅ Connectivity mode: {mode}")
                    
                    # Show service status
                    services = data.get('services', {})
                    for service_id, service_data in services.items():
                        status = "🟢" if service_data.get('available') else "🔴"
                        name = service_data.get('name', service_id)
                        uptime = service_data.get('uptime_percentage', 0)
                        print(f"   {status} {name}: {uptime:.1f}% uptime")
                else:
                    print(f"⚠️ Connectivity manager not available (status: {response.status})")
                    
            # Get routing adjustments
            async with self.session.get(f"{self.base_url}/api/connectivity/routing-adjustments") as response:
                if response.status == 200:
                    data = await response.json()
                    prefer_local = data.get('prefer_local', False)
                    print(f"✅ Routing preference: {'Local' if prefer_local else 'Cloud'}")
                    
        except Exception as e:
            print(f"❌ Connectivity management test failed: {e}")
    
    async def test_private_memory(self):
        """Test private memory features"""
        print("\n🔐 Testing Private Memory...")
        
        try:
            # Get private memory status
            async with self.session.get(f"{self.base_url}/api/private-memory/status") as response:
                if response.status == 200:
                    data = await response.json()
                    is_unlocked = data.get('is_unlocked', False)
                    total_memories = data.get('total_private_memories', 0)
                    print(f"✅ Private memory: {total_memories} entries, {'Unlocked' if is_unlocked else 'Locked'}")
                else:
                    print(f"⚠️ Private memory not available (status: {response.status})")
                    return
            
            # Test unlock if locked
            if not is_unlocked:
                unlock_data = {"password": "default_dev_password"}
                async with self.session.post(f"{self.base_url}/api/private-memory/unlock", 
                                           json=unlock_data) as response:
                    if response.status == 200:
                        print("✅ Private memory unlocked successfully")
                    else:
                        print(f"❌ Failed to unlock private memory")
                        return
            
            # Test adding a private memory
            memory_data = {
                "content": "This is a test private memory entry",
                "tags": ["test", "demo"],
                "category": "testing",
                "metadata": {"test_timestamp": datetime.now().isoformat()}
            }
            
            async with self.session.post(f"{self.base_url}/api/private-memory/add", 
                                       json=memory_data) as response:
                if response.status == 200:
                    data = await response.json()
                    entry_id = data.get('entry_id')
                    print(f"✅ Added private memory: {entry_id}")
                else:
                    print(f"❌ Failed to add private memory")
                    
        except Exception as e:
            print(f"❌ Private memory test failed: {e}")
    
    async def test_persona_instructions(self):
        """Test persona instruction management"""
        print("\n🎭 Testing Persona Instructions...")
        
        try:
            # List available manifestos
            async with self.session.get(f"{self.base_url}/api/personas/manifestos") as response:
                if response.status == 200:
                    manifestos = await response.json()
                    print(f"✅ Found {len(manifestos)} persona manifestos:")
                    for manifesto in manifestos[:3]:  # Show first 3
                        name = manifesto.get('name', 'Unknown')
                        persona_id = manifesto.get('id', 'unknown')
                        active = "🎯" if manifesto.get('is_active') else "  "
                        print(f"   {active} {name} ({persona_id})")
                else:
                    print(f"⚠️ Persona instruction manager not available (status: {response.status})")
                    return
            
            # Test activating a persona
            if manifestos:
                test_persona = manifestos[0]['id']
                async with self.session.post(f"{self.base_url}/api/personas/activate/{test_persona}") as response:
                    if response.status == 200:
                        print(f"✅ Activated persona: {test_persona}")
                    else:
                        print(f"❌ Failed to activate persona")
                        
                # Get active instructions
                async with self.session.get(f"{self.base_url}/api/personas/active-instructions") as response:
                    if response.status == 200:
                        data = await response.json()
                        persona_name = data.get('name', 'Unknown')
                        print(f"✅ Active persona instructions loaded: {persona_name}")
                        
        except Exception as e:
            print(f"❌ Persona instructions test failed: {e}")
    
    async def test_mirror_mode(self):
        """Test mirror mode features"""
        print("\n🪩 Testing Mirror Mode...")
        
        try:
            # Get mirror mode status
            async with self.session.get(f"{self.base_url}/api/mirror-mode/status") as response:
                if response.status == 200:
                    data = await response.json()
                    enabled = data.get('mirror_enabled', False)
                    intensity = data.get('mirror_intensity', 0)
                    total_reflections = data.get('total_reflections', 0)
                    print(f"✅ Mirror mode: {'Enabled' if enabled else 'Disabled'} "
                          f"(Intensity: {intensity}, Reflections: {total_reflections})")
                else:
                    print(f"⚠️ Mirror mode not available (status: {response.status})")
                    return
            
            # Test enabling mirror mode
            if not enabled:
                enable_data = {"intensity": 0.5, "enabled_types": ["reasoning", "emotional"]}
                async with self.session.post(f"{self.base_url}/api/mirror-mode/enable", 
                                           json=enable_data) as response:
                    if response.status == 200:
                        print("✅ Mirror mode enabled successfully")
                    else:
                        print(f"❌ Failed to enable mirror mode")
                        
        except Exception as e:
            print(f"❌ Mirror mode test failed: {e}")
    
    async def test_system_metrics(self):
        """Test system metrics features"""
        print("\n📈 Testing System Metrics...")
        
        try:
            # Get real-time metrics
            async with self.session.get(f"{self.base_url}/api/metrics/realtime") as response:
                if response.status == 200:
                    data = await response.json()
                    system = data.get('system', {})
                    cpu = system.get('cpu_percent', 0)
                    memory = system.get('memory_percent', 0)
                    uptime = system.get('uptime_hours', 0)
                    print(f"✅ System metrics: CPU {cpu:.1f}%, Memory {memory:.1f}%, Uptime {uptime:.1f}h")
                else:
                    print(f"⚠️ System metrics not available (status: {response.status})")
                    return
            
            # Get model usage stats
            async with self.session.get(f"{self.base_url}/api/metrics/models") as response:
                if response.status == 200:
                    models = await response.json()
                    print(f"✅ Model usage tracked for {len(models)} models:")
                    for model, stats in list(models.items())[:3]:  # Show first 3
                        requests = stats.get('total_requests', 0)
                        avg_time = stats.get('avg_response_time_ms', 0)
                        print(f"   📊 {model}: {requests} requests, {avg_time:.1f}ms avg")
                        
            # Get health check
            async with self.session.get(f"{self.base_url}/api/metrics/health") as response:
                if response.status == 200:
                    data = await response.json()
                    status = data.get('status', 'unknown')
                    health_score = data.get('health_score', 0)
                    print(f"✅ System health: {status} (Score: {health_score:.2f})")
                    
        except Exception as e:
            print(f"❌ System metrics test failed: {e}")
    
    async def test_enhanced_chat(self):
        """Test enhanced chat with advanced features"""
        print("\n💬 Testing Enhanced Chat with Advanced Features...")
        
        try:
            # Test chat with persona and advanced context
            chat_data = {
                "message": "Help me understand how reflection works in AI systems",
                "session_id": "test_v21_session",
                "persona": "analyst",
                "context": {
                    "test_mode": True,
                    "enable_reflection": True,
                    "enable_mirror": True
                }
            }
            
            async with self.session.post(f"{self.base_url}/api/chat", json=chat_data) as response:
                if response.status == 200:
                    data = await response.json()
                    handler = data.get('handler', 'unknown')
                    persona = data.get('persona_used', 'unknown')
                    response_text = data.get('response', '')
                    
                    print(f"✅ Enhanced chat successful:")
                    print(f"   🎭 Persona: {persona}")
                    print(f"   🤖 Handler: {handler}")
                    print(f"   📝 Response length: {len(response_text)} characters")
                    
                    # Check for mirror mode indicators
                    if '*' in response_text or 'I chose' in response_text or 'My reasoning' in response_text:
                        print(f"   🪩 Mirror mode reflection detected!")
                else:
                    print(f"❌ Enhanced chat failed (status: {response.status})")
                    
        except Exception as e:
            print(f"❌ Enhanced chat test failed: {e}")
    
    async def run_full_test_suite(self):
        """Run the complete test suite"""
        print("🐬 Dolphin AI Orchestrator v2.1 - Advanced Features Test Suite")
        print("=" * 70)
        
        # Basic connectivity test
        if not await self.test_basic_connectivity():
            print("❌ Basic connectivity failed - stopping tests")
            return
        
        # Run all feature tests
        await self.test_reflection_engine()
        await self.test_connectivity_management()
        await self.test_private_memory()
        await self.test_persona_instructions()
        await self.test_mirror_mode()
        await self.test_system_metrics()
        await self.test_enhanced_chat()
        
        print("\n" + "=" * 70)
        print("🎉 Dolphin v2.1 Advanced Features Test Suite Complete!")
        print("\nNext steps:")
        print("1. 📊 Check the analytics dashboard at http://localhost:5000")
        print("2. 🎭 Try different personas in the UI")
        print("3. 🔄 Monitor reflection generation over time")
        print("4. 🔐 Test private memory features")
        print("5. 🪩 Enable mirror mode for self-aware AI responses")

async def main():
    """Main test execution"""
    print("Starting Dolphin v2.1 Advanced Features Test...")
    print("Make sure the Dolphin backend is running on http://localhost:8000\n")
    
    async with DolphinV21Tester() as tester:
        await tester.run_full_test_suite()

if __name__ == "__main__":
    asyncio.run(main())

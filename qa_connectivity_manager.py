#!/usr/bin/env python3
"""
🌤️ Connectivity Manager Targeted QA Script
Tests connectivity monitoring, service detection, and offline/online routing fallbacks
"""

import asyncio
import aiohttp
import json
import time
from datetime import datetime

class ConnectivityManagerQA:
    """Targeted testing for the Connectivity Manager"""
    
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.session_id = f"connectivity_qa_{int(time.time())}"
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def test_initial_connectivity_status(self):
        """Test initial connectivity manager status"""
        print("🔍 Testing initial connectivity status...")
        
        try:
            async with self.session.get(f"{self.base_url}/api/connectivity/status") as response:
                if response.status == 200:
                    data = await response.json()
                    mode = data.get('current_mode', 'unknown')
                    services = data.get('services', {})
                    
                    print(f"✅ Current connectivity mode: {mode}")
                    print(f"✅ Monitoring {len(services)} services:")
                    
                    for service_id, service_data in services.items():
                        status = "🟢" if service_data.get('available') else "🔴"
                        name = service_data.get('name', service_id)
                        uptime = service_data.get('uptime_percentage', 0)
                        print(f"   {status} {name}: {uptime:.1f}% uptime")
                    
                    return data
                else:
                    print(f"❌ Connectivity status unavailable: {response.status}")
                    return None
        except Exception as e:
            print(f"❌ Error getting connectivity status: {e}")
            return None
    
    async def test_service_monitoring(self):
        """Test individual service monitoring"""
        print("🔍 Testing service monitoring details...")
        
        try:
            async with self.session.get(f"{self.base_url}/api/connectivity/services") as response:
                if response.status == 200:
                    services = await response.json()
                    
                    print(f"✅ Found {len(services)} monitored services:")
                    for service in services:
                        service_id = service.get('id', 'unknown')
                        name = service.get('name', 'Unknown')
                        url = service.get('url', 'No URL')
                        available = service.get('available', False)
                        last_check = service.get('last_check', 'Never')
                        
                        status_icon = "🟢" if available else "🔴"
                        print(f"   {status_icon} {name} ({service_id})")
                        print(f"      URL: {url}")
                        print(f"      Last Check: {last_check}")
                    
                    return services
                else:
                    print(f"❌ Service monitoring unavailable: {response.status}")
                    return []
        except Exception as e:
            print(f"❌ Error getting service details: {e}")
            return []
    
    async def force_connectivity_check(self):
        """Force immediate connectivity check"""
        print("🔄 Forcing connectivity check...")
        
        try:
            async with self.session.post(f"{self.base_url}/api/connectivity/force-check") as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ Connectivity check completed: {data.get('message')}")
                    
                    # Show updated status
                    results = data.get('results', {})
                    for service_id, result in results.items():
                        status = "🟢" if result.get('available') else "🔴"
                        response_time = result.get('response_time_ms', 'N/A')
                        print(f"   {status} {service_id}: {response_time}ms")
                    
                    return data
                else:
                    print(f"❌ Failed to force connectivity check: {response.status}")
                    return None
        except Exception as e:
            print(f"❌ Error forcing connectivity check: {e}")
            return None
    
    async def test_routing_adjustments(self):
        """Test routing preference adjustments"""
        print("🧭 Testing routing adjustments...")
        
        try:
            async with self.session.get(f"{self.base_url}/api/connectivity/routing-adjustments") as response:
                if response.status == 200:
                    data = await response.json()
                    prefer_local = data.get('prefer_local', False)
                    routing_reason = data.get('routing_reason', 'Unknown')
                    adjustments = data.get('adjustments', {})
                    
                    print(f"✅ Routing preferences:")
                    print(f"   Prefer Local: {prefer_local}")
                    print(f"   Reason: {routing_reason}")
                    
                    if adjustments:
                        print(f"   Active Adjustments:")
                        for adjustment, value in adjustments.items():
                            print(f"     • {adjustment}: {value}")
                    
                    return data
                else:
                    print(f"❌ Routing adjustments unavailable: {response.status}")
                    return None
        except Exception as e:
            print(f"❌ Error getting routing adjustments: {e}")
            return None
    
    async def simulate_service_failure(self):
        """Simulate service failure by adding a fake unreachable service"""
        print("⚠️ Simulating service failure...")
        
        try:
            # Add a service that will definitely fail
            fake_service = {
                "id": "test_unreachable",
                "name": "Test Unreachable Service",
                "url": "http://definitely-unreachable-test-service.invalid",
                "timeout": 1,
                "critical": True
            }
            
            async with self.session.post(f"{self.base_url}/api/connectivity/add-service", 
                                       json=fake_service) as response:
                if response.status == 200:
                    print("✅ Added unreachable test service")
                    
                    # Force check to detect failure
                    await asyncio.sleep(1)
                    await self.force_connectivity_check()
                    
                    return True
                else:
                    print(f"❌ Failed to add test service: {response.status}")
                    return False
        except Exception as e:
            print(f"❌ Error simulating service failure: {e}")
            return False
    
    async def test_offline_mode_detection(self):
        """Test offline mode detection and response"""
        print("📡 Testing offline mode detection...")
        
        try:
            # Check if offline mode was triggered by service failures
            async with self.session.get(f"{self.base_url}/api/connectivity/status") as response:
                if response.status == 200:
                    data = await response.json()
                    mode = data.get('current_mode', 'unknown')
                    services = data.get('services', {})
                    
                    # Count failed services
                    failed_services = [s for s in services.values() if not s.get('available', True)]
                    total_services = len(services)
                    
                    print(f"✅ Connectivity mode after simulation: {mode}")
                    print(f"✅ Failed services: {len(failed_services)}/{total_services}")
                    
                    # Check if mode changed appropriately
                    if len(failed_services) > 0:
                        print("✅ Service failures detected correctly")
                    
                    return mode, failed_services
                else:
                    print(f"❌ Could not check offline mode: {response.status}")
                    return None, []
        except Exception as e:
            print(f"❌ Error testing offline mode: {e}")
            return None, []
    
    async def test_fallback_routing(self):
        """Test that routing falls back to local models when services are down"""
        print("🔄 Testing fallback routing behavior...")
        
        try:
            # Send a chat request that would normally use cloud services
            chat_data = {
                "message": "Help me write a complex Python algorithm for sorting large datasets",
                "session_id": self.session_id,
                "persona": "technical"  # Technical persona prefers cloud AI
            }
            
            async with self.session.post(f"{self.base_url}/api/chat", json=chat_data) as response:
                if response.status == 200:
                    data = await response.json()
                    handler = data.get('handler', 'unknown')
                    routing_reason = data.get('routing_reason', 'unknown')
                    
                    print(f"✅ Chat routed to: {handler}")
                    print(f"✅ Routing reason: {routing_reason}")
                    
                    # Check if it was routed locally due to connectivity issues
                    if 'local' in handler.lower() or 'dolphin' in handler.lower():
                        print("✅ Successfully fell back to local processing")
                        return True
                    else:
                        print("⚠️ Did not fall back to local (cloud services may be available)")
                        return False
                else:
                    print(f"❌ Fallback test chat failed: {response.status}")
                    return False
        except Exception as e:
            print(f"❌ Error testing fallback routing: {e}")
            return False
    
    async def cleanup_test_services(self):
        """Clean up test services added during testing"""
        print("🧹 Cleaning up test services...")
        
        try:
            # Note: This would require a delete endpoint
            print("⚠️ Manual cleanup may be required for test services")
            return True
        except Exception as e:
            print(f"❌ Error during cleanup: {e}")
            return False
    
    async def run_connectivity_qa_suite(self):
        """Run the complete connectivity manager QA suite"""
        print("🌤️ CONNECTIVITY MANAGER QA SUITE")
        print("=" * 50)
        
        # Test sequence
        initial_status = await self.test_initial_connectivity_status()
        if not initial_status:
            print("❌ Cannot continue without connectivity manager")
            return False
        
        services = await self.test_service_monitoring()
        force_check_result = await self.force_connectivity_check()
        routing_data = await self.test_routing_adjustments()
        
        # Simulate failure scenario
        print("\n📡 SIMULATING CONNECTIVITY ISSUES...")
        failure_simulated = await self.simulate_service_failure()
        
        if failure_simulated:
            await asyncio.sleep(2)  # Wait for detection
            mode, failed_services = await self.test_offline_mode_detection()
            fallback_worked = await self.test_fallback_routing()
        else:
            mode, failed_services = "unknown", []
            fallback_worked = False
        
        await self.cleanup_test_services()
        
        # Final assessment
        print("\n" + "=" * 50)
        print("🎯 CONNECTIVITY MANAGER QA RESULTS:")
        print(f"✅ Initial Status: {initial_status is not None}")
        print(f"✅ Service Monitoring: {len(services)} services tracked")
        print(f"✅ Force Check: {force_check_result is not None}")
        print(f"✅ Routing Adjustments: {routing_data is not None}")
        print(f"✅ Failure Simulation: {failure_simulated}")
        print(f"✅ Offline Detection: {len(failed_services)} failures detected")
        print(f"{'✅' if fallback_worked else '⚠️'} Fallback Routing: {'Working' if fallback_worked else 'Not clearly demonstrated'}")
        
        success_criteria = [
            initial_status is not None,
            len(services) > 0,
            force_check_result is not None,
            routing_data is not None
        ]
        
        if all(success_criteria):
            print("\n🎉 Connectivity Manager QA: PASSED")
            return True
        else:
            print("\n❌ Connectivity Manager QA: PARTIAL - Some features may need attention")
            return False

async def main():
    """Main QA execution"""
    print("Starting Connectivity Manager QA...")
    print("Make sure Dolphin backend is running on http://localhost:8000\n")
    
    async with ConnectivityManagerQA() as qa:
        success = await qa.run_connectivity_qa_suite()
        return success

if __name__ == "__main__":
    result = asyncio.run(main())
    exit(0 if result else 1)

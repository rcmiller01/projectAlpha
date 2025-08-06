#!/usr/bin/env python3
"""
System Metrics Quick Test
Basic functionality test without Unicode characters
"""

import asyncio
import aiohttp
import json

async def test_system_metrics():
    print("Testing System Metrics...")
    
    try:
        async with aiohttp.ClientSession() as session:
            # Test real-time metrics
            async with session.get("http://localhost:8000/api/metrics/realtime") as response:
                if response.status == 200:
                    data = await response.json()
                    print("SUCCESS: Real-time metrics available")
                    
                    system = data.get('system', {})
                    if system:
                        cpu = system.get('cpu_percent', 0)
                        memory = system.get('memory_percent', 0)
                        print(f"  CPU: {cpu:.1f}%, Memory: {memory:.1f}%")
                    return True
                else:
                    print(f"FAILED: Status {response.status}")
                    return False
    except Exception as e:
        print(f"ERROR: {e}")
        return False

if __name__ == "__main__":
    result = asyncio.run(test_system_metrics())
    exit(0 if result else 1)

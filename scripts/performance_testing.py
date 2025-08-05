#!/usr/bin/env python3
"""
System Performance Testing Script
Tests backend API performance, database operations, and system reliability
"""

import asyncio
import aiohttp
import time
import json
import statistics
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import psutil
import pytest

class PerformanceTester:
    def __init__(self, base_url="http://localhost:5000"):
        self.base_url = base_url
        self.results = {
            'api_tests': [],
            'load_tests': [],
            'memory_tests': [],
            'database_tests': []
        }
        
    async def test_api_endpoints(self):
        """Test all major API endpoints for response time and reliability"""
        print("🚀 Testing API Performance...")
        
        endpoints = [
            # Core endpoints
            {'method': 'GET', 'path': '/api/health', 'name': 'Health Check'},
            {'method': 'GET', 'path': '/api/personas', 'name': 'Get Personas'},
            {'method': 'GET', 'path': '/api/memories/recent', 'name': 'Recent Memories'},
            
            # Creative Evolution endpoints
            {'method': 'POST', 'path': '/api/creative/generate', 'name': 'Generate Content', 
             'data': {'type': 'story', 'style': 'whimsical', 'theme': 'adventure'}},
            {'method': 'GET', 'path': '/api/creative/gallery', 'name': 'Creative Gallery'},
            {'method': 'GET', 'path': '/api/creative/personality/traits', 'name': 'Personality Traits'},
            
            # Romantic connection endpoints
            {'method': 'GET', 'path': '/api/romantic/moments', 'name': 'Romantic Moments'},
            {'method': 'POST', 'path': '/api/romantic/intimacy', 'name': 'Intimacy Interaction',
             'data': {'type': 'emotional', 'intensity': 'medium'}},
            
            # Autonomy endpoints
            {'method': 'GET', 'path': '/api/autonomy/initiatives', 'name': 'Autonomy Initiatives'},
            {'method': 'GET', 'path': '/api/autonomy/goals', 'name': 'Goal System'},
            
            # Memory system
            {'method': 'POST', 'path': '/api/memories/store', 'name': 'Store Memory',
             'data': {'content': 'Test memory', 'type': 'conversation', 'importance': 5}},
            
            # Real-time features
            {'method': 'GET', 'path': '/api/realtime/status', 'name': 'Realtime Status'},
        ]
        
        async with aiohttp.ClientSession() as session:
            for endpoint in endpoints:
                start_time = time.time()
                try:
                    if endpoint['method'] == 'GET':
                        async with session.get(f"{self.base_url}{endpoint['path']}") as response:
                            status_code = response.status
                            response_time = (time.time() - start_time) * 1000
                    else:
                        data = endpoint.get('data', {})
                        async with session.post(f"{self.base_url}{endpoint['path']}", 
                                              json=data) as response:
                            status_code = response.status
                            response_time = (time.time() - start_time) * 1000
                    
                    self.results['api_tests'].append({
                        'endpoint': endpoint['name'],
                        'response_time': response_time,
                        'status_code': status_code,
                        'success': 200 <= status_code < 300,
                        'timestamp': datetime.now().isoformat()
                    })
                    
                    status = "✅" if 200 <= status_code < 300 else "❌"
                    print(f"  {status} {endpoint['name']}: {response_time:.2f}ms (HTTP {status_code})")
                    
                except Exception as e:
                    print(f"  ❌ {endpoint['name']}: Failed - {str(e)}")
                    self.results['api_tests'].append({
                        'endpoint': endpoint['name'],
                        'response_time': None,
                        'status_code': None,
                        'success': False,
                        'error': str(e),
                        'timestamp': datetime.now().isoformat()
                    })
                
                # Small delay between requests
                await asyncio.sleep(0.1)
    
    async def load_test(self, concurrent_users=50, requests_per_user=10):
        """Test system under load with multiple concurrent users"""
        print(f"🔥 Load Testing: {concurrent_users} users, {requests_per_user} requests each...")
        
        async def user_simulation(session, user_id):
            """Simulate a single user's behavior"""
            user_results = []
            
            for request_num in range(requests_per_user):
                start_time = time.time()
                try:
                    # Simulate typical user flow
                    endpoints = [
                        '/api/personas',
                        '/api/memories/recent',
                        '/api/creative/gallery',
                        '/api/autonomy/status'
                    ]
                    
                    endpoint = endpoints[request_num % len(endpoints)]
                    async with session.get(f"{self.base_url}{endpoint}") as response:
                        response_time = (time.time() - start_time) * 1000
                        user_results.append({
                            'user_id': user_id,
                            'request_num': request_num,
                            'endpoint': endpoint,
                            'response_time': response_time,
                            'status_code': response.status,
                            'success': 200 <= response.status < 300
                        })
                        
                except Exception as e:
                    user_results.append({
                        'user_id': user_id,
                        'request_num': request_num,
                        'error': str(e),
                        'success': False
                    })
                
                # Simulate user think time
                await asyncio.sleep(0.1)
            
            return user_results
        
        # Run concurrent user simulations
        async with aiohttp.ClientSession() as session:
            tasks = [user_simulation(session, user_id) 
                    for user_id in range(concurrent_users)]
            
            start_test = time.time()
            results = await asyncio.gather(*tasks)
            total_time = time.time() - start_test
            
            # Flatten results
            all_results = [result for user_results in results for result in user_results]
            self.results['load_tests'] = all_results
            
            # Calculate statistics
            successful_requests = [r for r in all_results if r.get('success', False)]
            response_times = [r['response_time'] for r in successful_requests 
                            if 'response_time' in r]
            
            if response_times:
                avg_response = statistics.mean(response_times)
                p95_response = statistics.quantiles(response_times, n=20)[18]  # 95th percentile
                success_rate = len(successful_requests) / len(all_results) * 100
                requests_per_second = len(all_results) / total_time
                
                print(f"  📊 Load Test Results:")
                print(f"    Total requests: {len(all_results)}")
                print(f"    Success rate: {success_rate:.1f}%")
                print(f"    Requests/second: {requests_per_second:.1f}")
                print(f"    Avg response time: {avg_response:.2f}ms")
                print(f"    95th percentile: {p95_response:.2f}ms")
            else:
                print("  ❌ No successful requests in load test")
    
    def test_memory_usage(self):
        """Monitor system memory usage during operations"""
        print("🧠 Testing Memory Usage...")
        
        # Get current process memory
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Monitor system memory
        system_memory = psutil.virtual_memory()
        
        self.results['memory_tests'] = {
            'initial_process_memory_mb': initial_memory,
            'system_memory_percent': system_memory.percent,
            'system_memory_available_gb': system_memory.available / 1024 / 1024 / 1024,
            'timestamp': datetime.now().isoformat()
        }
        
        print(f"  Process memory usage: {initial_memory:.1f} MB")
        print(f"  System memory usage: {system_memory.percent:.1f}%")
        print(f"  Available memory: {system_memory.available / 1024 / 1024 / 1024:.1f} GB")
    
    async def test_database_performance(self):
        """Test database query performance"""
        print("🗄️ Testing Database Performance...")
        
        database_tests = [
            {'name': 'Memory Retrieval', 'endpoint': '/api/memories/search?q=test'},
            {'name': 'Persona Loading', 'endpoint': '/api/personas/active'},
            {'name': 'Creative Content Query', 'endpoint': '/api/creative/content/recent'},
            {'name': 'Autonomy Data', 'endpoint': '/api/autonomy/learning/progress'},
        ]
        
        async with aiohttp.ClientSession() as session:
            for test in database_tests:
                start_time = time.time()
                try:
                    async with session.get(f"{self.base_url}{test['endpoint']}") as response:
                        query_time = (time.time() - start_time) * 1000
                        
                        self.results['database_tests'].append({
                            'test_name': test['name'],
                            'query_time': query_time,
                            'status_code': response.status,
                            'success': 200 <= response.status < 300,
                            'timestamp': datetime.now().isoformat()
                        })
                        
                        status = "✅" if 200 <= response.status < 300 else "❌"
                        print(f"  {status} {test['name']}: {query_time:.2f}ms")
                        
                except Exception as e:
                    print(f"  ❌ {test['name']}: Failed - {str(e)}")
                    self.results['database_tests'].append({
                        'test_name': test['name'],
                        'query_time': None,
                        'success': False,
                        'error': str(e),
                        'timestamp': datetime.now().isoformat()
                    })
    
    def generate_report(self):
        """Generate comprehensive performance report"""
        print("\n📋 Performance Test Report")
        print("=" * 50)
        
        # API Performance Summary
        api_successful = [t for t in self.results['api_tests'] if t.get('success', False)]
        if api_successful:
            response_times = [t['response_time'] for t in api_successful]
            avg_api_time = statistics.mean(response_times)
            
            print(f"\n🚀 API Performance:")
            print(f"  Successful endpoints: {len(api_successful)}/{len(self.results['api_tests'])}")
            print(f"  Average response time: {avg_api_time:.2f}ms")
            print(f"  Fastest endpoint: {min(response_times):.2f}ms")
            print(f"  Slowest endpoint: {max(response_times):.2f}ms")
            
            # Check if meets performance targets
            target_response_time = 200  # ms
            fast_endpoints = len([t for t in response_times if t <= target_response_time])
            print(f"  Endpoints meeting <{target_response_time}ms target: {fast_endpoints}/{len(response_times)}")
        
        # Load Test Summary
        if self.results['load_tests']:
            load_successful = [t for t in self.results['load_tests'] if t.get('success', False)]
            load_response_times = [t['response_time'] for t in load_successful 
                                 if 'response_time' in t]
            
            if load_response_times:
                print(f"\n🔥 Load Test Performance:")
                print(f"  Success rate: {len(load_successful)/len(self.results['load_tests'])*100:.1f}%")
                print(f"  Average response under load: {statistics.mean(load_response_times):.2f}ms")
                
                if len(load_response_times) > 1:
                    p95 = statistics.quantiles(load_response_times, n=20)[18]
                    print(f"  95th percentile response time: {p95:.2f}ms")
        
        # Database Performance Summary
        db_successful = [t for t in self.results['database_tests'] if t.get('success', False)]
        if db_successful:
            db_times = [t['query_time'] for t in db_successful]
            print(f"\n🗄️ Database Performance:")
            print(f"  Successful queries: {len(db_successful)}/{len(self.results['database_tests'])}")
            print(f"  Average query time: {statistics.mean(db_times):.2f}ms")
        
        # Memory Usage Summary
        if self.results['memory_tests']:
            memory = self.results['memory_tests']
            print(f"\n🧠 Memory Usage:")
            print(f"  Process memory: {memory['initial_process_memory_mb']:.1f} MB")
            print(f"  System memory usage: {memory['system_memory_percent']:.1f}%")
        
        # Overall Assessment
        print(f"\n🎯 Overall Assessment:")
        
        # Performance criteria
        api_success_rate = len(api_successful) / len(self.results['api_tests']) * 100 if self.results['api_tests'] else 0
        load_success_rate = len(load_successful) / len(self.results['load_tests']) * 100 if self.results['load_tests'] else 100
        
        print(f"  API Success Rate: {api_success_rate:.1f}% (Target: >95%)")
        print(f"  Load Test Success Rate: {load_success_rate:.1f}% (Target: >90%)")
        
        if api_successful:
            avg_response = statistics.mean([t['response_time'] for t in api_successful])
            print(f"  Average Response Time: {avg_response:.1f}ms (Target: <200ms)")
            
            if avg_response <= 200 and api_success_rate >= 95 and load_success_rate >= 90:
                print("  ✅ System meets all performance targets!")
            else:
                print("  ⚠️ Some performance targets not met - optimization needed")
        
        return self.results
    
    def save_results(self, filename=None):
        """Save test results to JSON file"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"performance_test_results_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n💾 Results saved to: {filename}")

async def main():
    """Run complete performance test suite"""
    print("🧪 AI Companion System Performance Testing")
    print("=" * 50)
    
    tester = PerformanceTester()
    
    try:
        # Run all tests
        await tester.test_api_endpoints()
        await tester.load_test(concurrent_users=25, requests_per_user=5)  # Lighter load for testing
        tester.test_memory_usage()
        await tester.test_database_performance()
        
        # Generate and save report
        tester.generate_report()
        tester.save_results()
        
        print("\n✅ Performance testing completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Performance testing failed: {str(e)}")
        return False
    
    return True

if __name__ == "__main__":
    # Check if system is running
    print("Checking if system is available...")
    
    # Run the performance tests
    success = asyncio.run(main())
    
    if success:
        print("\n🎉 All performance tests completed!")
    else:
        print("\n❌ Performance testing encountered errors")
        exit(1)

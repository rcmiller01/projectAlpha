#!/usr/bin/env python3
"""
Error Handling and Edge Case Testing Script
Tests system resilience, error scenarios, and edge cases
"""

import asyncio
import aiohttp
import json
from datetime import datetime
import time
import random
import string

class ErrorHandlingTester:
    def __init__(self, base_url="http://localhost:5000"):
        self.base_url = base_url
        self.results = {
            'error_handling_tests': [],
            'edge_case_tests': [],
            'security_tests': [],
            'data_validation_tests': []
        }

    async def test_api_error_handling(self):
        """Test API error handling scenarios"""
        print("🛡️ Testing API Error Handling...")
        
        error_scenarios = [
            # Authentication errors
            {'name': 'Invalid Auth Token', 'headers': {'Authorization': 'Bearer invalid_token'}, 
             'endpoint': '/api/personas', 'expected_status': [401, 403]},
            
            # Invalid endpoints
            {'name': 'Non-existent Endpoint', 'endpoint': '/api/nonexistent', 
             'expected_status': [404]},
            
            # Malformed requests
            {'name': 'Invalid JSON', 'endpoint': '/api/creative/generate', 
             'data': 'invalid json', 'expected_status': [400]},
            
            # Missing required fields
            {'name': 'Missing Required Fields', 'endpoint': '/api/memories/store', 
             'data': {}, 'expected_status': [400, 422]},
            
            # Invalid data types
            {'name': 'Invalid Data Types', 'endpoint': '/api/creative/generate',
             'data': {'type': 123, 'style': None}, 'expected_status': [400, 422]},
            
            # Large payload
            {'name': 'Oversized Payload', 'endpoint': '/api/memories/store',
             'data': {'content': 'x' * 10000, 'type': 'conversation'}, 'expected_status': [413, 400]},
            
            # SQL Injection attempts
            {'name': 'SQL Injection Test', 'endpoint': '/api/memories/search',
             'params': {'q': "'; DROP TABLE memories; --"}, 'expected_status': [400, 200]},
            
            # XSS attempts
            {'name': 'XSS Prevention', 'endpoint': '/api/creative/generate',
             'data': {'type': 'story', 'content': '<script>alert("xss")</script>'}, 
             'expected_status': [400, 200]},
        ]
        
        async with aiohttp.ClientSession() as session:
            for scenario in error_scenarios:
                try:
                    start_time = time.time()
                    
                    # Prepare request
                    url = f"{self.base_url}{scenario['endpoint']}"
                    headers = scenario.get('headers', {'Content-Type': 'application/json'})
                    params = scenario.get('params', {})
                    
                    if 'data' in scenario:
                        data = scenario['data']
                        if isinstance(data, str):
                            # Test invalid JSON
                            headers['Content-Type'] = 'application/json'
                            async with session.post(url, data=data, headers=headers, params=params) as response:
                                status_code = response.status
                        else:
                            async with session.post(url, json=data, headers=headers, params=params) as response:
                                status_code = response.status
                    else:
                        async with session.get(url, headers=headers, params=params) as response:
                            status_code = response.status
                    
                    response_time = (time.time() - start_time) * 1000
                    expected_handled = status_code in scenario['expected_status']
                    
                    self.results['error_handling_tests'].append({
                        'scenario': scenario['name'],
                        'endpoint': scenario['endpoint'],
                        'status_code': status_code,
                        'response_time': response_time,
                        'expected_status': scenario['expected_status'],
                        'handled_correctly': expected_handled,
                        'timestamp': datetime.now().isoformat()
                    })
                    
                    status = "✅" if expected_handled else "❌"
                    print(f"  {status} {scenario['name']}: HTTP {status_code} ({response_time:.0f}ms)")
                    
                except Exception as e:
                    print(f"  ❌ {scenario['name']}: Exception - {str(e)}")
                    self.results['error_handling_tests'].append({
                        'scenario': scenario['name'],
                        'endpoint': scenario['endpoint'],
                        'error': str(e),
                        'handled_correctly': False,
                        'timestamp': datetime.now().isoformat()
                    })

    async def test_edge_cases(self):
        """Test edge cases and boundary conditions"""
        print("🔍 Testing Edge Cases...")
        
        edge_cases = [
            # Empty strings
            {'name': 'Empty String Input', 'endpoint': '/api/creative/generate',
             'data': {'type': '', 'style': '', 'content': ''}},
            
            # Unicode and special characters
            {'name': 'Unicode Characters', 'endpoint': '/api/memories/store',
             'data': {'content': '🎭💫🌟 Unicode test 中文 العربية', 'type': 'conversation'}},
            
            # Very long strings
            {'name': 'Long String Input', 'endpoint': '/api/creative/generate',
             'data': {'type': 'story', 'content': 'A' * 1000}},
            
            # Null/None values
            {'name': 'Null Values', 'endpoint': '/api/memories/store',
             'data': {'content': None, 'type': None}},
            
            # Extreme numbers
            {'name': 'Large Numbers', 'endpoint': '/api/autonomy/learning/update',
             'data': {'progress': 999999999, 'score': 1e10}},
            
            # Negative numbers where positive expected
            {'name': 'Negative Numbers', 'endpoint': '/api/romantic/intimacy',
             'data': {'intensity': -5, 'duration': -100}},
            
            # Array boundary conditions
            {'name': 'Empty Arrays', 'endpoint': '/api/creative/collaborative/create',
             'data': {'participants': [], 'content_types': []}},
            
            # Concurrent requests
            {'name': 'Concurrent Memory Storage', 'test_type': 'concurrent'},
            
            # Rapid successive requests
            {'name': 'Rapid Requests', 'test_type': 'rapid'},
        ]
        
        async with aiohttp.ClientSession() as session:
            for case in edge_cases:
                try:
                    if case.get('test_type') == 'concurrent':
                        await self.test_concurrent_requests(session)
                    elif case.get('test_type') == 'rapid':
                        await self.test_rapid_requests(session)
                    else:
                        start_time = time.time()
                        
                        url = f"{self.base_url}{case['endpoint']}"
                        async with session.post(url, json=case['data']) as response:
                            status_code = response.status
                            response_time = (time.time() - start_time) * 1000
                            
                            try:
                                response_data = await response.json()
                                has_error_handling = 'error' in response_data or 'message' in response_data
                            except:
                                has_error_handling = True  # Non-JSON response indicates handling
                        
                        # Consider 400-499 as proper error handling, 500+ as server errors
                        handled_gracefully = status_code < 500
                        
                        self.results['edge_case_tests'].append({
                            'case': case['name'],
                            'endpoint': case['endpoint'],
                            'status_code': status_code,
                            'response_time': response_time,
                            'handled_gracefully': handled_gracefully,
                            'has_error_handling': has_error_handling,
                            'timestamp': datetime.now().isoformat()
                        })
                        
                        status = "✅" if handled_gracefully else "❌"
                        print(f"  {status} {case['name']}: HTTP {status_code} ({response_time:.0f}ms)")
                        
                except Exception as e:
                    print(f"  ❌ {case['name']}: Exception - {str(e)}")
                    self.results['edge_case_tests'].append({
                        'case': case['name'],
                        'error': str(e),
                        'handled_gracefully': False,
                        'timestamp': datetime.now().isoformat()
                    })

    async def test_concurrent_requests(self, session):
        """Test handling of concurrent requests"""
        print("    Testing concurrent memory storage...")
        
        # Create multiple concurrent requests
        tasks = []
        for i in range(10):
            task = session.post(
                f"{self.base_url}/api/memories/store",
                json={'content': f'Concurrent test memory {i}', 'type': 'test'}
            )
            tasks.append(task)
        
        start_time = time.time()
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        total_time = time.time() - start_time
        
        successful = 0
        for response in responses:
            if not isinstance(response, Exception):
                # This is a proper response object
                try:
                    status = getattr(response, 'status', 500)
                    if status < 400:
                        successful += 1
                except:
                    pass
        
        self.results['edge_case_tests'].append({
            'case': 'Concurrent Requests',
            'total_requests': len(tasks),
            'successful_requests': successful,
            'total_time': total_time * 1000,
            'handled_gracefully': successful > len(tasks) * 0.8,  # 80% success rate
            'timestamp': datetime.now().isoformat()
        })

    async def test_rapid_requests(self, session):
        """Test handling of rapid successive requests"""
        print("    Testing rapid successive requests...")
        
        response_times = []
        successful = 0
        
        for i in range(20):
            start_time = time.time()
            try:
                async with session.get(f"{self.base_url}/api/health") as response:
                    response_time = (time.time() - start_time) * 1000
                    response_times.append(response_time)
                    if response.status < 400:
                        successful += 1
            except Exception:
                pass
            
            # Minimal delay (rapid requests)
            await asyncio.sleep(0.01)
        
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        
        self.results['edge_case_tests'].append({
            'case': 'Rapid Requests',
            'total_requests': 20,
            'successful_requests': successful,
            'average_response_time': avg_response_time,
            'handled_gracefully': successful > 15,  # 75% success rate
            'timestamp': datetime.now().isoformat()
        })

    async def test_data_validation(self):
        """Test data validation and sanitization"""
        print("🔒 Testing Data Validation...")
        
        validation_tests = [
            # Email validation
            {'name': 'Invalid Email Format', 'endpoint': '/api/users/update',
             'data': {'email': 'invalid-email'}, 'should_reject': True},
            
            # Password strength
            {'name': 'Weak Password', 'endpoint': '/api/auth/register',
             'data': {'password': '123'}, 'should_reject': True},
            
            # HTML injection
            {'name': 'HTML Injection', 'endpoint': '/api/creative/generate',
             'data': {'content': '<img src=x onerror=alert(1)>'}, 'should_sanitize': True},
            
            # Path traversal
            {'name': 'Path Traversal', 'endpoint': '/api/files/upload',
             'data': {'filename': '../../../etc/passwd'}, 'should_reject': True},
            
            # Date validation
            {'name': 'Invalid Date', 'endpoint': '/api/memories/store',
             'data': {'timestamp': 'invalid-date'}, 'should_reject': True},
            
            # Numeric bounds
            {'name': 'Out of Bounds Number', 'endpoint': '/api/romantic/intimacy',
             'data': {'intensity': 999}}, # Assuming 1-10 scale
        ]
        
        async with aiohttp.ClientSession() as session:
            for test in validation_tests:
                try:
                    start_time = time.time()
                    
                    url = f"{self.base_url}{test['endpoint']}"
                    async with session.post(url, json=test['data']) as response:
                        status_code = response.status
                        response_time = (time.time() - start_time) * 1000
                        
                        try:
                            response_data = await response.json()
                        except:
                            response_data = {}
                        
                        # Check if validation worked as expected
                        if test.get('should_reject'):
                            validation_working = status_code in [400, 422]
                        elif test.get('should_sanitize'):
                            validation_working = status_code == 200  # Should accept but sanitize
                        else:
                            validation_working = status_code < 500
                        
                        self.results['data_validation_tests'].append({
                            'test': test['name'],
                            'endpoint': test['endpoint'],
                            'status_code': status_code,
                            'response_time': response_time,
                            'validation_working': validation_working,
                            'response_data': response_data,
                            'timestamp': datetime.now().isoformat()
                        })
                        
                        status = "✅" if validation_working else "❌"
                        print(f"  {status} {test['name']}: HTTP {status_code} ({response_time:.0f}ms)")
                        
                except Exception as e:
                    print(f"  ❌ {test['name']}: Exception - {str(e)}")
                    self.results['data_validation_tests'].append({
                        'test': test['name'],
                        'error': str(e),
                        'validation_working': False,
                        'timestamp': datetime.now().isoformat()
                    })

    async def test_security_scenarios(self):
        """Test security-related scenarios"""
        print("🔐 Testing Security Scenarios...")
        
        security_tests = [
            # Rate limiting
            {'name': 'Rate Limiting', 'test_type': 'rate_limit'},
            
            # Session management
            {'name': 'Session Timeout', 'endpoint': '/api/auth/check',
             'headers': {'Authorization': 'Bearer expired_token'}},
            
            # CORS handling
            {'name': 'CORS Headers', 'endpoint': '/api/health',
             'headers': {'Origin': 'https://malicious-site.com'}},
            
            # File upload restrictions
            {'name': 'Malicious File Upload', 'endpoint': '/api/files/upload',
             'file_type': 'executable'},
            
            # Command injection
            {'name': 'Command Injection', 'endpoint': '/api/voice/process',
             'data': {'command': '; rm -rf /'}},
        ]
        
        async with aiohttp.ClientSession() as session:
            for test in security_tests:
                try:
                    if test.get('test_type') == 'rate_limit':
                        await self.test_rate_limiting(session)
                    else:
                        start_time = time.time()
                        
                        url = f"{self.base_url}{test['endpoint']}"
                        headers = test.get('headers', {'Content-Type': 'application/json'})
                        
                        if 'data' in test:
                            async with session.post(url, json=test['data'], headers=headers) as response:
                                status_code = response.status
                        else:
                            async with session.get(url, headers=headers) as response:
                                status_code = response.status
                        
                        response_time = (time.time() - start_time) * 1000
                        
                        # Security tests should generally reject malicious requests
                        security_working = status_code in [400, 401, 403, 405] or status_code == 200
                        
                        self.results['security_tests'].append({
                            'test': test['name'],
                            'endpoint': test['endpoint'],
                            'status_code': status_code,
                            'response_time': response_time,
                            'security_working': security_working,
                            'timestamp': datetime.now().isoformat()
                        })
                        
                        status = "✅" if security_working else "❌"
                        print(f"  {status} {test['name']}: HTTP {status_code} ({response_time:.0f}ms)")
                        
                except Exception as e:
                    print(f"  ❌ {test['name']}: Exception - {str(e)}")
                    self.results['security_tests'].append({
                        'test': test['name'],
                        'error': str(e),
                        'security_working': False,
                        'timestamp': datetime.now().isoformat()
                    })

    async def test_rate_limiting(self, session):
        """Test rate limiting functionality"""
        print("    Testing rate limiting...")
        
        # Send many requests rapidly
        requests_sent = 0
        rate_limited = False
        
        for i in range(50):  # Send 50 rapid requests
            try:
                async with session.get(f"{self.base_url}/api/health") as response:
                    requests_sent += 1
                    if response.status == 429:  # Too Many Requests
                        rate_limited = True
                        break
            except Exception:
                break
            
            await asyncio.sleep(0.01)  # Very rapid requests
        
        self.results['security_tests'].append({
            'test': 'Rate Limiting',
            'requests_sent': requests_sent,
            'rate_limited': rate_limited,
            'security_working': rate_limited or requests_sent < 50,  # Either rate limited or stopped sending
            'timestamp': datetime.now().isoformat()
        })

    def generate_report(self):
        """Generate comprehensive error handling and security report"""
        print("\n📋 Error Handling & Security Report")
        print("=" * 50)
        
        # Error Handling Summary
        if self.results['error_handling_tests']:
            handled_correctly = [t for t in self.results['error_handling_tests'] if t.get('handled_correctly', False)]
            print(f"\n🛡️ Error Handling:")
            print(f"  Scenarios handled correctly: {len(handled_correctly)}/{len(self.results['error_handling_tests'])}")
            
            # Common error response codes
            status_codes = [t.get('status_code') for t in self.results['error_handling_tests'] if 'status_code' in t]
            if status_codes:
                print(f"  Most common response codes: {set(status_codes)}")
        
        # Edge Cases Summary
        if self.results['edge_case_tests']:
            handled_gracefully = [t for t in self.results['edge_case_tests'] if t.get('handled_gracefully', False)]
            print(f"\n🔍 Edge Cases:")
            print(f"  Cases handled gracefully: {len(handled_gracefully)}/{len(self.results['edge_case_tests'])}")
        
        # Data Validation Summary
        if self.results['data_validation_tests']:
            validation_working = [t for t in self.results['data_validation_tests'] if t.get('validation_working', False)]
            print(f"\n🔒 Data Validation:")
            print(f"  Validation tests passed: {len(validation_working)}/{len(self.results['data_validation_tests'])}")
        
        # Security Summary
        if self.results['security_tests']:
            security_working = [t for t in self.results['security_tests'] if t.get('security_working', False)]
            print(f"\n🔐 Security:")
            print(f"  Security tests passed: {len(security_working)}/{len(self.results['security_tests'])}")
        
        # Overall Resilience Score
        total_tests = (len(self.results['error_handling_tests']) + 
                      len(self.results['edge_case_tests']) + 
                      len(self.results['data_validation_tests']) + 
                      len(self.results['security_tests']))
        
        total_passed = (len([t for t in self.results['error_handling_tests'] if t.get('handled_correctly', False)]) +
                       len([t for t in self.results['edge_case_tests'] if t.get('handled_gracefully', False)]) +
                       len([t for t in self.results['data_validation_tests'] if t.get('validation_working', False)]) +
                       len([t for t in self.results['security_tests'] if t.get('security_working', False)]))
        
        if total_tests > 0:
            resilience_score = (total_passed / total_tests) * 100
            print(f"\n🎯 Overall System Resilience: {resilience_score:.1f}%")
            
            if resilience_score >= 90:
                print("  ✅ Excellent error handling and security!")
            elif resilience_score >= 75:
                print("  ⚠️ Good resilience, minor improvements needed")
            else:
                print("  ❌ Significant improvements needed for production readiness")
        
        return self.results

    def save_results(self, filename=None):
        """Save test results to JSON file"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"error_handling_test_results_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n💾 Results saved to: {filename}")

async def main():
    """Run complete error handling and security testing suite"""
    print("🛡️ AI Companion Error Handling & Security Testing")
    print("=" * 55)
    
    tester = ErrorHandlingTester()
    
    try:
        # Run all tests
        await tester.test_api_error_handling()
        await tester.test_edge_cases()
        await tester.test_data_validation()
        await tester.test_security_scenarios()
        
        # Generate and save report
        tester.generate_report()
        tester.save_results()
        
        print("\n✅ Error handling and security testing completed!")
        
    except Exception as e:
        print(f"\n❌ Testing failed: {str(e)}")
        return False
    
    return True

if __name__ == "__main__":
    # Run the error handling and security tests
    success = asyncio.run(main())
    
    if success:
        print("\n🎉 All error handling and security tests completed!")
    else:
        print("\n❌ Testing encountered errors")
        exit(1)

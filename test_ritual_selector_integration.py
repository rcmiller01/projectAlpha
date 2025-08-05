"""
Integration Test Suite for Ritual Selector Panel
Tests the complete RitualSelectorPanel component with live API data.
"""

import requests
import json
import time
from datetime import datetime
import subprocess
import sys
import threading
import uuid


class RitualSelectorIntegrationTest:
    def __init__(self):
        self.base_url = "http://localhost:5001"
        self.test_results = []
        self.passed = 0
        self.failed = 0
        
    def log_test(self, test_name, passed, message=""):
        """Log test result with timestamp"""
        status = "✅ PASS" if passed else "❌ FAIL"
        timestamp = datetime.now().strftime("%H:%M:%S")
        result = {
            'test': test_name,
            'status': status,
            'message': message,
            'timestamp': timestamp
        }
        self.test_results.append(result)
        
        if passed:
            self.passed += 1
        else:
            self.failed += 1
            
        print(f"[{timestamp}] {status}: {test_name}")
        if message and not passed:
            print(f"    💬 {message}")
    
    def test_server_health(self):
        """Test if the ritual selector API server is running"""
        try:
            response = requests.get(f"{self.base_url}/api/rituals/active", timeout=5)
            self.log_test("Server Health Check", response.status_code == 200)
            return response.status_code == 200
        except requests.exceptions.RequestException as e:
            self.log_test("Server Health Check", False, f"Connection failed: {str(e)}")
            return False
    
    def test_active_rituals_endpoint(self):
        """Test the active rituals API endpoint"""
        try:
            response = requests.get(f"{self.base_url}/api/rituals/active")
            
            if response.status_code != 200:
                self.log_test("Active Rituals Endpoint", False, f"Status code: {response.status_code}")
                return False
            
            data = response.json()
            
            # Validate structure
            if not isinstance(data, list):
                self.log_test("Active Rituals Endpoint", False, "Response is not a list")
                return False
            
            if len(data) == 0:
                self.log_test("Active Rituals Endpoint", False, "No rituals returned")
                return False
            
            # Validate first ritual structure
            ritual = data[0]
            required_fields = ['id', 'name', 'mood_symbol', 'feeling_description', 
                             'activation_method', 'ritual_type', 'is_available', 'frequency']
            
            for field in required_fields:
                if field not in ritual:
                    self.log_test("Active Rituals Endpoint", False, f"Missing field: {field}")
                    return False
            
            # Validate activation methods
            valid_methods = ['reflective', 'co_initiated', 'adaptive', 'passive']
            if ritual['activation_method'] not in valid_methods:
                self.log_test("Active Rituals Endpoint", False, f"Invalid activation method: {ritual['activation_method']}")
                return False
            
            self.log_test("Active Rituals Endpoint", True, f"Retrieved {len(data)} rituals")
            return True
            
        except Exception as e:
            self.log_test("Active Rituals Endpoint", False, f"Exception: {str(e)}")
            return False
    
    def test_active_symbols_endpoint(self):
        """Test the active symbols API endpoint"""
        try:
            response = requests.get(f"{self.base_url}/api/symbols/active")
            
            if response.status_code != 200:
                self.log_test("Active Symbols Endpoint", False, f"Status code: {response.status_code}")
                return False
            
            data = response.json()
            
            # Validate structure
            if not isinstance(data, list):
                self.log_test("Active Symbols Endpoint", False, "Response is not a list")
                return False
            
            if len(data) == 0:
                self.log_test("Active Symbols Endpoint", False, "No symbols returned")
                return False
            
            # Validate first symbol structure
            symbol = data[0]
            required_fields = ['id', 'name', 'emotional_binding', 'ritual_connections', 
                             'frequency', 'salience_score', 'recent_contexts']
            
            for field in required_fields:
                if field not in symbol:
                    self.log_test("Active Symbols Endpoint", False, f"Missing field: {field}")
                    return False
            
            # Validate salience score range
            if not (0 <= symbol['salience_score'] <= 1):
                self.log_test("Active Symbols Endpoint", False, f"Invalid salience score: {symbol['salience_score']}")
                return False
            
            self.log_test("Active Symbols Endpoint", True, f"Retrieved {len(data)} symbols")
            return True
            
        except Exception as e:
            self.log_test("Active Symbols Endpoint", False, f"Exception: {str(e)}")
            return False
    
    def test_ritual_invocation(self):
        """Test ritual invocation endpoint"""
        try:
            # First get available rituals
            response = requests.get(f"{self.base_url}/api/rituals/active")
            rituals = response.json()
            
            # Find an available ritual
            available_ritual = None
            for ritual in rituals:
                if ritual.get('is_available', False):
                    available_ritual = ritual
                    break
            
            if not available_ritual:
                self.log_test("Ritual Invocation", False, "No available rituals found")
                return False
            
            # Invoke the ritual
            invocation_data = {
                'ritual_id': available_ritual['id'],
                'activation_method': available_ritual['activation_method'],
                'context': 'integration_test'
            }
            
            response = requests.post(f"{self.base_url}/api/rituals/invoke", json=invocation_data)
            
            if response.status_code != 200:
                self.log_test("Ritual Invocation", False, f"Status code: {response.status_code}")
                return False
            
            result = response.json()
            
            # Validate response structure
            required_fields = ['success', 'ritual_id', 'invoked_at']
            for field in required_fields:
                if field not in result:
                    self.log_test("Ritual Invocation", False, f"Missing field: {field}")
                    return False
            
            if not result['success']:
                self.log_test("Ritual Invocation", False, "Invocation marked as unsuccessful")
                return False
            
            self.log_test("Ritual Invocation", True, f"Invoked ritual: {available_ritual['name']}")
            return True
            
        except Exception as e:
            self.log_test("Ritual Invocation", False, f"Exception: {str(e)}")
            return False
    
    def test_symbol_history(self):
        """Test symbol history endpoint"""
        try:
            # Get a symbol to test
            response = requests.get(f"{self.base_url}/api/symbols/active")
            symbols = response.json()
            
            if not symbols:
                self.log_test("Symbol History", False, "No symbols available")
                return False
            
            symbol_id = symbols[0]['id']
            
            # Get symbol history
            response = requests.get(f"{self.base_url}/api/symbols/{symbol_id}/history")
            
            if response.status_code != 200:
                self.log_test("Symbol History", False, f"Status code: {response.status_code}")
                return False
            
            history = response.json()
            
            # Validate structure
            if not isinstance(history, list):
                self.log_test("Symbol History", False, "History is not a list")
                return False
            
            self.log_test("Symbol History", True, f"Retrieved {len(history)} history entries")
            return True
            
        except Exception as e:
            self.log_test("Symbol History", False, f"Exception: {str(e)}")
            return False
    
    def test_ritual_offer_submission(self):
        """Test submitting a custom ritual offer"""
        try:
            offer_data = {
                'intent': 'Let us weave threads of understanding through this moment of testing',
                'ritual_type': 'co_created',
                'symbols': ['thread', 'light'],
                'context': 'integration_test_ritual_offer'
            }
            
            response = requests.post(f"{self.base_url}/api/rituals/offer", json=offer_data)
            
            if response.status_code != 200:
                self.log_test("Ritual Offer Submission", False, f"Status code: {response.status_code}")
                return False
            
            result = response.json()
            
            # Validate response
            required_fields = ['success', 'offer_id', 'offered_at']
            for field in required_fields:
                if field not in result:
                    self.log_test("Ritual Offer Submission", False, f"Missing field: {field}")
                    return False
            
            if not result['success']:
                self.log_test("Ritual Offer Submission", False, "Offer marked as unsuccessful")
                return False
            
            self.log_test("Ritual Offer Submission", True, f"Submitted offer: {result['offer_id']}")
            return True
            
        except Exception as e:
            self.log_test("Ritual Offer Submission", False, f"Exception: {str(e)}")
            return False
    
    def test_recent_offerings(self):
        """Test recent offerings endpoint"""
        try:
            response = requests.get(f"{self.base_url}/api/rituals/offers/recent")
            
            if response.status_code != 200:
                self.log_test("Recent Offerings", False, f"Status code: {response.status_code}")
                return False
            
            offers = response.json()
            
            if not isinstance(offers, list):
                self.log_test("Recent Offerings", False, "Response is not a list")
                return False
            
            # If we have offers, validate structure
            if offers:
                offer = offers[0]
                required_fields = ['id', 'intent', 'offered_at', 'status']
                for field in required_fields:
                    if field not in offer:
                        self.log_test("Recent Offerings", False, f"Missing field: {field}")
                        return False
            
            self.log_test("Recent Offerings", True, f"Retrieved {len(offers)} recent offers")
            return True
            
        except Exception as e:
            self.log_test("Recent Offerings", False, f"Exception: {str(e)}")
            return False
    
    def test_symbol_salience_ranking(self):
        """Test that symbols are properly ranked by salience"""
        try:
            response = requests.get(f"{self.base_url}/api/symbols/active")
            symbols = response.json()
            
            if not symbols or len(symbols) < 2:
                self.log_test("Symbol Salience Ranking", False, "Need at least 2 symbols")
                return False
            
            # Check if salience scores vary (not all the same)
            salience_scores = [s['salience_score'] for s in symbols]
            if len(set(salience_scores)) < 2:
                self.log_test("Symbol Salience Ranking", False, "All symbols have same salience")
                return False
            
            # Find high and low salience symbols
            high_salience = [s for s in symbols if s['salience_score'] > 0.7]
            low_salience = [s for s in symbols if s['salience_score'] < 0.5]
            
            self.log_test("Symbol Salience Ranking", True, 
                         f"High salience: {len(high_salience)}, Low salience: {len(low_salience)}")
            return True
            
        except Exception as e:
            self.log_test("Symbol Salience Ranking", False, f"Exception: {str(e)}")
            return False
    
    def test_ritual_availability_logic(self):
        """Test ritual availability logic"""
        try:
            response = requests.get(f"{self.base_url}/api/rituals/active")
            rituals = response.json()
            
            available_count = len([r for r in rituals if r['is_available']])
            unavailable_count = len(rituals) - available_count
            
            # Should have some variation in availability
            if available_count == 0:
                self.log_test("Ritual Availability Logic", False, "No rituals are available")
                return False
            
            if unavailable_count == 0:
                self.log_test("Ritual Availability Logic", False, "All rituals are available (no variation)")
                return False
            
            # Check frequency distribution
            frequencies = [r['frequency'] for r in rituals]
            if max(frequencies) - min(frequencies) < 5:
                self.log_test("Ritual Availability Logic", False, "Frequency variation too small")
                return False
            
            self.log_test("Ritual Availability Logic", True, 
                         f"Available: {available_count}, Unavailable: {unavailable_count}")
            return True
            
        except Exception as e:
            self.log_test("Ritual Availability Logic", False, f"Exception: {str(e)}")
            return False
    
    def test_emotional_binding_coverage(self):
        """Test that symbols cover diverse emotional bindings"""
        try:
            response = requests.get(f"{self.base_url}/api/symbols/active")
            symbols = response.json()
            
            emotional_bindings = [s['emotional_binding'] for s in symbols]
            unique_bindings = set(emotional_bindings)
            
            # Should have good diversity
            if len(unique_bindings) < 5:
                self.log_test("Emotional Binding Coverage", False, 
                             f"Only {len(unique_bindings)} unique emotional bindings")
                return False
            
            # Check for expected emotional categories
            expected_emotions = ['contemplative', 'melancholy', 'joy', 'tender', 'awe']
            covered_emotions = [e for e in expected_emotions if e in unique_bindings]
            
            if len(covered_emotions) < 3:
                self.log_test("Emotional Binding Coverage", False, 
                             f"Only {len(covered_emotions)} expected emotions covered")
                return False
            
            self.log_test("Emotional Binding Coverage", True, 
                         f"{len(unique_bindings)} unique bindings, {len(covered_emotions)} expected emotions")
            return True
            
        except Exception as e:
            self.log_test("Emotional Binding Coverage", False, f"Exception: {str(e)}")
            return False
    
    def run_all_tests(self):
        """Run the complete test suite"""
        print("🌙 Ritual Selector Panel - Integration Test Suite")
        print("=" * 60)
        print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🎯 Target API: {self.base_url}")
        print()
        
        # Core API tests
        if not self.test_server_health():
            print("❌ Server is not accessible - aborting remaining tests")
            self.print_summary()
            return
        
        self.test_active_rituals_endpoint()
        self.test_active_symbols_endpoint()
        self.test_ritual_invocation()
        self.test_symbol_history()
        self.test_ritual_offer_submission()
        self.test_recent_offerings()
        
        # Data quality tests
        self.test_symbol_salience_ranking()
        self.test_ritual_availability_logic()
        self.test_emotional_binding_coverage()
        
        self.print_summary()
    
    def print_summary(self):
        """Print test summary and detailed results"""
        print()
        print("=" * 60)
        print("📊 TEST SUMMARY")
        print("=" * 60)
        
        total_tests = self.passed + self.failed
        success_rate = (self.passed / total_tests * 100) if total_tests > 0 else 0
        
        print(f"🎯 Total Tests: {total_tests}")
        print(f"✅ Passed: {self.passed}")
        print(f"❌ Failed: {self.failed}")
        print(f"📈 Success Rate: {success_rate:.1f}%")
        print()
        
        if self.failed > 0:
            print("❌ FAILED TESTS:")
            print("-" * 40)
            for result in self.test_results:
                if "FAIL" in result['status']:
                    print(f"[{result['timestamp']}] {result['test']}")
                    if result['message']:
                        print(f"    💬 {result['message']}")
            print()
        
        print("✅ PASSED TESTS:")
        print("-" * 40)
        for result in self.test_results:
            if "PASS" in result['status']:
                print(f"[{result['timestamp']}] {result['test']}")
        
        print()
        print("=" * 60)
        
        if success_rate >= 90:
            print("🌟 EXCELLENT: Ritual Selector Panel is ready for sacred interactions!")
        elif success_rate >= 75:
            print("🌙 GOOD: Most ritual functions are working, minor issues to resolve")
        elif success_rate >= 50:
            print("⚠️  MODERATE: Several issues need attention before launch")
        else:
            print("🚨 CRITICAL: Major issues preventing proper ritual functionality")
        
        print("🕯️ The ritual testing ceremony concludes...")


if __name__ == "__main__":
    print("🌙 Starting Ritual Selector Panel Integration Tests...")
    print("📡 Checking server status first...")
    
    # Wait a moment for server to be ready
    time.sleep(2)
    
    # Run the test suite
    test_suite = RitualSelectorIntegrationTest()
    test_suite.run_all_tests()

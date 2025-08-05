#!/usr/bin/env python3
"""
Integration test for MemoryAndSymbolViewer component and API.
Tests the complete flow from data generation to API responses.
"""

import asyncio
import json
import requests
import time
from pathlib import Path
import subprocess
import sys
from datetime import datetime

class MemorySymbolIntegrationTest:
    """Test suite for MemoryAndSymbolViewer integration"""
    
    def __init__(self):
        self.api_url = "http://localhost:5001"
        self.data_files = {
            "memory_trace": Path("data/emotional_memory_trace.json"),
            "symbolic_map": Path("data/symbolic_map.json"),
            "anchor_state": Path("data/anchor_state.json")
        }
        self.test_results = []

    def log_test(self, test_name: str, passed: bool, details: str = ""):
        """Log test result"""
        status = "✓ PASS" if passed else "✗ FAIL"
        self.test_results.append((test_name, passed, details))
        print(f"{status} {test_name}")
        if details and not passed:
            print(f"    Details: {details}")

    def test_data_files_exist(self):
        """Test that all required data files exist"""
        for name, file_path in self.data_files.items():
            exists = file_path.exists()
            self.log_test(
                f"Data file exists: {name}",
                exists,
                f"File: {file_path}" if not exists else ""
            )

    def test_data_file_structure(self):
        """Test that data files have correct structure"""
        try:
            # Test memory trace structure
            with open(self.data_files["memory_trace"], 'r') as f:
                memory_data = json.load(f)
            
            required_memory_fields = ["trace", "last_updated"]
            memory_valid = all(field in memory_data for field in required_memory_fields)
            
            if memory_valid and memory_data["trace"]:
                entry = memory_data["trace"][0]
                required_entry_fields = ["id", "timestamp", "dominant_mood", "memory_phrase", "tags"]
                entry_valid = all(field in entry for field in required_entry_fields)
            else:
                entry_valid = False
            
            self.log_test(
                "Memory trace structure",
                memory_valid and entry_valid,
                f"Entries: {len(memory_data.get('trace', []))}"
            )
            
            # Test symbolic map structure
            with open(self.data_files["symbolic_map"], 'r') as f:
                symbol_data = json.load(f)
            
            required_symbol_fields = ["symbols", "last_updated"]
            symbol_valid = all(field in symbol_data for field in required_symbol_fields)
            
            if symbol_valid and symbol_data["symbols"]:
                symbol = symbol_data["symbols"][0]
                required_symbol_entry_fields = ["id", "name", "affective_color", "frequency"]
                symbol_entry_valid = all(field in symbol for field in required_symbol_entry_fields)
            else:
                symbol_entry_valid = False
            
            self.log_test(
                "Symbolic map structure",
                symbol_valid and symbol_entry_valid,
                f"Symbols: {len(symbol_data.get('symbols', []))}"
            )
            
            # Test anchor state structure
            with open(self.data_files["anchor_state"], 'r') as f:
                anchor_data = json.load(f)
            
            required_anchor_fields = ["vectors", "tether_score", "identity_stability"]
            anchor_valid = all(field in anchor_data for field in required_anchor_fields)
            
            self.log_test(
                "Anchor state structure",
                anchor_valid,
                f"Vectors: {len(anchor_data.get('vectors', {}))}"
            )
            
        except Exception as e:
            self.log_test("Data file structure", False, str(e))

    def test_api_server_health(self):
        """Test if API server is responding"""
        try:
            response = requests.get(f"{self.api_url}/api/health", timeout=5)
            healthy = response.status_code == 200
            
            if healthy:
                health_data = response.json()
                components_healthy = all(health_data.get("components", {}).values())
                self.log_test(
                    "API server health",
                    components_healthy,
                    f"Status: {health_data.get('status', 'unknown')}"
                )
            else:
                self.log_test("API server health", False, f"Status code: {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            self.log_test("API server health", False, f"Connection error: {str(e)}")

    def test_memory_api_endpoints(self):
        """Test memory-related API endpoints"""
        try:
            # Test GET emotional trace
            response = requests.get(f"{self.api_url}/api/memory/emotional_trace", timeout=10)
            if response.status_code == 200:
                data = response.json()
                has_trace = "trace" in data and len(data["trace"]) > 0
                self.log_test(
                    "GET /api/memory/emotional_trace",
                    has_trace,
                    f"Entries: {len(data.get('trace', []))}"
                )
            else:
                self.log_test(
                    "GET /api/memory/emotional_trace",
                    False,
                    f"Status: {response.status_code}"
                )
            
            # Test POST add memory entry
            test_entry = {
                "dominant_mood": "contemplative",
                "memory_phrase": "Integration test memory entry",
                "tags": ["test", "integration"],
                "drift_score": 0.1,
                "intensity": 0.5,
                "context": "Automated testing scenario",
                "symbolic_connections": ["mirror"]
            }
            
            response = requests.post(f"{self.api_url}/api/memory/add_entry", json=test_entry, timeout=10)
            add_success = response.status_code == 200
            self.log_test(
                "POST /api/memory/add_entry",
                add_success,
                f"Status: {response.status_code}"
            )
            
        except requests.exceptions.RequestException as e:
            self.log_test("Memory API endpoints", False, f"Request error: {str(e)}")

    def test_symbol_api_endpoints(self):
        """Test symbol-related API endpoints"""
        try:
            # Test GET symbols
            response = requests.get(f"{self.api_url}/api/symbols/active", timeout=10)
            if response.status_code == 200:
                data = response.json()
                has_symbols = "symbols" in data and len(data["symbols"]) > 0
                self.log_test(
                    "GET /api/symbols/active",
                    has_symbols,
                    f"Symbols: {len(data.get('symbols', []))}"
                )
            else:
                self.log_test(
                    "GET /api/symbols/active",
                    False,
                    f"Status: {response.status_code}"
                )
            
            # Test POST invoke symbol
            test_invocation = {
                "name": "mirror",
                "affective_color": "contemplative"
            }
            
            response = requests.post(f"{self.api_url}/api/symbols/invoke", json=test_invocation, timeout=10)
            invoke_success = response.status_code == 200
            self.log_test(
                "POST /api/symbols/invoke",
                invoke_success,
                f"Status: {response.status_code}"
            )
            
        except requests.exceptions.RequestException as e:
            self.log_test("Symbol API endpoints", False, f"Request error: {str(e)}")

    def test_anchor_api_endpoints(self):
        """Test anchor-related API endpoints"""
        try:
            # Test GET anchor state
            response = requests.get(f"{self.api_url}/api/anchor/state", timeout=10)
            if response.status_code == 200:
                data = response.json()
                has_vectors = "vectors" in data and len(data["vectors"]) > 0
                has_tether_score = "tether_score" in data
                self.log_test(
                    "GET /api/anchor/state",
                    has_vectors and has_tether_score,
                    f"Tether score: {data.get('tether_score', 'missing')}"
                )
            else:
                self.log_test(
                    "GET /api/anchor/state",
                    False,
                    f"Status: {response.status_code}"
                )
            
            # Test POST adjust anchor
            test_adjustment = {
                "vector": "empathy",
                "value": 0.85
            }
            
            response = requests.post(f"{self.api_url}/api/anchor/adjust", json=test_adjustment, timeout=10)
            adjust_success = response.status_code == 200
            self.log_test(
                "POST /api/anchor/adjust",
                adjust_success,
                f"Status: {response.status_code}"
            )
            
        except requests.exceptions.RequestException as e:
            self.log_test("Anchor API endpoints", False, f"Request error: {str(e)}")

    def test_data_consistency(self):
        """Test consistency between data files and API responses"""
        try:
            # Compare memory trace
            with open(self.data_files["memory_trace"], 'r') as f:
                file_data = json.load(f)
            
            response = requests.get(f"{self.api_url}/api/memory/emotional_trace", timeout=10)
            if response.status_code == 200:
                api_data = response.json()
                
                file_count = len(file_data.get("trace", []))
                api_count = len(api_data.get("trace", []))
                
                # Allow for small differences due to test entries
                consistent = abs(file_count - api_count) <= 2
                self.log_test(
                    "Memory data consistency",
                    consistent,
                    f"File: {file_count}, API: {api_count}"
                )
            else:
                self.log_test("Memory data consistency", False, "API not responding")
                
        except Exception as e:
            self.log_test("Data consistency", False, str(e))

    def test_component_files_exist(self):
        """Test that component files exist"""
        component_files = [
            Path("ui/MemoryAndSymbolViewer.jsx"),
            Path("memory_symbol_api.py"),
            Path("demo_memory_symbol_viewer.py"),
            Path("demo_memory_viewer.html"),
            Path("README_MemoryAndSymbolViewer.md")
        ]
        
        for file_path in component_files:
            exists = file_path.exists()
            self.log_test(
                f"Component file: {file_path.name}",
                exists,
                f"Path: {file_path}" if not exists else ""
            )

    def run_all_tests(self):
        """Run all integration tests"""
        print("🧪 === MemoryAndSymbolViewer Integration Tests ===\n")
        
        print("📁 Testing Data Files...")
        self.test_data_files_exist()
        self.test_data_file_structure()
        
        print(f"\n🌐 Testing API Server ({self.api_url})...")
        self.test_api_server_health()
        
        print("\n📡 Testing API Endpoints...")
        self.test_memory_api_endpoints()
        self.test_symbol_api_endpoints()
        self.test_anchor_api_endpoints()
        
        print("\n🔍 Testing Data Consistency...")
        self.test_data_consistency()
        
        print("\n📦 Testing Component Files...")
        self.test_component_files_exist()
        
        # Print summary
        print(f"\n📊 === Test Results Summary ===")
        passed = sum(1 for _, success, _ in self.test_results if success)
        total = len(self.test_results)
        
        print(f"Tests Passed: {passed}/{total}")
        print(f"Success Rate: {(passed/total)*100:.1f}%")
        
        if passed == total:
            print("🎉 All tests passed! MemoryAndSymbolViewer is ready for use.")
        else:
            print(f"⚠️  {total-passed} test(s) failed. Check the details above.")
            
            # Show failed tests
            failed_tests = [name for name, success, _ in self.test_results if not success]
            if failed_tests:
                print(f"\nFailed tests:")
                for test_name in failed_tests:
                    print(f"  • {test_name}")
        
        return passed == total

def check_api_server_running():
    """Check if API server is running, if not provide instructions"""
    try:
        response = requests.get("http://localhost:5001/api/health", timeout=3)
        return response.status_code == 200
    except:
        return False

async def main():
    """Main test execution"""
    print("🌟 MemoryAndSymbolViewer Integration Testing")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Check if API server is running
    if not check_api_server_running():
        print("⚠️  API server not detected on localhost:5001")
        print("🚀 To start the API server, run:")
        print("   python memory_symbol_api.py")
        print("\nRunning tests without API server (data files only)...\n")
    
    # Run tests
    tester = MemorySymbolIntegrationTest()
    success = tester.run_all_tests()
    
    print(f"\n📋 Next Steps:")
    if success:
        print("1. ✅ All systems ready!")
        print("2. 🌐 Open demo_memory_viewer.html in your browser")
        print("3. 🧩 Integrate MemoryAndSymbolViewer.jsx into your React app")
        print("4. 🎨 Customize styling and behavior as needed")
    else:
        print("1. 🔧 Fix any failed tests above")
        print("2. 🔄 Re-run this test suite")
        print("3. 📖 Check README_MemoryAndSymbolViewer.md for troubleshooting")
    
    return success

if __name__ == "__main__":
    asyncio.run(main())

"""
Integration Test Suite for Drift Journal Renderer
Tests the complete drift tracking system including API, data generation, and frontend integration.
"""

import requests
import json
import time
import sys
from datetime import datetime, timedelta
import subprocess
import threading
import os


class DriftJournalTester:
    def __init__(self, api_url='http://localhost:5000'):
        self.api_url = api_url
        self.test_results = []
        self.server_process = None
        
    def log_test(self, test_name, success, message="", details=None):
        """Log test result"""
        result = {
            'test': test_name,
            'success': success,
            'message': message,
            'details': details,
            'timestamp': datetime.now().isoformat()
        }
        self.test_results.append(result)
        
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
        if message:
            print(f"     {message}")
        if not success and details:
            print(f"     Details: {details}")
    
    def test_api_health(self):
        """Test API health endpoint"""
        try:
            response = requests.get(f"{self.api_url}/api/health", timeout=5)
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'healthy':
                    self.log_test("API Health Check", True, f"Server healthy with {data.get('data_status', {}).get('drift_entries', 0)} entries")
                    return True
                else:
                    self.log_test("API Health Check", False, "Server unhealthy response")
                    return False
            else:
                self.log_test("API Health Check", False, f"HTTP {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            self.log_test("API Health Check", False, "Connection failed", str(e))
            return False
    
    def test_drift_history_endpoint(self):
        """Test drift history retrieval"""
        try:
            # Test default history
            response = requests.get(f"{self.api_url}/api/drift/history", timeout=5)
            if response.status_code == 200:
                data = response.json()
                entries = data.get('entries', [])
                if len(entries) > 0:
                    self.log_test("Drift History Default", True, f"Retrieved {len(entries)} entries")
                    
                    # Validate entry structure
                    sample_entry = entries[0]
                    required_fields = ['id', 'timestamp', 'mood_before', 'mood_after', 
                                     'internal_reflection', 'drift_cause', 'drift_magnitude']
                    missing_fields = [field for field in required_fields if field not in sample_entry]
                    
                    if not missing_fields:
                        self.log_test("Entry Structure Validation", True, "All required fields present")
                    else:
                        self.log_test("Entry Structure Validation", False, f"Missing fields: {missing_fields}")
                else:
                    self.log_test("Drift History Default", False, "No entries returned")
            else:
                self.log_test("Drift History Default", False, f"HTTP {response.status_code}")
                
            # Test time range filtering
            for time_range in ['day', 'week', 'month']:
                response = requests.get(f"{self.api_url}/api/drift/history?range={time_range}", timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    entries = data.get('entries', [])
                    self.log_test(f"Time Range Filter ({time_range})", True, f"{len(entries)} entries")
                else:
                    self.log_test(f"Time Range Filter ({time_range})", False, f"HTTP {response.status_code}")
                    
        except requests.exceptions.RequestException as e:
            self.log_test("Drift History Endpoint", False, "Connection failed", str(e))
    
    def test_drift_summary_endpoint(self):
        """Test drift summary analytics"""
        try:
            response = requests.get(f"{self.api_url}/api/drift/summary", timeout=5)
            if response.status_code == 200:
                data = response.json()
                
                # Check required fields
                required_fields = ['total_drifts', 'average_magnitude', 'drift_types', 'timeline_data']
                missing_fields = [field for field in required_fields if field not in data]
                
                if not missing_fields:
                    self.log_test("Drift Summary Structure", True, "All required fields present")
                    
                    # Validate data types and ranges
                    total_drifts = data.get('total_drifts', 0)
                    avg_magnitude = data.get('average_magnitude', 0)
                    timeline_data = data.get('timeline_data', [])
                    
                    if isinstance(total_drifts, int) and total_drifts >= 0:
                        self.log_test("Total Drifts Validation", True, f"{total_drifts} drifts")
                    else:
                        self.log_test("Total Drifts Validation", False, "Invalid total_drifts value")
                    
                    if isinstance(avg_magnitude, (int, float)) and 0 <= avg_magnitude <= 1:
                        self.log_test("Average Magnitude Validation", True, f"{avg_magnitude:.3f}")
                    else:
                        self.log_test("Average Magnitude Validation", False, "Invalid average_magnitude value")
                    
                    if isinstance(timeline_data, list) and len(timeline_data) > 0:
                        self.log_test("Timeline Data Validation", True, f"{len(timeline_data)} data points")
                    else:
                        self.log_test("Timeline Data Validation", False, "Invalid timeline_data")
                        
                else:
                    self.log_test("Drift Summary Structure", False, f"Missing fields: {missing_fields}")
            else:
                self.log_test("Drift Summary Endpoint", False, f"HTTP {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            self.log_test("Drift Summary Endpoint", False, "Connection failed", str(e))
    
    def test_drift_actions(self):
        """Test drift approval, reversion, and annotation"""
        try:
            # First get a drift entry to work with
            response = requests.get(f"{self.api_url}/api/drift/history?limit=5", timeout=5)
            if response.status_code != 200:
                self.log_test("Drift Actions Setup", False, "Could not retrieve drift entries")
                return
                
            entries = response.json().get('entries', [])
            if not entries:
                self.log_test("Drift Actions Setup", False, "No drift entries available")
                return
            
            test_entry = entries[0]
            drift_id = test_entry['id']
            
            # Test drift approval
            approval_data = {'drift_id': drift_id}
            response = requests.post(f"{self.api_url}/api/drift/approve", json=approval_data, timeout=5)
            if response.status_code == 200:
                self.log_test("Drift Approval", True, f"Approved drift {drift_id}")
            else:
                self.log_test("Drift Approval", False, f"HTTP {response.status_code}")
            
            # Test drift annotation
            annotation_data = {
                'drift_id': drift_id,
                'annotation': 'Test annotation: This drift represents meaningful growth.'
            }
            response = requests.post(f"{self.api_url}/api/drift/annotate", json=annotation_data, timeout=5)
            if response.status_code == 200:
                self.log_test("Drift Annotation", True, f"Annotated drift {drift_id}")
            else:
                self.log_test("Drift Annotation", False, f"HTTP {response.status_code}")
            
            # Test drift reversion (with a different entry to avoid conflicts)
            if len(entries) > 1:
                revert_entry = entries[1]
                revert_data = {'drift_id': revert_entry['id']}
                response = requests.post(f"{self.api_url}/api/drift/revert", json=revert_data, timeout=5)
                if response.status_code == 200:
                    self.log_test("Drift Reversion", True, f"Reverted drift {revert_entry['id']}")
                else:
                    self.log_test("Drift Reversion", False, f"HTTP {response.status_code}")
            
        except requests.exceptions.RequestException as e:
            self.log_test("Drift Actions", False, "Connection failed", str(e))
    
    def test_drift_generation(self):
        """Test drift entry generation"""
        try:
            response = requests.post(f"{self.api_url}/api/drift/generate", timeout=5)
            if response.status_code == 200:
                data = response.json()
                new_entry = data.get('entry')
                if new_entry and 'id' in new_entry:
                    self.log_test("Drift Generation", True, f"Generated entry {new_entry['id']}")
                else:
                    self.log_test("Drift Generation", False, "Invalid entry structure")
            else:
                self.log_test("Drift Generation", False, f"HTTP {response.status_code}")
        except requests.exceptions.RequestException as e:
            self.log_test("Drift Generation", False, "Connection failed", str(e))
    
    def test_configuration_endpoint(self):
        """Test configuration retrieval and update"""
        try:
            # Test GET configuration
            response = requests.get(f"{self.api_url}/api/drift/config", timeout=5)
            if response.status_code == 200:
                config = response.json()
                if 'auto_generate' in config and 'drift_sensitivity' in config:
                    self.log_test("Config Retrieval", True, "Configuration retrieved successfully")
                    
                    # Test POST configuration update
                    update_data = {
                        'drift_sensitivity': 0.7,
                        'max_history_entries': 75
                    }
                    response = requests.post(f"{self.api_url}/api/drift/config", json=update_data, timeout=5)
                    if response.status_code == 200:
                        self.log_test("Config Update", True, "Configuration updated successfully")
                    else:
                        self.log_test("Config Update", False, f"HTTP {response.status_code}")
                else:
                    self.log_test("Config Retrieval", False, "Invalid configuration structure")
            else:
                self.log_test("Config Retrieval", False, f"HTTP {response.status_code}")
        except requests.exceptions.RequestException as e:
            self.log_test("Configuration Endpoint", False, "Connection failed", str(e))
    
    def test_data_files_exist(self):
        """Test that required data files exist"""
        required_files = [
            'drift_history.json',
            'drift_summary.json', 
            'drift_annotations.json',
            'drift_config.json'
        ]
        
        for filename in required_files:
            if os.path.exists(filename):
                try:
                    with open(filename, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    self.log_test(f"Data File: {filename}", True, f"Valid JSON with {len(data) if isinstance(data, (list, dict)) else 'N/A'} items")
                except json.JSONDecodeError:
                    self.log_test(f"Data File: {filename}", False, "Invalid JSON format")
                except Exception as e:
                    self.log_test(f"Data File: {filename}", False, f"Read error: {str(e)}")
            else:
                self.log_test(f"Data File: {filename}", False, "File not found")
    
    def test_frontend_component_integration(self):
        """Test frontend component file structure"""
        component_file = "ui/DriftJournalRenderer.jsx"
        
        if os.path.exists(component_file):
            try:
                with open(component_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Check for key component features
                required_features = [
                    'DriftJournalRenderer',
                    'fetchDriftHistory',
                    'fetchDriftSummary', 
                    'handleDriftApproval',
                    'handleAnchorReversion',
                    'submitDriftAnnotation',
                    'drift_causes',
                    'moodStates'
                ]
                
                missing_features = []
                for feature in required_features:
                    if feature not in content:
                        missing_features.append(feature)
                
                if not missing_features:
                    self.log_test("Component Structure", True, "All required features present")
                else:
                    self.log_test("Component Structure", False, f"Missing features: {missing_features}")
                
                # Check for proper React patterns
                react_patterns = ['useState', 'useEffect', 'axios']
                for pattern in react_patterns:
                    if pattern in content:
                        self.log_test(f"React Pattern: {pattern}", True, "Present")
                    else:
                        self.log_test(f"React Pattern: {pattern}", False, "Missing")
                        
            except Exception as e:
                self.log_test("Component File Reading", False, f"Error: {str(e)}")
        else:
            self.log_test("Component File Exists", False, "DriftJournalRenderer.jsx not found")
    
    def run_comprehensive_test_suite(self):
        """Run all tests in sequence"""
        print("🌊 Starting Drift Journal Renderer Integration Tests")
        print("=" * 70)
        
        # Test data files first
        print("\n📁 Testing Data Files...")
        self.test_data_files_exist()
        
        # Test frontend component
        print("\n⚛️ Testing Frontend Component...")
        self.test_frontend_component_integration()
        
        # Test API endpoints
        print("\n🔌 Testing API Endpoints...")
        if self.test_api_health():
            self.test_drift_history_endpoint()
            self.test_drift_summary_endpoint() 
            self.test_drift_actions()
            self.test_drift_generation()
            self.test_configuration_endpoint()
        else:
            print("⚠️ API server not available - skipping API tests")
            print("   Start the server with: python drift_journal_api.py")
        
        # Generate test report
        self.generate_test_report()
    
    def generate_test_report(self):
        """Generate comprehensive test report"""
        print("\n" + "=" * 70)
        print("📊 TEST REPORT SUMMARY")
        print("=" * 70)
        
        total_tests = len(self.test_results)
        passed_tests = len([r for r in self.test_results if r['success']])
        failed_tests = total_tests - passed_tests
        
        print(f"Total Tests: {total_tests}")
        print(f"✅ Passed: {passed_tests}")
        print(f"❌ Failed: {failed_tests}")
        print(f"Success Rate: {(passed_tests/total_tests*100):.1f}%")
        
        if failed_tests > 0:
            print(f"\n❌ Failed Tests:")
            for result in self.test_results:
                if not result['success']:
                    print(f"   • {result['test']}: {result['message']}")
        
        # Component readiness assessment
        print(f"\n🎯 COMPONENT READINESS ASSESSMENT:")
        
        api_tests = [r for r in self.test_results if 'API' in r['test'] or 'Endpoint' in r['test']]
        api_success = len([r for r in api_tests if r['success']]) / len(api_tests) if api_tests else 0
        
        component_tests = [r for r in self.test_results if 'Component' in r['test'] or 'React' in r['test']]
        component_success = len([r for r in component_tests if r['success']]) / len(component_tests) if component_tests else 0
        
        data_tests = [r for r in self.test_results if 'Data File' in r['test']]
        data_success = len([r for r in data_tests if r['success']]) / len(data_tests) if data_tests else 0
        
        print(f"   📡 API Backend: {api_success*100:.0f}% ready")
        print(f"   ⚛️ React Component: {component_success*100:.0f}% ready")
        print(f"   📁 Data Layer: {data_success*100:.0f}% ready")
        
        overall_readiness = (api_success + component_success + data_success) / 3
        print(f"   🎯 Overall Readiness: {overall_readiness*100:.0f}%")
        
        if overall_readiness > 0.8:
            print("\n🚀 READY FOR DEPLOYMENT!")
            print("   The DriftJournalRenderer is ready for production use.")
        elif overall_readiness > 0.6:
            print("\n🔧 NEEDS MINOR FIXES")
            print("   The component is mostly ready but needs some attention.")
        else:
            print("\n⚠️ NEEDS SIGNIFICANT WORK")
            print("   Several critical issues need to be addressed.")
        
        # Save detailed report
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'failed_tests': failed_tests,
                'success_rate': passed_tests/total_tests*100,
                'overall_readiness': overall_readiness*100
            },
            'detailed_results': self.test_results
        }
        
        with open('drift_journal_test_report.json', 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n📄 Detailed report saved to: drift_journal_test_report.json")


def main():
    """Main test execution"""
    print("🌊 Drift Journal Renderer - Integration Test Suite")
    print("=" * 70)
    
    # Check if demo data should be generated first
    if '--generate-data' in sys.argv or not os.path.exists('drift_history.json'):
        print("📊 Generating demo data first...")
        try:
            import demo_drift_journal
            demo_drift_journal.save_demo_data()
            print("✅ Demo data generated successfully\n")
        except ImportError:
            print("❌ Could not import demo_drift_journal module")
            print("   Please run: python demo_drift_journal.py")
            return
        except Exception as e:
            print(f"❌ Error generating demo data: {e}")
            return
    
    # Run tests
    tester = DriftJournalTester()
    tester.run_comprehensive_test_suite()
    
    print("\n🎭 Testing Complete - May your drifts be meaningful and your anchors strong.")


if __name__ == "__main__":
    main()

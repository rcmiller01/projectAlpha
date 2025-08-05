#!/usr/bin/env python3
"""
🐬 Dolphin v2.1 Master QA Test Runner
Executes all targeted QA scripts and provides comprehensive validation report
"""

import asyncio
import time
import subprocess
import sys
from datetime import datetime
from pathlib import Path

class DolphinV21MasterQA:
    """Master test coordinator for all Dolphin v2.1 advanced features"""
    
    def __init__(self):
        self.start_time = time.time()
        self.test_results = {}
        self.base_path = Path(__file__).parent
        
    def print_header(self):
        """Print test suite header"""
        print("🐬" * 20)
        print("🐬 DOLPHIN AI ORCHESTRATOR v2.1")
        print("🐬 COMPREHENSIVE QA VALIDATION SUITE")
        print("🐬" * 20)
        print(f"📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🎯 Target: http://localhost:8000")
        print("=" * 70)
    
    async def check_backend_availability(self):
        """Check if Dolphin backend is running"""
        print("🔍 Checking Dolphin backend availability...")
        
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get("http://localhost:8000/api/status") as response:
                    if response.status == 200:
                        data = await response.json()
                        version = data.get('version', 'unknown')
                        print(f"✅ Dolphin backend online - Version: {version}")
                        return True
                    else:
                        print(f"❌ Backend returned status {response.status}")
                        return False
        except ImportError:
            print("⚠️ aiohttp not available - using subprocess check")
            return await self.check_backend_subprocess()
        except Exception as e:
            print(f"❌ Backend check failed: {e}")
            return False
    
    async def check_backend_subprocess(self):
        """Fallback backend check using subprocess"""
        try:
            # Try curl command
            result = subprocess.run(
                ["curl", "-s", "http://localhost:8000/api/status"],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0:
                print("✅ Dolphin backend responding to curl")
                return True
            else:
                print("❌ Backend not responding to curl")
                return False
        except Exception as e:
            print(f"❌ Subprocess backend check failed: {e}")
            return False
    
    async def run_individual_qa_test(self, test_name, script_name):
        """Run an individual QA test script"""
        print(f"\n🧪 RUNNING: {test_name}")
        print("─" * 50)
        
        start_time = time.time()
        script_path = self.base_path / script_name
        
        try:
            # Run the QA script
            process = await asyncio.create_subprocess_exec(
                sys.executable, str(script_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            execution_time = time.time() - start_time
            success = process.returncode == 0
            
            # Decode output
            stdout_text = stdout.decode('utf-8', errors='ignore')
            stderr_text = stderr.decode('utf-8', errors='ignore')
            
            # Print test output
            if stdout_text:
                print(stdout_text)
            
            if stderr_text and not success:
                print(f"❌ Errors:\n{stderr_text}")
            
            # Store results
            self.test_results[test_name] = {
                'success': success,
                'execution_time': execution_time,
                'script': script_name,
                'stdout': stdout_text,
                'stderr': stderr_text,
                'return_code': process.returncode
            }
            
            status = "✅ PASSED" if success else "❌ FAILED"
            print(f"\n{status} - {test_name} ({execution_time:.1f}s)")
            
            return success
            
        except Exception as e:
            execution_time = time.time() - start_time
            print(f"❌ Exception running {test_name}: {e}")
            
            self.test_results[test_name] = {
                'success': False,
                'execution_time': execution_time,
                'script': script_name,
                'stdout': '',
                'stderr': str(e),
                'return_code': -1,
                'exception': str(e)
            }
            
            return False
    
    def generate_comprehensive_report(self):
        """Generate a comprehensive test report"""
        print("\n" + "=" * 70)
        print("📊 COMPREHENSIVE QA VALIDATION REPORT")
        print("=" * 70)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result['success'])
        failed_tests = total_tests - passed_tests
        total_execution_time = time.time() - self.start_time
        
        # Overall summary
        print(f"🎯 OVERALL RESULTS:")
        print(f"   Total Tests: {total_tests}")
        print(f"   ✅ Passed: {passed_tests}")
        print(f"   ❌ Failed: {failed_tests}")
        print(f"   ⏱️ Total Time: {total_execution_time:.1f}s")
        print(f"   📈 Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        # Individual test results
        print(f"\n📋 INDIVIDUAL TEST RESULTS:")
        for test_name, result in self.test_results.items():
            status = "✅ PASS" if result['success'] else "❌ FAIL"
            time_taken = result['execution_time']
            print(f"   {status} {test_name} ({time_taken:.1f}s)")
            
            if not result['success']:
                if 'exception' in result:
                    print(f"        Exception: {result['exception']}")
                elif result['return_code'] != 0:
                    print(f"        Return code: {result['return_code']}")
        
        # Performance analysis
        print(f"\n⚡ PERFORMANCE ANALYSIS:")
        execution_times = [r['execution_time'] for r in self.test_results.values()]
        if execution_times:
            avg_time = sum(execution_times) / len(execution_times)
            max_time = max(execution_times)
            min_time = min(execution_times)
            
            print(f"   Average test time: {avg_time:.1f}s")
            print(f"   Longest test: {max_time:.1f}s")
            print(f"   Shortest test: {min_time:.1f}s")
        
        # Feature assessment
        print(f"\n🌟 FEATURE ASSESSMENT:")
        feature_status = {
            "🔄 Reflection Engine": self.test_results.get("Reflection Engine QA", {}).get('success', False),
            "🌤️ Connectivity Manager": self.test_results.get("Connectivity Manager QA", {}).get('success', False),
            "🔐 Private Memory": self.test_results.get("Private Memory System QA", {}).get('success', False),
            "🎭 Persona Instructions": self.test_results.get("Persona Instruction Manager QA", {}).get('success', False),
            "🪩 Mirror Mode": self.test_results.get("Mirror Mode QA", {}).get('success', False),
            "📈 System Metrics": self.test_results.get("System Metrics QA", {}).get('success', False)
        }
        
        for feature, status in feature_status.items():
            status_icon = "✅" if status else "❌"
            print(f"   {status_icon} {feature}")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        if failed_tests == 0:
            print("   🎉 All tests passed! Dolphin v2.1 is ready for production.")
            print("   🚀 Consider running load tests and security audits.")
        elif failed_tests <= 2:
            print("   ⚠️ Minor issues detected. Review failed tests and retry.")
            print("   🔧 Most features are working correctly.")
        else:
            print("   🚨 Multiple failures detected. System needs attention.")
            print("   🔍 Check backend logs and dependency installation.")
            print("   📋 Verify all required services are running.")
        
        # Next steps
        print(f"\n🎯 NEXT STEPS:")
        print("   1. 📊 Review individual test outputs above")
        print("   2. 🔧 Address any failed test cases")
        print("   3. 🧪 Re-run specific tests after fixes")
        print("   4. 🚀 Deploy to production environment")
        print("   5. 📈 Set up continuous monitoring")
        
        return passed_tests == total_tests
    
    async def run_all_qa_tests(self):
        """Run all QA test suites"""
        self.print_header()
        
        # Check backend availability first
        if not await self.check_backend_availability():
            print("❌ Dolphin backend is not available. Please start it first:")
            print("   python dolphin_backend.py")
            return False
        
        # Define test suites in execution order
        test_suites = [
            ("System Metrics QA", "qa_system_metrics.py"),
            ("Connectivity Manager QA", "qa_connectivity_manager.py"), 
            ("Private Memory System QA", "qa_private_memory.py"),
            ("Persona Instruction Manager QA", "qa_persona_instructions.py"),
            ("Mirror Mode QA", "qa_mirror_mode.py"),
            ("Reflection Engine QA", "qa_reflection_engine.py")
        ]
        
        print(f"\n🎯 EXECUTING {len(test_suites)} QA TEST SUITES...")
        
        # Run each test suite
        overall_success = True
        for test_name, script_name in test_suites:
            success = await self.run_individual_qa_test(test_name, script_name)
            if not success:
                overall_success = False
            
            # Small delay between tests
            await asyncio.sleep(1)
        
        # Generate comprehensive report
        final_success = self.generate_comprehensive_report()
        
        return final_success and overall_success

async def main():
    """Main QA coordinator execution"""
    print("🐬 Starting Dolphin v2.1 Comprehensive QA Suite...")
    print("Ensure the backend is running: python dolphin_backend.py\n")
    
    qa_master = DolphinV21MasterQA()
    success = await qa_master.run_all_qa_tests()
    
    if success:
        print("\n🎉 ALL QA TESTS PASSED! Dolphin v2.1 validation complete.")
        return 0
    else:
        print("\n⚠️ Some QA tests failed. Review the report above.")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())

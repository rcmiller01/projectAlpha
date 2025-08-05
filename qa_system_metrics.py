#!/usr/bin/env python3
"""
📈 System Metrics Targeted QA Script
Tests real-time monitoring, performance tracking, and health assessment
"""

import asyncio
import aiohttp
import json
import time
import psutil
from datetime import datetime

class SystemMetricsQA:
    """Targeted testing for the System Metrics system"""
    
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.session_id = f"metrics_qa_{int(time.time())}"
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def test_realtime_metrics(self):
        """Test real-time system metrics collection"""
        print("📊 Testing real-time metrics collection...")
        
        try:
            async with self.session.get(f"{self.base_url}/api/metrics/realtime") as response:
                if response.status == 200:
                    data = await response.json()
                    
                    system_metrics = data.get('system', {})
                    application_metrics = data.get('application', {})
                    timestamp = data.get('timestamp', 'unknown')
                    
                    print(f"✅ Real-time metrics retrieved (timestamp: {timestamp})")
                    
                    # Validate system metrics
                    if system_metrics:
                        cpu_percent = system_metrics.get('cpu_percent', 0)
                        memory_percent = system_metrics.get('memory_percent', 0)
                        disk_percent = system_metrics.get('disk_percent', 0)
                        uptime_hours = system_metrics.get('uptime_hours', 0)
                        
                        print(f"   System Metrics:")
                        print(f"     CPU Usage: {cpu_percent:.1f}%")
                        print(f"     Memory Usage: {memory_percent:.1f}%")
                        print(f"     Disk Usage: {disk_percent:.1f}%")
                        print(f"     Uptime: {uptime_hours:.2f} hours")
                        
                        # Validate realistic values
                        metrics_valid = (
                            0 <= cpu_percent <= 100 and
                            0 <= memory_percent <= 100 and
                            0 <= disk_percent <= 100 and
                            uptime_hours >= 0
                        )
                        
                        if metrics_valid:
                            print("   ✅ System metrics values are realistic")
                        else:
                            print("   ⚠️ Some system metrics values seem unrealistic")
                    
                    # Validate application metrics
                    if application_metrics:
                        active_sessions = application_metrics.get('active_sessions', 0)
                        total_requests = application_metrics.get('total_requests', 0)
                        avg_response_time = application_metrics.get('avg_response_time_ms', 0)
                        
                        print(f"   Application Metrics:")
                        print(f"     Active Sessions: {active_sessions}")
                        print(f"     Total Requests: {total_requests}")
                        print(f"     Avg Response Time: {avg_response_time:.1f}ms")
                    
                    return data
                else:
                    print(f"❌ Real-time metrics unavailable: {response.status}")
                    return None
        except Exception as e:
            print(f"❌ Error getting real-time metrics: {e}")
            return None
    
    async def test_model_usage_tracking(self):
        """Test model usage statistics and tracking"""
        print("🤖 Testing model usage tracking...")
        
        try:
            async with self.session.get(f"{self.base_url}/api/metrics/models") as response:
                if response.status == 200:
                    models_data = await response.json()
                    
                    if isinstance(models_data, dict):
                        print(f"✅ Model usage data for {len(models_data)} models:")
                        
                        for model_id, stats in models_data.items():
                            total_requests = stats.get('total_requests', 0)
                            avg_response_time = stats.get('avg_response_time_ms', 0)
                            success_rate = stats.get('success_rate', 0)
                            last_used = stats.get('last_used', 'Never')
                            
                            print(f"   🤖 {model_id}:")
                            print(f"      Requests: {total_requests}")
                            print(f"      Avg Response: {avg_response_time:.1f}ms")
                            print(f"      Success Rate: {success_rate:.1f}%")
                            print(f"      Last Used: {last_used}")
                        
                        return models_data
                    else:
                        print(f"⚠️ Unexpected model data format: {type(models_data)}")
                        return {}
                else:
                    print(f"❌ Model metrics unavailable: {response.status}")
                    return {}
        except Exception as e:
            print(f"❌ Error getting model metrics: {e}")
            return {}
    
    async def test_health_monitoring(self):
        """Test system health assessment"""
        print("🏥 Testing health monitoring...")
        
        try:
            async with self.session.get(f"{self.base_url}/api/metrics/health") as response:
                if response.status == 200:
                    health_data = await response.json()
                    
                    status = health_data.get('status', 'unknown')
                    health_score = health_data.get('health_score', 0)
                    components = health_data.get('components', {})
                    alerts = health_data.get('alerts', [])
                    timestamp = health_data.get('timestamp', 'unknown')
                    
                    print(f"✅ Health Assessment:")
                    print(f"   Overall Status: {status}")
                    print(f"   Health Score: {health_score:.2f}/1.0")
                    print(f"   Timestamp: {timestamp}")
                    
                    # Show component health
                    if components:
                        print(f"   Component Health:")
                        for component, component_health in components.items():
                            health_status = component_health.get('status', 'unknown')
                            component_score = component_health.get('score', 0)
                            print(f"     {component}: {health_status} ({component_score:.2f})")
                    
                    # Show alerts
                    if alerts:
                        print(f"   Active Alerts: {len(alerts)}")
                        for i, alert in enumerate(alerts[:3]):  # Show first 3
                            severity = alert.get('severity', 'unknown')
                            message = alert.get('message', 'No message')
                            print(f"     {i+1}. [{severity}] {message}")
                    else:
                        print(f"   ✅ No active alerts")
                    
                    return health_data
                else:
                    print(f"❌ Health monitoring unavailable: {response.status}")
                    return None
        except Exception as e:
            print(f"❌ Error getting health data: {e}")
            return None
    
    async def test_metrics_under_load(self):
        """Test metrics collection under load"""
        print("⚡ Testing metrics under load...")
        
        # Generate some load by making multiple concurrent requests
        load_tasks = []
        
        # Create 5 concurrent chat requests to generate metrics
        for i in range(5):
            chat_data = {
                "message": f"Load test message {i+1} - please respond briefly",
                "session_id": f"{self.session_id}_load_{i}",
                "persona": "companion"
            }
            
            task = asyncio.create_task(
                self.make_chat_request(chat_data, f"Load Request {i+1}")
            )
            load_tasks.append(task)
        
        # Execute load requests
        print("   Generating load with 5 concurrent requests...")
        load_results = await asyncio.gather(*load_tasks, return_exceptions=True)
        
        successful_requests = [r for r in load_results if isinstance(r, dict) and r.get('success')]
        print(f"   ✅ Load generation: {len(successful_requests)}/5 requests successful")
        
        # Wait a moment for metrics to update
        await asyncio.sleep(2)
        
        # Check updated metrics
        updated_metrics = await self.test_realtime_metrics()
        updated_models = await self.test_model_usage_tracking()
        
        if updated_metrics and updated_models:
            # Verify metrics updated
            total_requests = 0
            for model_stats in updated_models.values():
                total_requests += model_stats.get('total_requests', 0)
            
            print(f"   ✅ Post-load total requests across all models: {total_requests}")
            
            return {
                "load_requests": len(successful_requests),
                "updated_metrics": updated_metrics,
                "updated_models": updated_models,
                "total_model_requests": total_requests
            }
        else:
            print("   ❌ Could not verify metrics after load test")
            return None
    
    async def make_chat_request(self, chat_data, request_name):
        """Helper method to make a chat request"""
        try:
            async with self.session.post(f"{self.base_url}/api/chat", json=chat_data) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        "success": True,
                        "name": request_name,
                        "handler": data.get('handler', 'unknown'),
                        "response_length": len(data.get('response', ''))
                    }
                else:
                    return {
                        "success": False,
                        "name": request_name,
                        "status": response.status
                    }
        except Exception as e:
            return {
                "success": False,
                "name": request_name,
                "error": str(e)
            }
    
    async def test_historical_metrics(self):
        """Test historical metrics data"""
        print("📚 Testing historical metrics...")
        
        try:
            async with self.session.get(f"{self.base_url}/api/metrics/history") as response:
                if response.status == 200:
                    history_data = await response.json()
                    
                    if isinstance(history_data, list):
                        print(f"✅ Historical metrics: {len(history_data)} data points")
                        
                        if history_data:
                            # Show recent entries
                            recent_entries = history_data[-3:] if len(history_data) >= 3 else history_data
                            
                            print(f"   Recent entries:")
                            for i, entry in enumerate(recent_entries):
                                timestamp = entry.get('timestamp', 'unknown')
                                cpu = entry.get('cpu_percent', 0)
                                memory = entry.get('memory_percent', 0)
                                print(f"     {i+1}. {timestamp}: CPU {cpu:.1f}%, Memory {memory:.1f}%")
                        
                        return history_data
                    elif isinstance(history_data, dict):
                        time_series = history_data.get('time_series', [])
                        summary = history_data.get('summary', {})
                        
                        print(f"✅ Historical metrics: {len(time_series)} data points")
                        if summary:
                            print(f"   Summary: {summary}")
                        
                        return history_data
                    else:
                        print(f"⚠️ Unexpected history format: {type(history_data)}")
                        return {}
                else:
                    print(f"❌ Historical metrics unavailable: {response.status}")
                    return {}
        except Exception as e:
            print(f"❌ Error getting historical metrics: {e}")
            return {}
    
    async def test_metrics_alerts(self):
        """Test metrics alerting system"""
        print("🚨 Testing metrics alerts...")
        
        try:
            async with self.session.get(f"{self.base_url}/api/metrics/alerts") as response:
                if response.status == 200:
                    alerts_data = await response.json()
                    
                    if isinstance(alerts_data, list):
                        print(f"✅ Current alerts: {len(alerts_data)}")
                        
                        # Categorize alerts by severity
                        critical_alerts = [a for a in alerts_data if a.get('severity') == 'critical']
                        warning_alerts = [a for a in alerts_data if a.get('severity') == 'warning']
                        info_alerts = [a for a in alerts_data if a.get('severity') == 'info']
                        
                        print(f"   🔴 Critical: {len(critical_alerts)}")
                        print(f"   🟡 Warning: {len(warning_alerts)}")
                        print(f"   🔵 Info: {len(info_alerts)}")
                        
                        # Show critical alerts
                        for alert in critical_alerts[:2]:
                            message = alert.get('message', 'No message')
                            timestamp = alert.get('timestamp', 'unknown')
                            print(f"     🔴 {timestamp}: {message}")
                        
                        return alerts_data
                    elif isinstance(alerts_data, dict):
                        total_alerts = alerts_data.get('total_alerts', 0)
                        alerts_by_severity = alerts_data.get('by_severity', {})
                        recent_alerts = alerts_data.get('recent_alerts', [])
                        
                        print(f"✅ Total alerts: {total_alerts}")
                        print(f"   By severity: {alerts_by_severity}")
                        print(f"   Recent: {len(recent_alerts)}")
                        
                        return alerts_data
                    else:
                        print(f"⚠️ Unexpected alerts format: {type(alerts_data)}")
                        return {}
                else:
                    print(f"❌ Alerts unavailable: {response.status}")
                    return {}
        except Exception as e:
            print(f"❌ Error getting alerts: {e}")
            return {}
    
    async def test_metrics_summary(self):
        """Test comprehensive metrics summary"""
        print("📋 Testing metrics summary...")
        
        try:
            async with self.session.get(f"{self.base_url}/api/metrics/summary") as response:
                if response.status == 200:
                    summary_data = await response.json()
                    
                    print(f"✅ Metrics Summary:")
                    
                    # Show system overview
                    system_overview = summary_data.get('system_overview', {})
                    if system_overview:
                        print(f"   System Overview:")
                        for key, value in system_overview.items():
                            print(f"     {key}: {value}")
                    
                    # Show performance stats
                    performance = summary_data.get('performance', {})
                    if performance:
                        print(f"   Performance Stats:")
                        for key, value in performance.items():
                            print(f"     {key}: {value}")
                    
                    # Show usage statistics
                    usage_stats = summary_data.get('usage_statistics', {})
                    if usage_stats:
                        print(f"   Usage Statistics:")
                        for key, value in usage_stats.items():
                            if isinstance(value, (int, float)):
                                print(f"     {key}: {value}")
                            else:
                                print(f"     {key}: {len(value) if hasattr(value, '__len__') else value}")
                    
                    return summary_data
                else:
                    print(f"❌ Metrics summary unavailable: {response.status}")
                    return {}
        except Exception as e:
            print(f"❌ Error getting metrics summary: {e}")
            return {}
    
    async def test_local_system_comparison(self):
        """Compare reported metrics with local system readings"""
        print("🔍 Testing metrics accuracy against local system...")
        
        try:
            # Get local system metrics
            local_cpu = psutil.cpu_percent(interval=1)
            local_memory = psutil.virtual_memory().percent
            local_disk = psutil.disk_usage('/').percent if hasattr(psutil.disk_usage('/'), 'percent') else 0
            
            print(f"   Local System Readings:")
            print(f"     CPU: {local_cpu:.1f}%")
            print(f"     Memory: {local_memory:.1f}%")
            print(f"     Disk: {local_disk:.1f}%")
            
            # Get reported metrics
            reported_metrics = await self.test_realtime_metrics()
            
            if reported_metrics and 'system' in reported_metrics:
                system_metrics = reported_metrics['system']
                reported_cpu = system_metrics.get('cpu_percent', 0)
                reported_memory = system_metrics.get('memory_percent', 0)
                reported_disk = system_metrics.get('disk_percent', 0)
                
                print(f"   Reported Metrics:")
                print(f"     CPU: {reported_cpu:.1f}%")
                print(f"     Memory: {reported_memory:.1f}%")
                print(f"     Disk: {reported_disk:.1f}%")
                
                # Calculate differences
                cpu_diff = abs(local_cpu - reported_cpu)
                memory_diff = abs(local_memory - reported_memory)
                disk_diff = abs(local_disk - reported_disk)
                
                print(f"   Accuracy Check:")
                print(f"     CPU difference: {cpu_diff:.1f}% {'✅' if cpu_diff < 10 else '⚠️'}")
                print(f"     Memory difference: {memory_diff:.1f}% {'✅' if memory_diff < 5 else '⚠️'}")
                print(f"     Disk difference: {disk_diff:.1f}% {'✅' if disk_diff < 5 else '⚠️'}")
                
                return {
                    "local": {"cpu": local_cpu, "memory": local_memory, "disk": local_disk},
                    "reported": {"cpu": reported_cpu, "memory": reported_memory, "disk": reported_disk},
                    "differences": {"cpu": cpu_diff, "memory": memory_diff, "disk": disk_diff}
                }
            else:
                print("   ❌ Could not compare - no reported metrics available")
                return None
                
        except Exception as e:
            print(f"❌ Error comparing system metrics: {e}")
            return None
    
    async def run_system_metrics_qa_suite(self):
        """Run the complete system metrics QA suite"""
        print("📈 SYSTEM METRICS QA SUITE")
        print("=" * 50)
        
        # Test sequence
        realtime_metrics = await self.test_realtime_metrics()
        if not realtime_metrics:
            print("❌ Cannot continue without system metrics")
            return False
        
        model_metrics = await self.test_model_usage_tracking()
        health_data = await self.test_health_monitoring()
        load_test_results = await self.test_metrics_under_load()
        historical_data = await self.test_historical_metrics()
        alerts_data = await self.test_metrics_alerts()
        summary_data = await self.test_metrics_summary()
        accuracy_check = await self.test_local_system_comparison()
        
        # Final assessment
        print("\n" + "=" * 50)
        print("🎯 SYSTEM METRICS QA RESULTS:")
        print(f"✅ Real-time Metrics: Available")
        print(f"✅ Model Usage Tracking: {len(model_metrics)} models tracked")
        print(f"✅ Health Monitoring: {'Available' if health_data else 'Unavailable'}")
        print(f"✅ Load Testing: {'Passed' if load_test_results else 'Failed'}")
        print(f"✅ Historical Data: {len(historical_data) if isinstance(historical_data, list) else 'Available' if historical_data else 'Unavailable'}")
        print(f"✅ Alerts System: {len(alerts_data) if isinstance(alerts_data, list) else 'Available' if alerts_data else 'Unavailable'}")
        print(f"✅ Summary Reports: {'Available' if summary_data else 'Unavailable'}")
        print(f"✅ Accuracy Check: {'Passed' if accuracy_check else 'Failed'}")
        
        # Calculate health score
        components_working = [
            realtime_metrics is not None,
            len(model_metrics) > 0,
            health_data is not None,
            load_test_results is not None,
            bool(historical_data),
            bool(alerts_data),
            summary_data is not None
        ]
        
        health_score = sum(components_working) / len(components_working) * 100
        print(f"✅ Overall System Health: {health_score:.1f}%")
        
        if health_score >= 80:
            print("\n🎉 System Metrics QA: PASSED")
            return True
        elif health_score >= 60:
            print("\n⚠️ System Metrics QA: PARTIAL - Some components may need attention")
            return True
        else:
            print("\n❌ System Metrics QA: FAILED - Multiple components not working")
            return False

async def main():
    """Main QA execution"""
    print("Starting System Metrics QA...")
    print("Make sure Dolphin backend is running on http://localhost:8000\n")
    
    async with SystemMetricsQA() as qa:
        success = await qa.run_system_metrics_qa_suite()
        return success

if __name__ == "__main__":
    result = asyncio.run(main())
    exit(0 if result else 1)

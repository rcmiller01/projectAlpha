#!/usr/bin/env python3
"""
Autopilot State Management - Handles crash recovery and shutdown resumption
Manages persistent state for autonomous operation continuity
"""

import json
import time
import signal
import threading
import psutil
import sys
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict

# Fix Windows console encoding for Unicode characters
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

@dataclass
class AutopilotState:
    """Current autopilot operational state"""
    is_running: bool = False
    current_job_id: Optional[str] = None
    current_run_id: Optional[str] = None
    last_activity: Optional[str] = None
    jobs_completed_today: int = 0
    system_load_average: float = 0.0
    memory_usage_percent: float = 0.0
    emergency_stop_triggered: bool = False
    pause_until: Optional[str] = None
    last_checkpoint: Optional[str] = None

class AutopilotStateManager:
    """
    Manages autopilot state persistence and recovery
    Handles graceful shutdown, crash recovery, and resource monitoring
    """
    
    def __init__(self, state_file: str = "autopilot_state.json", config: Optional[Dict[str, Any]] = None):
        self.state_file = Path(state_file)
        self.config = config or {}
        
        # State management
        self.state = AutopilotState()
        self.state_lock = threading.Lock()
        
        # Resource monitoring
        self.cpu_threshold = self.config.get("cpu_threshold_percent", 80)
        self.memory_threshold = self.config.get("memory_threshold_percent", 85)
        self.monitoring_interval = self.config.get("monitoring_interval_seconds", 30)
        
        # Watchdog configuration
        self.enable_watchdog = self.config.get("enable_watchdog", True)
        self.watchdog_thread: Optional[threading.Thread] = None
        self.shutdown_event = threading.Event()
        
        # Load existing state
        self._load_state()
        
        # Setup signal handlers for graceful shutdown
        self._setup_signal_handlers()
        
        print(f"🔄 AutopilotStateManager initialized")
        print(f"   State file: {self.state_file}")
        print(f"   Watchdog enabled: {self.enable_watchdog}")
    
    def _load_state(self) -> None:
        """Load state from persistent storage"""
        try:
            if self.state_file.exists():
                with open(self.state_file, 'r') as f:
                    state_data = json.load(f)
                
                # Update state object
                for key, value in state_data.items():
                    if hasattr(self.state, key):
                        setattr(self.state, key, value)
                
                print(f"📂 Loaded autopilot state from {self.state_file}")
                
                # Check if we were running when we shut down
                if self.state.is_running:
                    print("⚠️ Detected unclean shutdown - autopilot was running")
                    self.state.is_running = False  # Reset running state
                    self._save_state()
            else:
                print("🆕 No existing state file - starting fresh")
                self._save_state()
                
        except Exception as e:
            print(f"❌ Failed to load state: {e}")
            self.state = AutopilotState()  # Reset to default
    
    def _save_state(self) -> None:
        """Save state to persistent storage"""
        try:
            with self.state_lock:
                state_dict = asdict(self.state)
                state_dict['last_checkpoint'] = datetime.now().isoformat()
                
                # Atomic write
                temp_file = self.state_file.with_suffix('.tmp')
                with open(temp_file, 'w') as f:
                    json.dump(state_dict, f, indent=2)
                
                temp_file.replace(self.state_file)
                
        except Exception as e:
            print(f"❌ Failed to save state: {e}")
    
    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            print(f"\n🛑 Received signal {signum} - initiating graceful shutdown")
            self.initiate_shutdown()
        
        try:
            signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
            signal.signal(signal.SIGTERM, signal_handler)  # Termination request
            print("📡 Signal handlers registered")
        except Exception as e:
            print(f"⚠️ Could not register signal handlers: {e}")
    
    def start_monitoring(self) -> None:
        """Start the resource monitoring watchdog"""
        if not self.enable_watchdog:
            return
        
        if self.watchdog_thread and self.watchdog_thread.is_alive():
            return
        
        self.shutdown_event.clear()
        self.watchdog_thread = threading.Thread(target=self._watchdog_loop, daemon=True)
        self.watchdog_thread.start()
        
        print("👁️ Resource monitoring watchdog started")
    
    def stop_monitoring(self) -> None:
        """Stop the resource monitoring watchdog"""
        if self.watchdog_thread:
            self.shutdown_event.set()
            self.watchdog_thread.join(timeout=5)
            print("🛑 Resource monitoring watchdog stopped")
    
    def _watchdog_loop(self) -> None:
        """Main watchdog monitoring loop"""
        while not self.shutdown_event.is_set():
            try:
                self._update_system_metrics()
                self._check_resource_limits()
                self._save_state()
                
            except Exception as e:
                print(f"⚠️ Watchdog error: {e}")
            
            # Wait for next check or shutdown signal
            if self.shutdown_event.wait(timeout=self.monitoring_interval):
                break
    
    def _update_system_metrics(self) -> None:
        """Update current system metrics"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            with self.state_lock:
                self.state.system_load_average = cpu_percent
                self.state.memory_usage_percent = memory_percent
                self.state.last_activity = datetime.now().isoformat()
                
        except Exception as e:
            print(f"⚠️ Failed to update system metrics: {e}")
    
    def _check_resource_limits(self) -> None:
        """Check if system resources exceed limits"""
        try:
            should_pause = False
            pause_reason = ""
            
            # Check CPU usage
            if self.state.system_load_average > self.cpu_threshold:
                should_pause = True
                pause_reason = f"High CPU usage: {self.state.system_load_average:.1f}%"
            
            # Check memory usage
            if self.state.memory_usage_percent > self.memory_threshold:
                should_pause = True
                pause_reason = f"High memory usage: {self.state.memory_usage_percent:.1f}%"
            
            # Check for emergency stop file
            emergency_file = Path(self.config.get("emergency_stop_file", "emergency_stop.flag"))
            if emergency_file.exists():
                should_pause = True
                pause_reason = "Emergency stop file detected"
                self.state.emergency_stop_triggered = True
            
            # Pause if needed
            if should_pause and not self.is_paused():
                self.pause_operations(pause_reason, duration_minutes=10)
            
            # Resume if conditions are good and we were paused due to resources
            elif not should_pause and self.is_paused() and not self.state.emergency_stop_triggered:
                if self.state.pause_until and datetime.now() > datetime.fromisoformat(self.state.pause_until):
                    self.resume_operations("Resource conditions normalized")
                    
        except Exception as e:
            print(f"⚠️ Failed to check resource limits: {e}")
    
    def start_autopilot(self, job_id: Optional[str] = None, run_id: Optional[str] = None) -> None:
        """Mark autopilot as started"""
        with self.state_lock:
            self.state.is_running = True
            self.state.current_job_id = job_id
            self.state.current_run_id = run_id
            self.state.last_activity = datetime.now().isoformat()
        
        self._save_state()
        print(f"🚀 Autopilot started - Job: {job_id}, Run: {run_id}")
    
    def stop_autopilot(self) -> None:
        """Mark autopilot as stopped"""
        with self.state_lock:
            self.state.is_running = False
            self.state.current_job_id = None
            self.state.current_run_id = None
        
        self._save_state()
        print("🛑 Autopilot stopped")
    
    def update_job_progress(self, job_id: str, run_id: str) -> None:
        """Update current job progress"""
        with self.state_lock:
            self.state.current_job_id = job_id
            self.state.current_run_id = run_id
            self.state.last_activity = datetime.now().isoformat()
        
        self._save_state()
    
    def complete_job(self, success: bool = True) -> None:
        """Mark current job as completed"""
        with self.state_lock:
            if success:
                self.state.jobs_completed_today += 1
            self.state.current_job_id = None
            self.state.current_run_id = None
            self.state.last_activity = datetime.now().isoformat()
        
        self._save_state()
        print(f"✅ Job completed - Total today: {self.state.jobs_completed_today}")
    
    def pause_operations(self, reason: str, duration_minutes: int = 60) -> None:
        """Pause operations for specified duration"""
        pause_until = datetime.now() + timedelta(minutes=duration_minutes)
        
        with self.state_lock:
            self.state.pause_until = pause_until.isoformat()
        
        self._save_state()
        print(f"⏸️ Operations paused: {reason}")
        print(f"   Resuming at: {pause_until.strftime('%H:%M:%S')}")
    
    def resume_operations(self, reason: str = "Manual resume") -> None:
        """Resume paused operations"""
        with self.state_lock:
            self.state.pause_until = None
            self.state.emergency_stop_triggered = False
        
        self._save_state()
        print(f"▶️ Operations resumed: {reason}")
    
    def is_paused(self) -> bool:
        """Check if operations are currently paused"""
        if not self.state.pause_until:
            return False
        
        return datetime.now() < datetime.fromisoformat(self.state.pause_until)
    
    def is_running(self) -> bool:
        """Check if autopilot is currently running"""
        return self.state.is_running and not self.is_paused()
    
    def can_start_new_job(self, daily_limit: int = 3) -> bool:
        """Check if we can start a new job based on daily limits"""
        return (not self.is_paused() and 
                self.state.jobs_completed_today < daily_limit and
                not self.state.emergency_stop_triggered)
    
    def get_recovery_info(self) -> Dict[str, Any]:
        """Get information needed for crash recovery"""
        return {
            "was_running": self.state.is_running,
            "current_job_id": self.state.current_job_id,
            "current_run_id": self.state.current_run_id,
            "last_activity": self.state.last_activity,
            "jobs_completed_today": self.state.jobs_completed_today,
            "is_paused": self.is_paused(),
            "pause_reason": "Resource limits" if self.is_paused() else None
        }
    
    def reset_daily_counters(self) -> None:
        """Reset daily job counters (typically called at midnight)"""
        with self.state_lock:
            self.state.jobs_completed_today = 0
        
        self._save_state()
        print("🔄 Daily counters reset")
    
    def initiate_shutdown(self) -> None:
        """Initiate graceful shutdown"""
        print("🛑 Initiating graceful shutdown...")
        
        # Stop monitoring
        self.stop_monitoring()
        
        # Update state
        with self.state_lock:
            self.state.is_running = False
            self.state.current_job_id = None
            self.state.current_run_id = None
        
        # Save final state
        self._save_state()
        
        print("✅ Graceful shutdown complete")
    
    def get_status_summary(self) -> Dict[str, Any]:
        """Get comprehensive status summary"""
        return {
            "is_running": self.is_running(),
            "is_paused": self.is_paused(),
            "current_job": self.state.current_job_id,
            "current_run": self.state.current_run_id,
            "jobs_today": self.state.jobs_completed_today,
            "last_activity": self.state.last_activity,
            "system_metrics": {
                "cpu_percent": self.state.system_load_average,
                "memory_percent": self.state.memory_usage_percent
            },
            "emergency_stop": self.state.emergency_stop_triggered,
            "pause_until": self.state.pause_until,
            "watchdog_active": self.watchdog_thread.is_alive() if self.watchdog_thread else False
        }

if __name__ == "__main__":
    # Test the state manager
    import time
    
    config = {
        "cpu_threshold_percent": 70,
        "memory_threshold_percent": 80,
        "monitoring_interval_seconds": 5,
        "emergency_stop_file": "test_emergency_stop.flag"
    }
    
    manager = AutopilotStateManager("test_state.json", config)
    
    # Test basic operations
    print("\n🧪 Testing basic operations...")
    manager.start_monitoring()
    
    manager.start_autopilot("test_job_123", "test_run_456")
    time.sleep(2)
    
    manager.update_job_progress("test_job_123", "test_run_456")
    time.sleep(2)
    
    manager.complete_job(success=True)
    time.sleep(2)
    
    # Test pause/resume
    print("\n🧪 Testing pause/resume...")
    manager.pause_operations("Testing pause functionality", duration_minutes=1)
    print(f"Is paused: {manager.is_paused()}")
    
    time.sleep(2)
    manager.resume_operations("Testing resume functionality")
    print(f"Is paused: {manager.is_paused()}")
    
    # Show status
    print("\n📊 Final status:")
    status = manager.get_status_summary()
    for key, value in status.items():
        print(f"  {key}: {value}")
    
    # Cleanup
    manager.initiate_shutdown()
    
    print("\n🎉 State manager testing complete!")

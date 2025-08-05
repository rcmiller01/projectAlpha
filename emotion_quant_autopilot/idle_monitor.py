#!/usr/bin/env python3
"""
Idle System Monitor for Emotion Quantization Autopilot
Monitors system resources and user activity to determine optimal execution windows
"""

import time
import psutil
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, Optional, Callable
from dataclasses import dataclass
from pathlib import Path

try:
    from pynput import mouse, keyboard
    PYNPUT_AVAILABLE = True
except ImportError:
    PYNPUT_AVAILABLE = False
    logging.warning("pynput not available - using CPU/memory monitoring only")

if PYNPUT_AVAILABLE:
    from typing import Union
    ListenerType = Union[mouse.Listener, keyboard.Listener]
else:
    ListenerType = None

@dataclass
class SystemMetrics:
    """System resource metrics snapshot"""
    cpu_percent: float
    memory_percent: float
    disk_free_gb: float
    active_processes: int
    last_user_activity: Optional[datetime] = None

@dataclass
class IdleConfig:
    """Configuration for idle monitoring"""
    min_idle_minutes: int = 30
    cpu_threshold_percent: float = 15.0
    memory_threshold_percent: float = 80.0
    disk_threshold_gb: float = 50.0
    check_interval_seconds: int = 300  # 5 minutes
    user_activity_timeout_minutes: int = 15

class IdleMonitor:
    """
    Monitors system idle state for autonomous quantization execution
    
    Features:
    - CPU and memory usage monitoring
    - User input activity detection (mouse/keyboard)
    - Disk space verification
    - Configurable thresholds and timeouts
    - Thread-safe operation
    """
    
    def __init__(self, config: IdleConfig, log_directory: Optional[str] = None):
        self.config = config
        self.log_directory = Path(log_directory) if log_directory else Path("emotion_quant_autopilot/logs")
        
        # Ensure log directory exists
        self.log_directory.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
        # Activity tracking
        self.last_user_activity = datetime.now()
        self.is_monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        
        # User activity listeners
        self.mouse_listener: Optional[mouse.Listener] = None
        self.keyboard_listener: Optional[keyboard.Listener] = None
        
        # Callbacks
        self.idle_callback: Optional[Callable] = None
        self.active_callback: Optional[Callable] = None
        
        # State tracking
        self.current_state = "unknown"  # "idle", "active", "unknown"
        self.state_change_time = datetime.now()
        
        self.logger.info("🔍 Idle Monitor initialized")
        self.logger.info(f"   Min idle time: {self.config.min_idle_minutes} minutes")
        self.logger.info(f"   CPU threshold: {self.config.cpu_threshold_percent}%")
        self.logger.info(f"   Memory threshold: {self.config.memory_threshold_percent}%")
    
    def _setup_logging(self) -> None:
        """Setup logging configuration"""
        log_file = self.log_directory / f"idle_monitor_{datetime.now().strftime('%Y%m%d')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger("IdleMonitor")
    
    def _on_user_activity(self) -> None:
        """Called when user input is detected"""
        self.last_user_activity = datetime.now()
        
        if self.current_state == "idle":
            self.logger.info("👆 User activity detected - switching to active state")
            self._change_state("active")
    
    def _on_mouse_move(self, x: int, y: int) -> None:
        """Mouse movement callback"""
        self._on_user_activity()
    
    def _on_mouse_click(self, x: int, y: int, button, pressed: bool) -> None:
        """Mouse click callback"""
        if pressed:
            self._on_user_activity()
    
    def _on_key_press(self, key) -> None:
        """Keyboard press callback"""
        self._on_user_activity()
    
    def _change_state(self, new_state: str) -> None:
        """Change system state and trigger callbacks"""
        if new_state != self.current_state:
            old_state = self.current_state
            self.current_state = new_state
            self.state_change_time = datetime.now()
            
            self.logger.info(f"🔄 State change: {old_state} → {new_state}")
            
            # Trigger callbacks
            if new_state == "idle" and self.idle_callback:
                try:
                    self.idle_callback()
                except Exception as e:
                    self.logger.error(f"❌ Error in idle callback: {e}")
            
            elif new_state == "active" and self.active_callback:
                try:
                    self.active_callback()
                except Exception as e:
                    self.logger.error(f"❌ Error in active callback: {e}")
    
    def get_system_metrics(self) -> SystemMetrics:
        """Get current system resource metrics"""
        # CPU usage (1-second average)
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # Memory usage
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        
        # Disk space (for root/primary drive)
        import os
        if os.name == 'nt':  # Windows
            disk = psutil.disk_usage('C:\\')
        else:  # Unix/Linux
            disk = psutil.disk_usage('/')
        disk_free_gb = disk.free / (1024**3)
        
        # Active process count
        active_processes = len(psutil.pids())
        
        return SystemMetrics(
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            disk_free_gb=disk_free_gb,
            active_processes=active_processes,
            last_user_activity=self.last_user_activity
        )
    
    def is_system_idle(self) -> bool:
        """
        Determine if system is idle based on all criteria
        
        Returns:
            bool: True if system meets all idle criteria
        """
        metrics = self.get_system_metrics()
        now = datetime.now()
        
        # Check user activity timeout
        if metrics.last_user_activity:
            user_idle_minutes = (now - metrics.last_user_activity).total_seconds() / 60
            user_idle_ok = user_idle_minutes >= self.config.min_idle_minutes
        else:
            user_idle_ok = True  # No activity recorded yet
        
        # Check system resource usage
        cpu_ok = metrics.cpu_percent <= self.config.cpu_threshold_percent
        memory_ok = metrics.memory_percent <= self.config.memory_threshold_percent
        disk_ok = metrics.disk_free_gb >= self.config.disk_threshold_gb
        
        # Log current status
        if not user_idle_ok:
            self.logger.debug(f"⏱️ User activity too recent: {user_idle_minutes:.1f} min (need {self.config.min_idle_minutes})")
        if not cpu_ok:
            self.logger.debug(f"💻 CPU usage too high: {metrics.cpu_percent:.1f}% (max {self.config.cpu_threshold_percent}%)")
        if not memory_ok:
            self.logger.debug(f"🧠 Memory usage too high: {metrics.memory_percent:.1f}% (max {self.config.memory_threshold_percent}%)")
        if not disk_ok:
            self.logger.debug(f"💽 Disk space too low: {metrics.disk_free_gb:.1f}GB (min {self.config.disk_threshold_gb}GB)")
        
        is_idle = user_idle_ok and cpu_ok and memory_ok and disk_ok
        
        return is_idle
    
    def start_monitoring(self) -> None:
        """Start monitoring system idle state"""
        if self.is_monitoring:
            self.logger.warning("⚠️ Monitor already running")
            return
        
        self.is_monitoring = True
        self.logger.info("🚀 Starting idle monitoring")
        
        # Start user activity listeners if available
        if PYNPUT_AVAILABLE:
            try:
                self.mouse_listener = mouse.Listener(
                    on_move=self._on_mouse_move,
                    on_click=self._on_mouse_click
                )
                self.mouse_listener.start()
                
                self.keyboard_listener = keyboard.Listener(
                    on_press=self._on_key_press
                )
                self.keyboard_listener.start()
                
                self.logger.info("👁️ User activity monitoring enabled")
                
            except Exception as e:
                self.logger.warning(f"⚠️ Could not start user activity monitoring: {e}")
        
        # Start monitoring thread
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
    
    def stop_monitoring(self) -> None:
        """Stop monitoring system idle state"""
        if not self.is_monitoring:
            return
        
        self.logger.info("🛑 Stopping idle monitoring")
        self.is_monitoring = False
        
        # Stop user activity listeners
        if self.mouse_listener:
            self.mouse_listener.stop()
            self.mouse_listener = None
        
        if self.keyboard_listener:
            self.keyboard_listener.stop()
            self.keyboard_listener = None
        
        # Wait for monitoring thread to finish
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop (runs in separate thread)"""
        self.logger.info("🔄 Monitoring loop started")
        
        while self.is_monitoring:
            try:
                is_idle = self.is_system_idle()
                
                # Determine new state
                if is_idle and self.current_state != "idle":
                    self._change_state("idle")
                elif not is_idle and self.current_state != "active":
                    self._change_state("active")
                
                # Log status periodically
                metrics = self.get_system_metrics()
                self.logger.debug(
                    f"📊 System status: {self.current_state} | "
                    f"CPU: {metrics.cpu_percent:.1f}% | "
                    f"MEM: {metrics.memory_percent:.1f}% | "
                    f"DISK: {metrics.disk_free_gb:.1f}GB"
                )
                
                time.sleep(self.config.check_interval_seconds)
                
            except Exception as e:
                self.logger.error(f"❌ Error in monitoring loop: {e}")
                time.sleep(60)  # Wait a minute on error
        
        self.logger.info("✅ Monitoring loop stopped")
    
    def set_idle_callback(self, callback: Callable) -> None:
        """Set callback function for when system becomes idle"""
        self.idle_callback = callback
        self.logger.info("📞 Idle callback registered")
    
    def set_active_callback(self, callback: Callable) -> None:
        """Set callback function for when system becomes active"""
        self.active_callback = callback
        self.logger.info("📞 Active callback registered")
    
    def get_idle_status(self) -> Dict:
        """Get current idle status information"""
        metrics = self.get_system_metrics()
        now = datetime.now()
        
        user_idle_minutes = 0
        if metrics.last_user_activity:
            user_idle_minutes = (now - metrics.last_user_activity).total_seconds() / 60
        
        state_duration_minutes = (now - self.state_change_time).total_seconds() / 60
        
        return {
            "current_state": self.current_state,
            "state_duration_minutes": state_duration_minutes,
            "user_idle_minutes": user_idle_minutes,
            "is_idle": self.is_system_idle(),
            "metrics": {
                "cpu_percent": metrics.cpu_percent,
                "memory_percent": metrics.memory_percent,
                "disk_free_gb": metrics.disk_free_gb,
                "active_processes": metrics.active_processes
            },
            "thresholds": {
                "min_idle_minutes": self.config.min_idle_minutes,
                "cpu_threshold": self.config.cpu_threshold_percent,
                "memory_threshold": self.config.memory_threshold_percent,
                "disk_threshold": self.config.disk_threshold_gb
            }
        }
    
    def force_idle_state(self) -> None:
        """Force system into idle state (for testing)"""
        self.logger.warning("⚠️ Forcing idle state (override)")
        self._change_state("idle")
    
    def force_active_state(self) -> None:
        """Force system into active state (for testing)"""
        self.logger.warning("⚠️ Forcing active state (override)")
        self._change_state("active")

def create_idle_monitor_from_config(config_dict: Dict) -> IdleMonitor:
    """Create IdleMonitor from configuration dictionary"""
    system_config = config_dict.get("system_monitoring", {})
    
    idle_config = IdleConfig(
        min_idle_minutes=config_dict.get("min_idle_minutes", 30),
        cpu_threshold_percent=system_config.get("cpu_threshold_percent", 15.0),
        memory_threshold_percent=system_config.get("memory_threshold_percent", 80.0),
        disk_threshold_gb=system_config.get("disk_space_threshold_gb", 50.0),
        check_interval_seconds=system_config.get("check_interval_seconds", 300)
    )
    
    log_dir = config_dict.get("output_paths", {}).get("logs_directory", "emotion_quant_autopilot/logs")
    
    return IdleMonitor(idle_config, log_dir)

# CLI interface for testing
if __name__ == "__main__":
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description="Idle Monitor for Emotion Quantization")
    parser.add_argument("--config", default="autopilot_config.json", help="Configuration file")
    parser.add_argument("--test", action="store_true", help="Run in test mode")
    parser.add_argument("--duration", type=int, default=60, help="Test duration in seconds")
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        with open(args.config, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"❌ Config file not found: {args.config}")
        exit(1)
    
    # Create and start monitor
    monitor = create_idle_monitor_from_config(config)
    
    if args.test:
        print(f"🧪 Running idle monitor test for {args.duration} seconds...")
        
        def on_idle():
            print("💤 System entered idle state")
        
        def on_active():
            print("🏃 System entered active state")
        
        monitor.set_idle_callback(on_idle)
        monitor.set_active_callback(on_active)
        
        monitor.start_monitoring()
        
        try:
            time.sleep(args.duration)
        except KeyboardInterrupt:
            pass
        
        monitor.stop_monitoring()
        
        # Print final status
        status = monitor.get_idle_status()
        print("\n📊 Final Status:")
        print(json.dumps(status, indent=2, default=str))
    
    else:
        print("🔍 Starting continuous idle monitoring...")
        print("Press Ctrl+C to stop")
        
        monitor.start_monitoring()
        
        try:
            while True:
                status = monitor.get_idle_status()
                print(f"State: {status['current_state']} | "
                      f"CPU: {status['metrics']['cpu_percent']:.1f}% | "
                      f"MEM: {status['metrics']['memory_percent']:.1f}%")
                time.sleep(30)
        except KeyboardInterrupt:
            pass
        
        monitor.stop_monitoring()
        print("✅ Monitoring stopped")

#!/usr/bin/env python3
"""
Autopilot Bootloader - Autonomous Launch System
Monitors system usage and schedules background quantization sessions
"""

import json
import time
import subprocess
import psutil
import logging
import signal
import sys
import os
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict

# Import quantization tracking
try:
    from quant_tracking import (
        QuantLoopResult, save_loop_result, 
        eval_emotion, eval_fluency, get_tracker
    )
    TRACKING_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Quantization tracking not available: {e}")
    TRACKING_AVAILABLE = False

# Import model judging system
try:
    from judge_model_quality import (
        compare_models, swap_out_baseline, archive_old_model,
        get_current_baseline_info, judge_and_replace_if_better
    )
    JUDGING_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Model judging system not available: {e}")
    JUDGING_AVAILABLE = False

# Fix Windows console encoding
if sys.platform == 'win32':
    try:
        import codecs
        if hasattr(sys.stdout, 'buffer'):
            sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
            sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')
    except Exception:
        pass

@dataclass
class SystemMetrics:
    """System performance metrics"""
    cpu_percent: float
    memory_percent: float
    disk_free_gb: float
    temperature: Optional[float]
    timestamp: str
    idle_duration_minutes: float

@dataclass
class BootloaderStatus:
    """Current bootloader status"""
    is_running: bool
    last_check: str
    autopilot_running: bool
    launch_count: int
    last_launch: Optional[str]
    current_metrics: Optional[SystemMetrics]
    next_scheduled: Optional[str]
    error_count: int
    last_error: Optional[str]

class AutopilotBootloader:
    """
    Autonomous system for launching emotional quantization autopilot
    """
    
    def __init__(self, config_path: str = "bootloader_config.json"):
        self.config_path = config_path
        self.config = self._load_config()
        self.running = False
        self.autopilot_process = None
        self.launch_count = 0
        self.error_count = 0
        self.last_error = None
        self.idle_start_time = None
        
        # Setup logging
        self._setup_logging()
        
        # Register signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        self.logger.info("Autopilot Bootloader initialized")
        self.logger.info(f"Mode: {self.config['mode']}")
        self.logger.info(f"Check interval: {self.config['check_interval']}s")
        
    def _load_config(self) -> Dict[str, Any]:
        """Load bootloader configuration"""
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            
            # Validate required fields
            required_fields = ['mode', 'idle_threshold', 'check_interval']
            for field in required_fields:
                if field not in config:
                    raise ValueError(f"Missing required config field: {field}")
            
            return config
        except Exception as e:
            print(f"Error loading config: {e}")
            # Return default config
            return {
                "mode": "idle",
                "idle_threshold": 20.0,
                "check_interval": 300,
                "max_memory_percent": 70.0,
                "autopilot_config_path": "emotion_quant_autopilot/autopilot_config.json",
                "autopilot_script_path": "emotion_quant_autopilot/quant_autopilot.py"
            }
    
    def _setup_logging(self):
        """Setup logging configuration"""
        log_file = self.config.get('log_file', 'bootloader.log')
        
        # Create logs directory if it doesn't exist
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger('AutopilotBootloader')
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.running = False
        
        if self.autopilot_process and self.autopilot_process.poll() is None:
            self.logger.info("Terminating autopilot process...")
            try:
                self.autopilot_process.terminate()
                self.autopilot_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.logger.warning("Autopilot process didn't terminate, killing...")
                self.autopilot_process.kill()
        
        sys.exit(0)
    
    def get_system_metrics(self) -> SystemMetrics:
        """Get current system performance metrics"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Try to get temperature (may not be available on all systems)
            temperature = None
            try:
                temps = psutil.sensors_temperatures()
                if temps:
                    # Get first available temperature reading
                    for name, entries in temps.items():
                        if entries:
                            temperature = entries[0].current
                            break
            except (AttributeError, PermissionError):
                pass
            
            # Calculate idle duration
            idle_duration = 0.0
            if self.idle_start_time:
                idle_duration = (datetime.now() - self.idle_start_time).total_seconds() / 60
            
            return SystemMetrics(
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                disk_free_gb=disk.free / (1024**3),
                temperature=temperature,
                timestamp=datetime.now().isoformat(),
                idle_duration_minutes=idle_duration
            )
            
        except Exception as e:
            self.logger.error(f"Failed to get system metrics: {e}")
            return SystemMetrics(
                cpu_percent=100.0,  # Conservative fallback
                memory_percent=100.0,
                disk_free_gb=0.0,
                temperature=None,
                timestamp=datetime.now().isoformat(),
                idle_duration_minutes=0.0
            )
    
    def check_safety_conditions(self, metrics: SystemMetrics) -> bool:
        """Check if it's safe to launch autopilot"""
        safety = self.config.get('safety_checks', {})
        
        # Check disk space
        min_disk = safety.get('min_free_disk_gb', 20)
        if metrics.disk_free_gb < min_disk:
            self.logger.warning(f"Insufficient disk space: {metrics.disk_free_gb:.1f}GB < {min_disk}GB")
            return False
        
        # Check temperature
        max_temp = safety.get('max_cpu_temp_celsius', 80)
        if metrics.temperature and metrics.temperature > max_temp:
            self.logger.warning(f"CPU temperature too high: {metrics.temperature:.1f}°C > {max_temp}°C")
            return False
        
        # Check work hours
        if safety.get('prevent_during_work_hours', False):
            now = datetime.now()
            work_start = datetime.strptime(safety.get('work_hours_start', '09:00'), '%H:%M').time()
            work_end = datetime.strptime(safety.get('work_hours_end', '17:00'), '%H:%M').time()
            
            if work_start <= now.time() <= work_end:
                self.logger.info("Preventing launch during work hours")
                return False
        
        # Check Ollama availability
        if safety.get('check_ollama_available', True):
            if not self._check_ollama_available():
                self.logger.warning("Ollama not available")
                return False
        
        return True
    
    def _check_ollama_available(self) -> bool:
        """Check if Ollama is running and available"""
        try:
            import requests
            response = requests.get('http://localhost:11434/api/tags', timeout=5)
            return response.status_code == 200
        except Exception:
            return False
    
    def is_system_idle(self, metrics: SystemMetrics) -> bool:
        """Check if system meets idle criteria"""
        cpu_threshold = self.config['idle_threshold']
        memory_threshold = self.config['max_memory_percent']
        min_idle_duration = self.config.get('min_idle_duration_minutes', 15)
        
        # Check CPU usage
        if metrics.cpu_percent > cpu_threshold:
            if self.idle_start_time:
                self.idle_start_time = None
                self.logger.debug(f"CPU usage {metrics.cpu_percent:.1f}% > {cpu_threshold}% - no longer idle")
            return False
        
        # Check memory usage
        if metrics.memory_percent > memory_threshold:
            if self.idle_start_time:
                self.idle_start_time = None
                self.logger.debug(f"Memory usage {metrics.memory_percent:.1f}% > {memory_threshold}% - no longer idle")
            return False
        
        # Start tracking idle time
        if not self.idle_start_time:
            self.idle_start_time = datetime.now()
            self.logger.info(f"System became idle - CPU: {metrics.cpu_percent:.1f}%, Memory: {metrics.memory_percent:.1f}%")
            return False
        
        # Check if we've been idle long enough
        idle_duration = (datetime.now() - self.idle_start_time).total_seconds() / 60
        if idle_duration >= min_idle_duration:
            self.logger.info(f"System idle for {idle_duration:.1f} minutes (>= {min_idle_duration})")
            return True
        
        return False
    
    def should_launch_cron(self) -> bool:
        """Check if it's time for scheduled launch"""
        if self.config['mode'] != 'cron':
            return False
        
        cron_time = self.config.get('cron_schedule', '02:00')
        try:
            target_time = datetime.strptime(cron_time, '%H:%M').time()
            now = datetime.now()
            current_time = now.time()
            
            # Check if we're within 5 minutes of the target time
            target_datetime = datetime.combine(now.date(), target_time)
            time_diff = abs((now - target_datetime).total_seconds())
            
            return time_diff <= 300  # 5 minutes window
            
        except ValueError:
            self.logger.error(f"Invalid cron schedule format: {cron_time}")
            return False
    
    def is_autopilot_running(self) -> bool:
        """Check if autopilot is already running"""
        if self.autopilot_process and self.autopilot_process.poll() is None:
            return True
        
        # Check for other autopilot processes
        try:
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                cmdline = proc.info.get('cmdline', [])
                if cmdline and 'quant_autopilot.py' in ' '.join(cmdline):
                    self.logger.info(f"Found existing autopilot process: PID {proc.info['pid']}")
                    return True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
        
        return False
    
    def launch_autopilot(self) -> bool:
        """Launch the quantization autopilot with tracking"""
        try:
            autopilot_script = self.config.get('autopilot_script_path', 'emotion_quant_autopilot/quant_autopilot.py')
            autopilot_config = self.config.get('autopilot_config_path', 'emotion_quant_autopilot/autopilot_config.json')
            
            if not Path(autopilot_script).exists():
                self.logger.error(f"Autopilot script not found: {autopilot_script}")
                return False
            
            if not Path(autopilot_config).exists():
                self.logger.error(f"Autopilot config not found: {autopilot_config}")
                return False
            
            # Generate unique loop ID for tracking
            loop_start_time = datetime.now()
            loop_id = f"loop_{loop_start_time.strftime('%Y%m%d_%H%M%S')}_{self.launch_count + 1}"
            
            # Launch autopilot as background process
            cmd = [
                sys.executable, autopilot_script,
                '--config', autopilot_config,
                'start'
            ]
            
            self.logger.info(f"Launching autopilot: {' '.join(cmd)} (Loop ID: {loop_id})")
            
            # Start system monitoring for this loop
            start_metrics = self.get_system_metrics()
            
            self.autopilot_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.DEVNULL
            )
            
            self.launch_count += 1
            self.logger.info(f"Autopilot launched successfully (PID: {self.autopilot_process.pid})")
            self.logger.info(f"Total launches: {self.launch_count}")
            
            # Start background thread to monitor and track this quantization loop
            if TRACKING_AVAILABLE:
                tracking_thread = threading.Thread(
                    target=self._track_quantization_loop,
                    args=(loop_id, loop_start_time, start_metrics),
                    daemon=True
                )
                tracking_thread.start()
                self.logger.info(f"Started tracking thread for loop {loop_id}")
            
            return True
            
        except Exception as e:
            self.error_count += 1
            self.last_error = str(e)
            self.logger.error(f"Failed to launch autopilot: {e}")
            return False
    
    def _track_quantization_loop(self, loop_id: str, start_time: datetime, start_metrics: SystemMetrics):
        """Track a quantization loop and record results"""
        if not TRACKING_AVAILABLE:
            return
        
        try:
            self.logger.info(f"Tracking quantization loop {loop_id}")
            
            # Wait for the autopilot process to complete
            error_count = 0
            peak_memory = start_metrics.memory_percent
            cpu_samples = [start_metrics.cpu_percent]
            
            while self.autopilot_process and self.autopilot_process.poll() is None:
                time.sleep(30)  # Check every 30 seconds
                
                try:
                    current_metrics = self.get_system_metrics()
                    peak_memory = max(peak_memory, current_metrics.memory_percent)
                    cpu_samples.append(current_metrics.cpu_percent)
                except Exception as e:
                    error_count += 1
                    self.logger.warning(f"Error getting metrics during tracking: {e}")
            
            # Process completed, calculate duration
            end_time = datetime.now()
            duration_seconds = (end_time - start_time).total_seconds()
            
            # Get process exit code
            exit_code = self.autopilot_process.returncode if self.autopilot_process else -1
            
            self.logger.info(f"Loop {loop_id} completed in {duration_seconds:.1f}s with exit code {exit_code}")
            
            # Look for generated model files to evaluate
            model_path = self._find_latest_model_file()
            model_name = f"autopilot-{loop_id}"
            
            # Evaluate model quality if model file found
            emotional_score = 0.5  # Default fallback
            token_quality = 0.5    # Default fallback
            size_mb = 0.0
            
            if model_path and Path(model_path).exists():
                self.logger.info(f"Evaluating model: {model_path}")
                emotional_score = eval_emotion(model_path)
                token_quality = eval_fluency(model_path)
                size_mb = Path(model_path).stat().st_size / (1024 * 1024)  # Convert to MB
                model_name = Path(model_path).stem
            else:
                self.logger.warning(f"No model file found for loop {loop_id}")
                error_count += 1
            
            # Determine if loop passed quality thresholds
            tracker = get_tracker()
            passed_threshold = tracker.should_accept_loop(emotional_score, token_quality)
            
            # Create and save tracking result
            result = QuantLoopResult(
                loop_id=loop_id,
                model_name=model_name,
                quant_format="q4_K_M",  # Default format, could be configurable
                size_mb=size_mb,
                emotional_score=emotional_score,
                token_quality=token_quality,
                passed_threshold=passed_threshold,
                timestamp=start_time,
                duration_seconds=duration_seconds,
                error_count=error_count,
                memory_peak_mb=peak_memory,
                cpu_avg_percent=sum(cpu_samples) / len(cpu_samples) if cpu_samples else 0,
                sentiment_variance=abs(emotional_score - 0.7) if emotional_score > 0 else 0,
                coherence_score=token_quality * 0.9,  # Approximation based on token quality
                creativity_index=min(0.95, emotional_score * 1.1)  # Creativity correlates with emotion
            )
            
            # Save the result
            success = save_loop_result(result)
            
            if success:
                self.logger.info(
                    f"Saved tracking result for {loop_id}: "
                    f"emotional={emotional_score:.3f}, token={token_quality:.3f}, "
                    f"passed={passed_threshold}, duration={duration_seconds:.1f}s"
                )
                
                # Perform model quality judging and conditional replacement
                if JUDGING_AVAILABLE and model_path and passed_threshold:
                    self.logger.info(f"🧪 Starting model quality evaluation for {model_path}")
                    
                    try:
                        # Get current baseline info
                        baseline_info = get_current_baseline_info()
                        baseline_path = baseline_info.get("baseline_path", "")
                        
                        if baseline_path and Path(baseline_path).exists():
                            # Compare the new model against baseline
                            self.logger.info(f"🔍 Comparing against baseline: {baseline_path}")
                            comparison_result = compare_models(model_path, baseline_path)
                            
                            if comparison_result["replacement_recommended"]:
                                self.logger.info(
                                    f"🎯 Model quality improvement detected! "
                                    f"Emotional gain: {comparison_result['emotionality_gain']:+.3f}, "
                                    f"Fluency gain: {comparison_result['fluency_gain']:+.3f}, "
                                    f"Confidence: {comparison_result['confidence_score']:.1%}"
                                )
                                
                                # Perform the replacement
                                try:
                                    swap_out_baseline(model_path)
                                    self.logger.info("🎉 New core model accepted and baseline updated!")
                                    
                                    # Log successful replacement
                                    self._log_model_replacement(loop_id, model_path, baseline_path, comparison_result)
                                    
                                except Exception as e:
                                    self.logger.error(f"❌ Failed to replace baseline model: {e}")
                                    
                            else:
                                self.logger.info(
                                    f"📋 Model quality assessment: replacement not recommended "
                                    f"(emotional: {comparison_result['emotionality_gain']:+.3f}, "
                                    f"fluency: {comparison_result['fluency_gain']:+.3f}, "
                                    f"confidence: {comparison_result['confidence_score']:.1%})"
                                )
                        else:
                            # No baseline exists, set this as the first baseline
                            if emotional_score > 0.7 and token_quality > 0.65:  # Basic quality threshold
                                self.logger.info(f"🆕 No baseline model found, setting new baseline: {model_path}")
                                swap_out_baseline(model_path)
                                self.logger.info("✅ Initial baseline model established!")
                            else:
                                self.logger.info(f"📋 Model quality too low to establish as baseline")
                                
                    except Exception as e:
                        self.logger.error(f"❌ Error during model quality evaluation: {e}")
                else:
                    if not JUDGING_AVAILABLE:
                        self.logger.debug("Model judging system not available")
                    elif not model_path:
                        self.logger.debug("No model path available for judging")
                    elif not passed_threshold:
                        self.logger.debug("Model did not pass quality threshold, skipping judging")
                        
            else:
                self.logger.error(f"Failed to save tracking result for {loop_id}")
                
        except Exception as e:
            self.logger.error(f"Error tracking quantization loop {loop_id}: {e}")
    
    def _find_latest_model_file(self) -> Optional[str]:
        """Find the most recently created model file"""
        try:
            # Common model file patterns and locations
            search_patterns = [
                "*.gguf",
                "*.bin", 
                "*.safetensors",
                "models/*.gguf",
                "output/*.gguf",
                "emotion_quant_autopilot/*.gguf"
            ]
            
            latest_file = None
            latest_time = 0
            
            for pattern in search_patterns:
                for model_file in Path('.').glob(pattern):
                    if model_file.is_file():
                        mtime = model_file.stat().st_mtime
                        if mtime > latest_time:
                            latest_time = mtime
                            latest_file = str(model_file)
            
            return latest_file
            
        except Exception as e:
            self.logger.warning(f"Error finding model files: {e}")
            return None
    
    def _log_model_replacement(self, loop_id: str, new_model_path: str, old_baseline_path: str, comparison_result: Dict[str, Any]):
        """Log model replacement event with detailed metrics"""
        try:
            replacement_log = {
                "event": "model_replacement",
                "timestamp": datetime.now().isoformat(),
                "loop_id": loop_id,
                "new_model": {
                    "path": new_model_path,
                    "name": Path(new_model_path).name,
                    "size_mb": Path(new_model_path).stat().st_size / (1024 * 1024) if Path(new_model_path).exists() else 0
                },
                "old_baseline": {
                    "path": old_baseline_path,
                    "name": Path(old_baseline_path).name if old_baseline_path else "none"
                },
                "quality_gains": {
                    "emotionality": comparison_result.get("emotionality_gain", 0),
                    "fluency": comparison_result.get("fluency_gain", 0),
                    "size_reduction": comparison_result.get("size_reduction", 0),
                    "speed_improvement": comparison_result.get("speed_improvement", 0),
                    "confidence": comparison_result.get("confidence_score", 0)
                }
            }
            
            # Save to replacement log file
            log_file = Path("data/model_replacements.json")
            log_file.parent.mkdir(exist_ok=True)
            
            # Load existing log or create new
            if log_file.exists():
                with open(log_file, 'r') as f:
                    replacements = json.load(f)
            else:
                replacements = {"replacements": []}
            
            replacements["replacements"].append(replacement_log)
            
            with open(log_file, 'w') as f:
                json.dump(replacements, f, indent=2)
            
            self.logger.info(f"📝 Model replacement logged to {log_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to log model replacement: {e}")

    def update_status(self, metrics: SystemMetrics):
        """Update bootloader status file"""
        try:
            status_file = self.config.get('monitoring', {}).get('status_file', 'bootloader_status.json')
            
            status = BootloaderStatus(
                is_running=self.running,
                last_check=datetime.now().isoformat(),
                autopilot_running=self.is_autopilot_running(),
                launch_count=self.launch_count,
                last_launch=None,  # TODO: Track last launch time
                current_metrics=metrics,
                next_scheduled=None,  # TODO: Calculate next scheduled time
                error_count=self.error_count,
                last_error=self.last_error
            )
            
            with open(status_file, 'w') as f:
                json.dump(asdict(status), f, indent=2, default=str)
                
        except Exception as e:
            self.logger.error(f"Failed to update status file: {e}")
    
    def run_check_cycle(self):
        """Run a single check cycle"""
        try:
            # Get current system metrics
            metrics = self.get_system_metrics()
            
            self.logger.debug(f"System metrics - CPU: {metrics.cpu_percent:.1f}%, "
                            f"Memory: {metrics.memory_percent:.1f}%, "
                            f"Disk: {metrics.disk_free_gb:.1f}GB")
            
            # Update status
            self.update_status(metrics)
            
            # Check if autopilot is already running
            if self.is_autopilot_running():
                self.logger.debug("Autopilot already running, skipping launch")
                return
            
            # Check safety conditions
            if not self.check_safety_conditions(metrics):
                self.logger.debug("Safety conditions not met, skipping launch")
                return
            
            # Determine launch condition based on mode
            should_launch = False
            reason = ""
            
            if self.config['mode'] == 'idle':
                if self.is_system_idle(metrics):
                    should_launch = True
                    reason = f"System idle (CPU: {metrics.cpu_percent:.1f}%, Memory: {metrics.memory_percent:.1f}%)"
            
            elif self.config['mode'] == 'cron':
                if self.should_launch_cron():
                    should_launch = True
                    reason = f"Scheduled time reached: {self.config['cron_schedule']}"
            
            elif self.config['mode'] == 'manual':
                self.logger.debug("Manual mode - waiting for external trigger")
                return
            
            # Launch if conditions are met
            if should_launch:
                self.logger.info(f"Launch condition met: {reason}")
                if self.launch_autopilot():
                    # Reset idle tracking after successful launch
                    self.idle_start_time = None
                else:
                    self.logger.error("Failed to launch autopilot")
            
        except Exception as e:
            self.error_count += 1
            self.last_error = str(e)
            self.logger.error(f"Error in check cycle: {e}")
    
    def run(self):
        """Main bootloader loop"""
        self.running = True
        self.logger.info("Autopilot Bootloader started")
        
        try:
            while self.running:
                self.run_check_cycle()
                
                # Sleep until next check
                time.sleep(self.config['check_interval'])
                
        except KeyboardInterrupt:
            self.logger.info("Received keyboard interrupt")
        except Exception as e:
            self.logger.error(f"Fatal error in main loop: {e}")
        finally:
            self.logger.info("Autopilot Bootloader stopped")
    
    def status_report(self) -> Dict[str, Any]:
        """Get current status report"""
        metrics = self.get_system_metrics()
        
        return {
            'bootloader': {
                'running': self.running,
                'mode': self.config['mode'],
                'check_interval': self.config['check_interval'],
                'launch_count': self.launch_count,
                'error_count': self.error_count,
                'last_error': self.last_error
            },
            'system': asdict(metrics),
            'autopilot': {
                'running': self.is_autopilot_running(),
                'process_id': self.autopilot_process.pid if self.autopilot_process else None
            },
            'conditions': {
                'is_idle': self.is_system_idle(metrics),
                'safety_ok': self.check_safety_conditions(metrics),
                'cron_ready': self.should_launch_cron() if self.config['mode'] == 'cron' else False
            }
        }

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Autopilot Bootloader - Autonomous Launch System')
    parser.add_argument('--config', default='bootloader_config.json', 
                       help='Path to bootloader configuration file')
    parser.add_argument('--status', action='store_true',
                       help='Show current status and exit')
    parser.add_argument('--launch-now', action='store_true',
                       help='Launch autopilot immediately (bypass conditions)')
    parser.add_argument('--stop', action='store_true',
                       help='Stop any running autopilot processes')
    
    args = parser.parse_args()
    
    try:
        bootloader = AutopilotBootloader(args.config)
        
        if args.status:
            # Show status and exit
            status = bootloader.status_report()
            print(json.dumps(status, indent=2, default=str))
            return
        
        if args.launch_now:
            # Launch immediately
            print("Launching autopilot immediately...")
            if bootloader.launch_autopilot():
                print("Autopilot launched successfully")
            else:
                print("Failed to launch autopilot")
            return
        
        if args.stop:
            # Stop running autopilot processes
            print("Stopping autopilot processes...")
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    cmdline = proc.info.get('cmdline', [])
                    if cmdline and 'quant_autopilot.py' in ' '.join(cmdline):
                        print(f"Terminating autopilot process: PID {proc.info['pid']}")
                        proc.terminate()
                        proc.wait(timeout=10)
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.TimeoutExpired):
                    pass
            return
        
        # Run main bootloader loop
        bootloader.run()
        
    except Exception as e:
        print(f"Fatal error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())

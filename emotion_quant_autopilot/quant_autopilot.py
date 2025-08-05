#!/usr/bin/env python3
"""
Emotion Quantization Autopilot - Master Control Loop
Autonomous emotional quantization system with idle-triggered execution
"""

import os
import sys
import json

# Fix Windows console encoding for Unicode characters
if sys.platform == 'win32':
    try:
        import codecs
        if hasattr(sys.stdout, 'buffer'):
            sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
            sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')
    except Exception:
        # Fallback: just continue without UTF-8 encoding
        pass
import sqlite3
import logging
import subprocess
import threading
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Import our components
from idle_monitor import IdleMonitor, create_idle_monitor_from_config
from emotional_dataset_builder import EmotionalDatasetBuilder
from emotion_training_tracker import EmotionTrainingTracker, EmotionalMetrics, QuantLevel, PassType

@dataclass
class AutopilotRun:
    """Represents a single autopilot execution run"""
    run_id: str
    trigger_type: str  # "idle", "manual", "scheduled"
    timestamp: datetime
    model_path: str
    base_model: str
    quantization_method: str
    target_size_gb: float
    result_summary: str
    judgment_score: float
    success: bool
    error_message: str = ""
    execution_time_minutes: float = 0.0

@dataclass
class QuantizationJob:
    """Represents a quantization job in the queue"""
    job_id: str
    base_model: str
    quantization_method: str
    priority: int = 5
    created_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()

class AutopilotDatabase:
    """Database manager for autopilot operations"""
    
    def __init__(self, db_path: str = "emotion_training.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self) -> None:
        """Initialize database schema with autopilot tables"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create autopilot_runs table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS autopilot_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT UNIQUE NOT NULL,
                    trigger_type TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    model_path TEXT NOT NULL,
                    base_model TEXT NOT NULL,
                    quantization_method TEXT NOT NULL,
                    target_size_gb REAL NOT NULL,
                    result_summary TEXT NOT NULL,
                    judgment_score REAL NOT NULL,
                    success BOOLEAN NOT NULL,
                    error_message TEXT DEFAULT '',
                    execution_time_minutes REAL DEFAULT 0.0,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create quantization_queue table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS quantization_queue (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT UNIQUE NOT NULL,
                    base_model TEXT NOT NULL,
                    quantization_method TEXT NOT NULL,
                    priority INTEGER DEFAULT 5,
                    status TEXT DEFAULT 'pending',
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    started_at TEXT NULL,
                    completed_at TEXT NULL,
                    run_id TEXT NULL,
                    FOREIGN KEY (run_id) REFERENCES autopilot_runs (run_id)
                )
            """)
            
            conn.commit()
    
    def add_autopilot_run(self, run: AutopilotRun) -> int:
        """Add autopilot run record"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO autopilot_runs (
                    run_id, trigger_type, timestamp, model_path, base_model,
                    quantization_method, target_size_gb, result_summary, 
                    judgment_score, success, error_message, execution_time_minutes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                run.run_id, run.trigger_type, run.timestamp.isoformat(),
                run.model_path, run.base_model, run.quantization_method,
                run.target_size_gb, run.result_summary, run.judgment_score,
                run.success, run.error_message, run.execution_time_minutes
            ))
            return cursor.lastrowid or 0
    
    def add_quantization_job(self, job: QuantizationJob) -> int:
        """Add job to quantization queue"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            created_at = job.created_at or datetime.now()
            cursor.execute("""
                INSERT INTO quantization_queue (
                    job_id, base_model, quantization_method, priority, created_at
                ) VALUES (?, ?, ?, ?, ?)
            """, (
                job.job_id, job.base_model, job.quantization_method,
                job.priority, created_at.isoformat()
            ))
            return cursor.lastrowid or 0
    
    def get_pending_jobs(self, limit: int = 10) -> List[QuantizationJob]:
        """Get pending jobs from queue, ordered by priority and creation time"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT job_id, base_model, quantization_method, priority, created_at
                FROM quantization_queue
                WHERE status = 'pending'
                ORDER BY priority DESC, created_at ASC
                LIMIT ?
            """, (limit,))
            
            jobs = []
            for row in cursor.fetchall():
                jobs.append(QuantizationJob(
                    job_id=row[0],
                    base_model=row[1],
                    quantization_method=row[2],
                    priority=row[3],
                    created_at=datetime.fromisoformat(row[4])
                ))
            return jobs
    
    def update_job_status(self, job_id: str, status: str, run_id: Optional[str] = None) -> None:
        """Update job status in queue"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            timestamp_field = None
            if status == 'running':
                timestamp_field = 'started_at'
            elif status in ['completed', 'failed']:
                timestamp_field = 'completed_at'
            
            if timestamp_field:
                cursor.execute(f"""
                    UPDATE quantization_queue 
                    SET status = ?, {timestamp_field} = ?, run_id = ?
                    WHERE job_id = ?
                """, (status, datetime.now().isoformat(), run_id, job_id))
            else:
                cursor.execute("""
                    UPDATE quantization_queue 
                    SET status = ?, run_id = ?
                    WHERE job_id = ?
                """, (status, run_id, job_id))
            
            conn.commit()
    
    def get_daily_run_count(self, date: Optional[datetime] = None) -> int:
        """Get number of autopilot runs for a given date"""
        if date is None:
            date = datetime.now()
        
        start_of_day = date.replace(hour=0, minute=0, second=0, microsecond=0)
        end_of_day = start_of_day + timedelta(days=1)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT COUNT(*) FROM autopilot_runs
                WHERE timestamp >= ? AND timestamp < ?
            """, (start_of_day.isoformat(), end_of_day.isoformat()))
            
            return cursor.fetchone()[0]

class QuantizationAutopilot:
    """
    Main autopilot controller for emotional quantization
    
    Features:
    - Idle-triggered autonomous execution
    - Queue-based job management
    - Integration with existing quantization pipeline
    - Comprehensive logging and tracking
    - Safety limits and emergency stops
    """
    
    def __init__(self, config_path: str = "autopilot_config.json"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
        # Initialize components
        self.db = AutopilotDatabase()
        self.idle_monitor = create_idle_monitor_from_config(self.config)
        self.dataset_builder = EmotionalDatasetBuilder()
        self.training_tracker = EmotionTrainingTracker()
        
        # Setup logging
        self._setup_logging()
        
        # State management
        self.is_running = False
        self.current_job: Optional[QuantizationJob] = None
        self.current_run: Optional[AutopilotRun] = None
        
        # Setup directories
        self._setup_directories()
        
        # Setup idle monitoring callbacks
        self.idle_monitor.set_idle_callback(self._on_system_idle)
        self.idle_monitor.set_active_callback(self._on_system_active)
        
        # Attempt crash recovery
        self.recover_from_crash()
        
        # Check integration status
        integration_status = self.get_integration_status()
        self.logger.info(f"🔗 Integration status: {integration_status['integration_status']}")
        
        if integration_status["integration_status"] == "fully_integrated":
            self.logger.info("✅ All components integrated successfully")
        elif integration_status["integration_status"] == "partially_integrated":
            self.logger.warning("⚠️ Some components not available - limited functionality")
        else:
            self.logger.warning("❌ Integration not available - using fallback mode")
        
        self.logger.info("🤖 Quantization Autopilot initialized")
        self.logger.info(f"   Config: {self.config_path}")
        self.logger.info(f"   Auto-start: {self.config.get('auto_start', False)}")
        self.logger.info(f"   Max daily loops: {self.config.get('max_active_loops_per_day', 3)}")
    
    def _load_config(self) -> Dict:
        """Load configuration from JSON file"""
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            return config
        except FileNotFoundError:
            self.logger.error(f"❌ Config file not found: {self.config_path}")
            raise
        except json.JSONDecodeError as e:
            self.logger.error(f"❌ Invalid JSON in config file: {e}")
            raise
    
    def _setup_logging(self) -> None:
        """Setup logging configuration"""
        log_dir = Path(self.config["output_paths"]["logs_directory"])
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = log_dir / f"autopilot_{datetime.now().strftime('%Y%m%d')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger("QuantizationAutopilot")
    
    def _setup_directories(self) -> None:
        """Create necessary directories"""
        paths = self.config["output_paths"]
        
        for path_key, path_value in paths.items():
            path = Path(path_value)
            path.mkdir(parents=True, exist_ok=True)
            self.logger.debug(f"📁 Ensured directory: {path}")
    
    def _check_safety_limits(self) -> bool:
        """Check if it's safe to start a new quantization"""
        # Check daily limits
        daily_runs = self.db.get_daily_run_count()
        max_daily = self.config.get("max_active_loops_per_day", 3)
        
        if daily_runs >= max_daily:
            self.logger.info(f"🛑 Daily limit reached: {daily_runs}/{max_daily} runs")
            return False
        
        # Check emergency stop file
        emergency_stop = Path(self.config["safety_limits"]["emergency_stop_file"])
        if emergency_stop.exists():
            self.logger.warning(f"🚨 Emergency stop file detected: {emergency_stop}")
            return False
        
        # Check disk space
        max_disk_usage = self.config["safety_limits"]["max_disk_usage_gb"]
        models_dir = Path(self.config["output_paths"]["models_directory"])
        
        if models_dir.exists():
            total_size = sum(f.stat().st_size for f in models_dir.rglob('*') if f.is_file())
            total_size_gb = total_size / (1024**3)
            
            if total_size_gb > max_disk_usage:
                self.logger.warning(f"💽 Disk usage limit exceeded: {total_size_gb:.1f}GB > {max_disk_usage}GB")
                return False
        
        # Check concurrent processes
        max_concurrent = self.config["safety_limits"]["max_concurrent_processes"]
        if self.current_job is not None:
            self.logger.info("⏳ Another quantization job is already running")
            return False
        
        return True
    
    def _on_system_idle(self) -> None:
        """Called when system becomes idle"""
        self.logger.info("💤 System idle detected - checking for queued jobs")
        
        if not self._check_safety_limits():
            return
        
        # Get next job from queue
        pending_jobs = self.db.get_pending_jobs(limit=1)
        
        if not pending_jobs:
            self.logger.info("📭 No pending jobs in queue")
            return
        
        job = pending_jobs[0]
        self.logger.info(f"🎯 Starting quantization job: {job.base_model} -> {job.quantization_method}")
        
        # Start quantization in background thread
        thread = threading.Thread(target=self._execute_quantization_job, args=(job,), daemon=True)
        thread.start()
    
    def _on_system_active(self) -> None:
        """Called when system becomes active"""
        if self.current_job:
            self.logger.info("🏃 System active - quantization will continue but may be slower")
        else:
            self.logger.debug("🏃 System active - no quantization running")
    
    def _execute_quantization_job(self, job: QuantizationJob) -> None:
        """Execute a single quantization job"""
        start_time = datetime.now()
        run_id = str(uuid.uuid4())
        
        # Create autopilot run record
        self.current_run = AutopilotRun(
            run_id=run_id,
            trigger_type="idle",
            timestamp=start_time,
            model_path="",
            base_model=job.base_model,
            quantization_method=job.quantization_method,
            target_size_gb=self.config["target_model_size_range_gb"][1],
            result_summary="",
            judgment_score=0.0,
            success=False
        )
        
        self.current_job = job
        
        try:
            # Update job status
            self.db.update_job_status(job.job_id, "running", run_id)
            
            self.logger.info(f"🚀 Starting quantization: {job.base_model} -> {job.quantization_method}")
            
            # Execute quantization using existing pass1_quantization_loop
            result = self._run_quantization_process(job)
            
            if result["success"]:
                self.current_run.success = True
                self.current_run.model_path = result["model_path"]
                self.current_run.judgment_score = result["judgment_score"]
                self.current_run.result_summary = result["summary"]
                
                self.db.update_job_status(job.job_id, "completed", run_id)
                self.logger.info(f"✅ Quantization completed: score {result['judgment_score']:.3f}")
                
                # Send success notification
                self._send_notification(f"✅ Quantization successful: {job.base_model} -> {job.quantization_method}")
                
            else:
                self.current_run.error_message = result["error"]
                self.current_run.result_summary = f"Failed: {result['error']}"
                
                self.db.update_job_status(job.job_id, "failed", run_id)
                self.logger.error(f"❌ Quantization failed: {result['error']}")
                
                # Send failure notification
                self._send_notification(f"❌ Quantization failed: {job.base_model} -> {result['error']}")
        
        except Exception as e:
            self.current_run.error_message = str(e)
            self.current_run.result_summary = f"Exception: {e}"
            
            self.db.update_job_status(job.job_id, "failed", run_id)
            self.logger.error(f"💥 Quantization exception: {e}")
            
            self._send_notification(f"💥 Quantization error: {job.base_model} -> {e}")
        
        finally:
            # Calculate execution time
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds() / 60
            self.current_run.execution_time_minutes = execution_time
            
            # Save run record
            self.db.add_autopilot_run(self.current_run)
            
            # Clear current job
            self.current_job = None
            self.current_run = None
            
            self.logger.info(f"🏁 Job completed in {execution_time:.1f} minutes")
    
    def _run_quantization_process(self, job: QuantizationJob) -> Dict[str, Any]:
        """Run the actual quantization process with full integration"""
        import uuid
        from datetime import datetime
        
        run_id = str(uuid.uuid4())
        start_time = datetime.now()
        
        try:
            # Import our integrated modules
            import sys
            sys.path.append('..')  # Add parent directory to path
            
            from quantize_model import quantize_model
            from judge_emotion import judge_emotion
            from emotion_core_tracker import EmotionalQuantDatabase
            from autopilot_state import AutopilotStateManager
            
            # Initialize enhanced database and state manager
            db = EmotionalQuantDatabase("../emotion_training.db")
            state_manager = AutopilotStateManager("../autopilot_state.json", self.config)
            
            # Update state - starting quantization
            state_manager.start_autopilot(job.job_id, run_id)
            
            self.logger.info(f"🚀 Starting integrated quantization process")
            self.logger.info(f"   Run ID: {run_id}")
            self.logger.info(f"   Job: {job.base_model} -> {job.quantization_method}")
            
            # Step 1: Quantize the model
            self.logger.info("📦 Step 1: Quantizing model...")
            
            quantizer_config = {
                "output_dir": self.config["output_paths"]["models_directory"],
                "backend": self.config.get("quantization_backend", "mock"),  # Use mock for testing
                "temp_dir": self.config["output_paths"]["temp_directory"]
            }
            
            # Determine target size range from config
            size_range = None
            if "target_model_size_range_gb" in self.config:
                size_range = tuple(self.config["target_model_size_range_gb"])
            
            quant_result = quantize_model(
                base_model=job.base_model,
                quantization_method=job.quantization_method,
                config=quantizer_config,
                target_size_range_gb=size_range
            )
            
            if not quant_result.success:
                # Log failure and update state
                error_msg = f"Quantization failed: {quant_result.error_message}"
                self.logger.error(f"❌ {error_msg}")
                
                execution_time = (datetime.now() - start_time).total_seconds() / 60
                db.log_autopilot_run(
                    run_id=run_id,
                    model_path="",
                    base_model=job.base_model,
                    quantization_method=job.quantization_method,
                    judgment_score=0.0,
                    emotional_deviation=1.0,
                    execution_time_minutes=execution_time,
                    success=False,
                    result_summary=error_msg,
                    error_message=quant_result.error_message
                )
                
                state_manager.complete_job(success=False)
                
                return {
                    "success": False,
                    "error": error_msg,
                    "model_path": "",
                    "judgment_score": 0.0,
                    "summary": ""
                }
            
            self.logger.info(f"✅ Quantization successful: {quant_result.model_path}")
            self.logger.info(f"   Size: {quant_result.model_size_mb:.1f}MB")
            self.logger.info(f"   Duration: {quant_result.duration_seconds:.1f}s")
            
            # Step 2: Evaluate emotional intelligence
            self.logger.info("🧠 Step 2: Evaluating emotional intelligence...")
            
            judge_config = {
                "silent_mode": True,  # Unattended operation
                "evaluation_count": self.config["evaluation_settings"]["evaluation_prompt_count"],
                "baseline_model_path": self.config.get("baseline_model_path"),
                "response_timeout": self.config["evaluation_settings"]["response_timeout_seconds"]
            }
            
            judgment_result = judge_emotion(
                model_path=quant_result.model_path,
                base_model=job.base_model,
                quantization_method=job.quantization_method,
                config=judge_config
            )
            
            if not judgment_result.success:
                error_msg = f"Evaluation failed: {judgment_result.error_message}"
                self.logger.error(f"❌ {error_msg}")
                
                execution_time = (datetime.now() - start_time).total_seconds() / 60
                db.log_autopilot_run(
                    run_id=run_id,
                    model_path=quant_result.model_path,
                    base_model=job.base_model,
                    quantization_method=job.quantization_method,
                    judgment_score=0.0,
                    emotional_deviation=1.0,
                    execution_time_minutes=execution_time,
                    success=False,
                    result_summary=error_msg,
                    error_message=judgment_result.error_message
                )
                
                state_manager.complete_job(success=False)
                
                return {
                    "success": False,
                    "error": error_msg,
                    "model_path": quant_result.model_path,
                    "judgment_score": 0.0,
                    "summary": ""
                }
            
            self.logger.info(f"✅ Evaluation successful:")
            self.logger.info(f"   Overall Score: {judgment_result.judgment_score:.3f}")
            self.logger.info(f"   Fluency: {judgment_result.fluency_score:.3f}")
            self.logger.info(f"   Emotional Intensity: {judgment_result.emotional_intensity_score:.3f}")
            self.logger.info(f"   Emotional Match: {judgment_result.emotional_match_score:.3f}")
            self.logger.info(f"   Empathy: {judgment_result.empathy_score:.3f}")
            
            # Step 3: Calculate emotional deviation
            baseline_score = self.config.get("baseline_emotional_score", 0.85)
            emotional_deviation = abs(baseline_score - judgment_result.judgment_score)
            
            # Step 4: Log comprehensive results to database
            self.logger.info("💾 Step 3: Logging results to database...")
            
            execution_time = (datetime.now() - start_time).total_seconds() / 60
            result_summary = (
                f"Size: {quant_result.model_size_mb:.1f}MB, "
                f"Score: {judgment_result.judgment_score:.3f}, "
                f"Deviation: {emotional_deviation:.3f}, "
                f"Duration: {execution_time:.1f}min"
            )
            
            # Log main autopilot run
            autopilot_run_id = db.log_autopilot_run(
                run_id=run_id,
                model_path=quant_result.model_path,
                base_model=job.base_model,
                quantization_method=job.quantization_method,
                judgment_score=judgment_result.judgment_score,
                emotional_deviation=emotional_deviation,
                execution_time_minutes=execution_time,
                success=True,
                result_summary=result_summary
            )
            
            # Log detailed evaluation results
            eval_id = db.log_evaluation_result(run_id, judgment_result)
            
            # Track model lineage
            lineage_id = db.track_model_lineage(
                model_path=quant_result.model_path,
                parent_model=job.base_model,
                quantization_method=job.quantization_method,
                generation=1,  # First generation quantization
                size_mb=quant_result.model_size_mb
            )
            
            # Step 5: Evaluate if this should be a seed candidate
            deviation_threshold = self.config.get("emotion_degradation_threshold", 0.07)
            size_threshold_gb = self.config.get("target_model_size_range_gb", [20.0, 24.0])[1]
            size_gb = quant_result.model_size_mb / 1024
            
            is_seed_candidate = (
                emotional_deviation <= deviation_threshold and
                size_gb <= size_threshold_gb and
                judgment_result.judgment_score >= 0.75
            )
            
            if is_seed_candidate:
                # Calculate performance rank (higher score = lower rank number)
                performance_rank = int((1.0 - judgment_result.judgment_score) * 100)
                db.mark_as_seed_candidate(quant_result.model_path, performance_rank)
                
                self.logger.info(f"🌱 Marked as seed candidate (rank {performance_rank})")
                
                # Check if this is the best candidate so far
                performance_summary = db.get_performance_summary()
                seed_candidates = performance_summary.get("seed_candidates", [])
                
                if not seed_candidates or performance_rank <= min(c["rank"] for c in seed_candidates):
                    db.select_as_seed(quant_result.model_path)
                    self.logger.info(f"🎯 Selected as new seed model!")
            
            # Step 6: Update state and complete job
            state_manager.complete_job(success=True)
            
            # Step 7: Save state for crash recovery
            db.set_autopilot_state("last_successful_run", run_id)
            db.set_autopilot_state("last_successful_model", quant_result.model_path)
            
            self.logger.info(f"✅ Integrated quantization process complete")
            self.logger.info(f"   Run ID: {run_id}")
            self.logger.info(f"   Emotional deviation: {emotional_deviation:.3f}")
            self.logger.info(f"   Seed candidate: {is_seed_candidate}")
            
            return {
                "success": True,
                "model_path": quant_result.model_path,
                "judgment_score": judgment_result.judgment_score,
                "emotional_deviation": emotional_deviation,
                "summary": result_summary,
                "run_id": run_id,
                "seed_candidate": is_seed_candidate,
                "execution_time_minutes": execution_time,
                "quantization_result": quant_result,
                "judgment_result": judgment_result
            }
            
        except Exception as e:
            # Handle any unexpected errors
            execution_time = (datetime.now() - start_time).total_seconds() / 60
            error_msg = f"Unexpected error in quantization process: {str(e)}"
            
            self.logger.error(f"❌ {error_msg}")
            
            try:
                # Try to log the error
                from emotion_core_tracker import EmotionalQuantDatabase
                from autopilot_state import AutopilotStateManager
                
                db = EmotionalQuantDatabase("../emotion_training.db")
                state_manager = AutopilotStateManager("../autopilot_state.json", self.config)
                
                db.log_autopilot_run(
                    run_id=run_id,
                    model_path="",
                    base_model=job.base_model,
                    quantization_method=job.quantization_method,
                    judgment_score=0.0,
                    emotional_deviation=1.0,
                    execution_time_minutes=execution_time,
                    success=False,
                    result_summary=error_msg,
                    error_message=str(e)
                )
                
                state_manager.complete_job(success=False)
                
            except Exception as log_error:
                self.logger.error(f"❌ Failed to log error: {log_error}")
            
            return {
                "success": False,
                "error": error_msg,
                "model_path": "",
                "judgment_score": 0.0,
                "summary": "",
                "run_id": run_id
            }
    
    def _send_notification(self, message: str) -> None:
        """Send notification via configured channels"""
        self.logger.info(f"📢 Notification: {message}")
        
        # Log-only mode (always enabled)
        notification_file = Path(self.config["output_paths"]["logs_directory"]) / "notifications.log"
        with open(notification_file, 'a') as f:
            f.write(f"{datetime.now().isoformat()} - {message}\n")
        
        # TODO: Add email and Slack notifications based on config
        # if self.config["notifications"]["email"]["enabled"]:
        #     self._send_email_notification(message)
        #
        # if self.config["notifications"]["slack"]["enabled"]:
        #     self._send_slack_notification(message)
    
    def recover_from_crash(self) -> bool:
        """Attempt to recover from a crash or unclean shutdown"""
        try:
            import sys
            sys.path.append('..')
            from autopilot_state import AutopilotStateManager
            from emotion_core_tracker import EmotionalQuantDatabase
            
            # Initialize state manager and database
            state_manager = AutopilotStateManager("../autopilot_state.json", self.config)
            db = EmotionalQuantDatabase("../emotion_training.db")
            
            # Get recovery information
            recovery_info = state_manager.get_recovery_info()
            
            if not recovery_info["was_running"]:
                self.logger.info("✅ Clean shutdown detected - no recovery needed")
                return True
            
            self.logger.warning("⚠️ Unclean shutdown detected - attempting recovery")
            self.logger.info(f"   Last job: {recovery_info['current_job_id']}")
            self.logger.info(f"   Last run: {recovery_info['current_run_id']}")
            self.logger.info(f"   Last activity: {recovery_info['last_activity']}")
            
            # Check if we were in the middle of a job
            if recovery_info["current_job_id"] and recovery_info["current_run_id"]:
                # Mark the interrupted job as failed
                self.db.update_job_status(recovery_info["current_job_id"], "failed")
                
                # Log the recovery event
                db.set_autopilot_state("crash_recovery", 
                    f"Recovered from crash at {datetime.now().isoformat()}")
                
                self.logger.info("🔧 Marked interrupted job as failed")
            
            # Reset state
            state_manager.stop_autopilot()
            
            # Check for any jobs that were marked as running but not completed
            pending_jobs = self.db.get_pending_jobs()
            for job in pending_jobs:
                # Reset any jobs that might be stuck in "running" state
                self.db.update_job_status(job.job_id, "pending")
            
            self.logger.info("✅ Crash recovery completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Crash recovery failed: {e}")
            return False
    
    def get_integration_status(self) -> Dict[str, Any]:
        """Get status of all integrated components"""
        try:
            import sys
            sys.path.append('..')
            from autopilot_state import AutopilotStateManager
            from emotion_core_tracker import EmotionalQuantDatabase
            
            # Test component availability
            status = {
                "quantizer_available": False,
                "judge_available": False,
                "database_available": False,
                "state_manager_available": False,
                "integration_status": "disconnected"
            }
            
            # Test quantizer
            try:
                from quantize_model import ModelQuantizer
                status["quantizer_available"] = True
            except ImportError as e:
                self.logger.warning(f"⚠️ Quantizer not available: {e}")
            
            # Test judge
            try:
                from judge_emotion import EmotionalJudge
                status["judge_available"] = True
            except ImportError as e:
                self.logger.warning(f"⚠️ Judge not available: {e}")
            
            # Test database
            try:
                db = EmotionalQuantDatabase("../emotion_training.db")
                status["database_available"] = True
            except Exception as e:
                self.logger.warning(f"⚠️ Database not available: {e}")
            
            # Test state manager
            try:
                state_manager = AutopilotStateManager("../autopilot_state.json", self.config)
                status["state_manager_available"] = True
            except Exception as e:
                self.logger.warning(f"⚠️ State manager not available: {e}")
            
            # Determine overall integration status
            if all([status["quantizer_available"], status["judge_available"], 
                   status["database_available"], status["state_manager_available"]]):
                status["integration_status"] = "fully_integrated"
            elif any([status["quantizer_available"], status["judge_available"]]):
                status["integration_status"] = "partially_integrated"
            else:
                status["integration_status"] = "disconnected"
            
            return status
            
        except Exception as e:
            self.logger.error(f"❌ Failed to get integration status: {e}")
            return {"integration_status": "error", "error": str(e)}

    def add_quantization_job(self, base_model: str, quantization_method: str, priority: int = 5) -> str:
        """Add a new quantization job to the queue"""
        job_id = str(uuid.uuid4())
        
        job = QuantizationJob(
            job_id=job_id,
            base_model=base_model,
            quantization_method=quantization_method,
            priority=priority
        )
        
        self.db.add_quantization_job(job)
        self.logger.info(f"➕ Added job to queue: {base_model} -> {quantization_method} (priority: {priority})")
        
        return job_id
    
    def populate_default_queue(self) -> None:
        """Populate queue with default quantization jobs"""
        base_models = self.config["preferred_base_models"]
        quant_methods = self.config["quantization_methods"]
        
        job_count = 0
        for base_model in base_models:
            for quant_method in quant_methods:
                self.add_quantization_job(base_model, quant_method)
                job_count += 1
        
        self.logger.info(f"📋 Populated queue with {job_count} default jobs")
    
    def start(self) -> None:
        """Start the autopilot system"""
        if self.is_running:
            self.logger.warning("⚠️ Autopilot already running")
            return
        
        self.is_running = True
        self.logger.info("🚀 Starting Quantization Autopilot")
        
        # Start idle monitoring
        self.idle_monitor.start_monitoring()
        
        # Populate default queue if empty
        pending_jobs = self.db.get_pending_jobs(limit=1)
        if not pending_jobs:
            self.logger.info("📋 Queue empty - populating with default jobs")
            self.populate_default_queue()
        
        self.logger.info("✅ Autopilot started - monitoring for idle conditions")
    
    def stop(self) -> None:
        """Stop the autopilot system"""
        if not self.is_running:
            return
        
        self.logger.info("🛑 Stopping Quantization Autopilot")
        self.is_running = False
        
        # Stop idle monitoring
        self.idle_monitor.stop_monitoring()
        
        self.logger.info("✅ Autopilot stopped")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current autopilot status"""
        idle_status = self.idle_monitor.get_idle_status()
        pending_jobs = self.db.get_pending_jobs()
        daily_runs = self.db.get_daily_run_count()
        
        return {
            "is_running": self.is_running,
            "idle_status": idle_status,
            "current_job": asdict(self.current_job) if self.current_job else None,
            "pending_jobs_count": len(pending_jobs),
            "daily_runs": daily_runs,
            "max_daily_runs": self.config.get("max_active_loops_per_day", 3),
            "safety_checks": {
                "emergency_stop_exists": Path(self.config["safety_limits"]["emergency_stop_file"]).exists(),
                "within_daily_limits": daily_runs < self.config.get("max_active_loops_per_day", 3)
            }
        }

# CLI interface
def create_cli():
    """Create command-line interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Emotion Quantization Autopilot')
    
    parser.add_argument('--config', default='autopilot_config.json', 
                       help='Configuration file path')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Start command
    start_parser = subparsers.add_parser('start', help='Start autopilot system')
    
    # Stop command
    stop_parser = subparsers.add_parser('stop', help='Stop autopilot system')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Show autopilot status')
    
    # Add job command
    add_parser = subparsers.add_parser('add-job', help='Add quantization job')
    add_parser.add_argument('--model', required=True, help='Base model name')
    add_parser.add_argument('--quant', required=True, help='Quantization method')
    add_parser.add_argument('--priority', type=int, default=5, help='Job priority (1-10)')
    
    # Queue command
    queue_parser = subparsers.add_parser('queue', help='Show job queue')
    
    # Populate command
    populate_parser = subparsers.add_parser('populate', help='Populate default queue')
    
    # Recovery command
    recover_parser = subparsers.add_parser('recover', help='Recover from crash')
    
    # Integration status command
    integration_parser = subparsers.add_parser('integration', help='Show integration status')
    
    return parser

def main():
    """Main execution function"""
    parser = create_cli()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    try:
        # Create autopilot instance
        autopilot = QuantizationAutopilot(args.config)
        
        if args.command == 'start':
            autopilot.start()
            
            try:
                # Keep running until interrupted
                while autopilot.is_running:
                    time.sleep(10)
            except KeyboardInterrupt:
                autopilot.stop()
        
        elif args.command == 'stop':
            autopilot.stop()
            print("✅ Autopilot stopped")
        
        elif args.command == 'status':
            status = autopilot.get_status()
            print(json.dumps(status, indent=2, default=str))
        
        elif args.command == 'add-job':
            job_id = autopilot.add_quantization_job(args.model, args.quant, args.priority)
            print(f"✅ Added job {job_id}: {args.model} -> {args.quant}")
        
        elif args.command == 'queue':
            pending_jobs = autopilot.db.get_pending_jobs()
            print(f"📋 Pending Jobs ({len(pending_jobs)}):")
            for job in pending_jobs:
                print(f"   {job.base_model} -> {job.quantization_method} (priority: {job.priority})")
        
        elif args.command == 'populate':
            autopilot.populate_default_queue()
            print("Queue populated with default jobs")
        
        elif args.command == 'recover':
            print("Attempting crash recovery...")
            success = autopilot.recover_from_crash()
            if success:
                print("Recovery completed successfully")
            else:
                print("Recovery failed")
                return 1
        
        elif args.command == 'integration':
            status = autopilot.get_integration_status()
            print("Integration Status:")
            print(f"   Overall: {status['integration_status']}")
            print(f"   Quantizer: {'[OK]' if status['quantizer_available'] else '[FAIL]'}")
            print(f"   Judge: {'[OK]' if status['judge_available'] else '[FAIL]'}")
            print(f"   Database: {'[OK]' if status['database_available'] else '[FAIL]'}")
            print(f"   State Manager: {'[OK]' if status['state_manager_available'] else '[FAIL]'}")
            
            if status.get('error'):
                print(f"   Error: {status['error']}")
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())

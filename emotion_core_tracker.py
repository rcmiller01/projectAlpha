#!/usr/bin/env python3
"""
Enhanced Database Schema for Emotional Quantization Autopilot
Extends the existing emotion_training.db with additional fields for autopilot integration
"""

import sqlite3
import logging
from pathlib import Path
from typing import Optional
from datetime import datetime

class EmotionalQuantDatabase:
    """Enhanced database for emotional quantization tracking with autopilot integration"""
    
    def __init__(self, db_path: str = "emotion_training.db"):
        self.db_path = Path(db_path)
        self.logger = logging.getLogger("EmotionalQuantDatabase")
        
        # Ensure database exists and is properly initialized
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize or upgrade database schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Create core tables first if they don't exist
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
                    error_message TEXT,
                    execution_time_minutes REAL,
                    created_at TEXT
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS quantization_queue (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT UNIQUE NOT NULL,
                    base_model TEXT NOT NULL,
                    quantization_method TEXT NOT NULL,
                    priority INTEGER DEFAULT 5,
                    status TEXT DEFAULT 'pending',
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    completed_at TEXT,
                    run_id TEXT
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS training_iterations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT NOT NULL,
                    quant_level TEXT NOT NULL,
                    pass_type TEXT NOT NULL,
                    pass_count INTEGER NOT NULL,
                    iteration INTEGER NOT NULL,
                    timestamp TEXT NOT NULL,
                    model_size_mb REAL NOT NULL,
                    response_fluency REAL NOT NULL,
                    emotional_intensity REAL NOT NULL,
                    emotional_match REAL NOT NULL,
                    empathy_score REAL NOT NULL,
                    metaphor_usage REAL NOT NULL,
                    sentiment_accuracy REAL NOT NULL,
                    overall_score REAL NOT NULL,
                    notes TEXT,
                    config_hash TEXT
                )
            """)
            
            # Check if we need to add autopilot integration fields
            self._add_autopilot_integration_fields(cursor)
            
            # Create autopilot state table for resumption
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS autopilot_state (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    state_key TEXT UNIQUE NOT NULL,
                    state_value TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            
            # Create enhanced evaluation results table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS evaluation_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    autopilot_run_id TEXT NOT NULL,
                    model_path TEXT NOT NULL,
                    base_model TEXT NOT NULL,
                    quantization_method TEXT NOT NULL,
                    judgment_score REAL NOT NULL,
                    fluency_score REAL NOT NULL,
                    emotional_intensity_score REAL NOT NULL,
                    emotional_match_score REAL NOT NULL,
                    empathy_score REAL NOT NULL,
                    baseline_preference REAL NOT NULL,
                    evaluation_count INTEGER NOT NULL,
                    reflection_notes TEXT,
                    evaluation_time_seconds REAL NOT NULL,
                    success BOOLEAN NOT NULL,
                    error_message TEXT,
                    created_at TEXT NOT NULL
                )
            """)
            
            # Create model lineage tracking table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS model_lineage (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_path TEXT UNIQUE NOT NULL,
                    parent_model TEXT,
                    quantization_method TEXT NOT NULL,
                    generation INTEGER NOT NULL,
                    size_mb REAL NOT NULL,
                    creation_timestamp TEXT NOT NULL,
                    is_seed_candidate BOOLEAN DEFAULT FALSE,
                    selected_as_seed BOOLEAN DEFAULT FALSE,
                    seed_selection_date TEXT,
                    performance_rank INTEGER,
                    notes TEXT
                )
            """)
            
            # Create performance comparison table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS performance_comparisons (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    comparison_id TEXT NOT NULL,
                    model_a_path TEXT NOT NULL,
                    model_b_path TEXT NOT NULL,
                    preferred_model TEXT NOT NULL,
                    confidence_score REAL NOT NULL,
                    comparison_type TEXT NOT NULL,
                    comparison_date TEXT NOT NULL,
                    evaluator_notes TEXT
                )
            """)
            
            conn.commit()
            self.logger.info("🗄️ Database schema initialized/upgraded successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Database initialization failed: {e}")
            conn.rollback()
            raise
        finally:
            conn.close()
    
    def _add_autopilot_integration_fields(self, cursor):
        """Add autopilot integration fields to existing tables if needed"""
        try:
            # Add fields to autopilot_runs table if missing
            cursor.execute("PRAGMA table_info(autopilot_runs)")
            columns = [col[1] for col in cursor.fetchall()]
            
            if 'emotional_deviation' not in columns:
                cursor.execute("ALTER TABLE autopilot_runs ADD COLUMN emotional_deviation REAL")
            
            if 'selected_as_seed' not in columns:
                cursor.execute("ALTER TABLE autopilot_runs ADD COLUMN selected_as_seed BOOLEAN DEFAULT FALSE")
            
            if 'seed_selection_date' not in columns:
                cursor.execute("ALTER TABLE autopilot_runs ADD COLUMN seed_selection_date TEXT")
            
            if 'performance_rank' not in columns:
                cursor.execute("ALTER TABLE autopilot_runs ADD COLUMN performance_rank INTEGER")
            
            # Add fields to training_iterations table if missing
            cursor.execute("PRAGMA table_info(training_iterations)")
            columns = [col[1] for col in cursor.fetchall()]
            
            if 'autopilot_run_id' not in columns:
                cursor.execute("ALTER TABLE training_iterations ADD COLUMN autopilot_run_id TEXT")
            
            if 'baseline_comparison_score' not in columns:
                cursor.execute("ALTER TABLE training_iterations ADD COLUMN baseline_comparison_score REAL")
            
            self.logger.info("📊 Autopilot integration fields added to existing tables")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Could not add autopilot fields (may already exist): {e}")
    
    def log_autopilot_run(self, 
                         run_id: str,
                         model_path: str,
                         base_model: str,
                         quantization_method: str,
                         judgment_score: float,
                         emotional_deviation: float,
                         execution_time_minutes: float,
                         success: bool,
                         result_summary: str = "",
                         error_message: str = "") -> int:
        """Log an autopilot run with enhanced tracking"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT INTO autopilot_runs (
                    run_id, trigger_type, timestamp, model_path, base_model,
                    quantization_method, target_size_gb, result_summary,
                    judgment_score, success, error_message, execution_time_minutes,
                    emotional_deviation, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                run_id, "idle_triggered", datetime.now().isoformat(),
                model_path, base_model, quantization_method, 24.0,
                result_summary, judgment_score, success, error_message,
                execution_time_minutes, emotional_deviation, datetime.now().isoformat()
            ))
            
            run_db_id = cursor.lastrowid
            conn.commit()
            
            self.logger.info(f"📝 Logged autopilot run: {run_id}")
            return run_db_id
            
        except Exception as e:
            self.logger.error(f"❌ Failed to log autopilot run: {e}")
            conn.rollback()
            raise
        finally:
            conn.close()
    
    def log_evaluation_result(self, 
                             autopilot_run_id: str,
                             judgment_result) -> int:
        """Log detailed evaluation results"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT INTO evaluation_results (
                    autopilot_run_id, model_path, base_model, quantization_method,
                    judgment_score, fluency_score, emotional_intensity_score,
                    emotional_match_score, empathy_score, baseline_preference,
                    evaluation_count, reflection_notes, evaluation_time_seconds,
                    success, error_message, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                autopilot_run_id, judgment_result.model_path, judgment_result.base_model,
                judgment_result.quantization_method, judgment_result.judgment_score,
                judgment_result.fluency_score, judgment_result.emotional_intensity_score,
                judgment_result.emotional_match_score, judgment_result.empathy_score,
                judgment_result.baseline_preference, judgment_result.evaluation_count,
                judgment_result.reflection_notes, judgment_result.evaluation_time_seconds,
                judgment_result.success, judgment_result.error_message,
                datetime.now().isoformat()
            ))
            
            eval_id = cursor.lastrowid
            conn.commit()
            
            self.logger.info(f"📊 Logged evaluation result for run: {autopilot_run_id}")
            return eval_id
            
        except Exception as e:
            self.logger.error(f"❌ Failed to log evaluation result: {e}")
            conn.rollback()
            raise
        finally:
            conn.close()
    
    def track_model_lineage(self,
                           model_path: str,
                           parent_model: Optional[str],
                           quantization_method: str,
                           generation: int,
                           size_mb: float) -> int:
        """Track model lineage and relationships"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT OR REPLACE INTO model_lineage (
                    model_path, parent_model, quantization_method, generation,
                    size_mb, creation_timestamp
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (
                model_path, parent_model, quantization_method, generation,
                size_mb, datetime.now().isoformat()
            ))
            
            lineage_id = cursor.lastrowid
            conn.commit()
            
            self.logger.info(f"🧬 Tracked model lineage: {model_path}")
            return lineage_id
            
        except Exception as e:
            self.logger.error(f"❌ Failed to track model lineage: {e}")
            conn.rollback()
            raise
        finally:
            conn.close()
    
    def mark_as_seed_candidate(self, model_path: str, performance_rank: int) -> bool:
        """Mark a model as a seed candidate for next generation"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                UPDATE model_lineage 
                SET is_seed_candidate = TRUE, performance_rank = ?
                WHERE model_path = ?
            """, (performance_rank, model_path))
            
            cursor.execute("""
                UPDATE autopilot_runs 
                SET performance_rank = ?
                WHERE model_path = ?
            """, (performance_rank, model_path))
            
            conn.commit()
            
            self.logger.info(f"🌱 Marked as seed candidate: {model_path} (rank {performance_rank})")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to mark seed candidate: {e}")
            conn.rollback()
            return False
        finally:
            conn.close()
    
    def select_as_seed(self, model_path: str) -> bool:
        """Select a model as the seed for next generation"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # First, unselect any previous seeds
            cursor.execute("UPDATE model_lineage SET selected_as_seed = FALSE")
            cursor.execute("UPDATE autopilot_runs SET selected_as_seed = FALSE")
            
            # Select the new seed
            current_time = datetime.now().isoformat()
            
            cursor.execute("""
                UPDATE model_lineage 
                SET selected_as_seed = TRUE, seed_selection_date = ?
                WHERE model_path = ?
            """, (current_time, model_path))
            
            cursor.execute("""
                UPDATE autopilot_runs 
                SET selected_as_seed = TRUE, seed_selection_date = ?
                WHERE model_path = ?
            """, (current_time, model_path))
            
            conn.commit()
            
            self.logger.info(f"🎯 Selected as seed: {model_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to select seed: {e}")
            conn.rollback()
            return False
        finally:
            conn.close()
    
    def get_autopilot_state(self, key: str) -> Optional[str]:
        """Get autopilot state for resumption"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("SELECT state_value FROM autopilot_state WHERE state_key = ?", (key,))
            result = cursor.fetchone()
            return result[0] if result else None
            
        except Exception as e:
            self.logger.error(f"❌ Failed to get autopilot state: {e}")
            return None
        finally:
            conn.close()
    
    def set_autopilot_state(self, key: str, value: str) -> bool:
        """Set autopilot state for resumption"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT OR REPLACE INTO autopilot_state (state_key, state_value, updated_at)
                VALUES (?, ?, ?)
            """, (key, value, datetime.now().isoformat()))
            
            conn.commit()
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to set autopilot state: {e}")
            conn.rollback()
            return False
        finally:
            conn.close()
    
    def get_performance_summary(self) -> dict:
        """Get performance summary for decision making"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Get recent performance metrics
            cursor.execute("""
                SELECT quantization_method, AVG(judgment_score), COUNT(*), AVG(emotional_deviation)
                FROM autopilot_runs 
                WHERE success = TRUE AND timestamp > datetime('now', '-30 days')
                GROUP BY quantization_method
                ORDER BY AVG(judgment_score) DESC
            """)
            
            performance_by_method = []
            for row in cursor.fetchall():
                performance_by_method.append({
                    "method": row[0],
                    "avg_score": row[1],
                    "run_count": row[2],
                    "avg_deviation": row[3]
                })
            
            # Get seed candidates
            cursor.execute("""
                SELECT model_path, performance_rank, judgment_score
                FROM autopilot_runs
                WHERE is_seed_candidate = TRUE OR selected_as_seed = TRUE
                ORDER BY performance_rank
                LIMIT 5
            """)
            
            seed_candidates = []
            for row in cursor.fetchall():
                seed_candidates.append({
                    "model_path": row[0],
                    "rank": row[1],
                    "score": row[2]
                })
            
            return {
                "performance_by_method": performance_by_method,
                "seed_candidates": seed_candidates,
                "last_updated": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"❌ Failed to get performance summary: {e}")
            return {}
        finally:
            conn.close()

if __name__ == "__main__":
    # Test the enhanced database
    db = EmotionalQuantDatabase("test_enhanced.db")
    
    # Test logging
    run_id = "test_run_123"
    db_id = db.log_autopilot_run(
        run_id=run_id,
        model_path="/path/to/model",
        base_model="llama2-13b",
        quantization_method="q4_K_M",
        judgment_score=0.85,
        emotional_deviation=0.05,
        execution_time_minutes=45.0,
        success=True,
        result_summary="Successful quantization"
    )
    
    print(f"✅ Test run logged with ID: {db_id}")
    
    # Test state management
    db.set_autopilot_state("last_run", run_id)
    retrieved_state = db.get_autopilot_state("last_run")
    print(f"✅ State management test: {retrieved_state}")
    
    # Test performance summary
    summary = db.get_performance_summary()
    print(f"✅ Performance summary: {summary}")
    
    print("🎉 Enhanced database testing complete!")

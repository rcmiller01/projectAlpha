#!/usr/bin/env python3
"""
Emotion Training Tracker
Monitors and visualizes emotional model training progress across quantization passes
"""

import json
import sqlite3
import csv
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass, asdict
from enum import Enum

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PassType(Enum):
    """Training pass types"""
    PASS_1 = "pass_1"  # Autonomous quantization
    PASS_2 = "pass_2"  # Model comparison and selection

class QuantLevel(Enum):
    """Quantization levels"""
    ORIGINAL = "original"
    FOUR_BIT = "4bit"
    EIGHT_BIT = "8bit"
    GPTQ = "gptq"
    CUSTOM = "custom"

@dataclass
class EmotionalMetrics:
    """Container for emotional evaluation metrics"""
    response_fluency: float  # 0-1 scale, how coherent/natural responses are
    emotional_intensity: float  # 0-1 scale, strength of emotional expression
    emotional_match: float  # 0-1 scale, how well emotion matches expected
    empathy_score: float  # 0-1 scale, demonstration of empathy
    metaphor_usage: float  # 0-1 scale, appropriate use of metaphors
    sentiment_accuracy: float  # 0-1 scale, accuracy of sentiment detection
    
    def overall_score(self) -> float:
        """Calculate weighted overall emotional score"""
        weights = {
            'response_fluency': 0.20,
            'emotional_intensity': 0.15,
            'emotional_match': 0.25,
            'empathy_score': 0.20,
            'metaphor_usage': 0.10,
            'sentiment_accuracy': 0.10
        }
        
        return (
            self.response_fluency * weights['response_fluency'] +
            self.emotional_intensity * weights['emotional_intensity'] +
            self.emotional_match * weights['emotional_match'] +
            self.empathy_score * weights['empathy_score'] +
            self.metaphor_usage * weights['metaphor_usage'] +
            self.sentiment_accuracy * weights['sentiment_accuracy']
        )

@dataclass
class TrainingIteration:
    """Single training iteration record"""
    id: int
    model_name: str
    quant_level: str
    pass_type: str
    pass_count: int
    iteration: int
    timestamp: str
    model_size_mb: float
    emotional_metrics: EmotionalMetrics
    notes: str = ""
    config_hash: str = ""

class EmotionTrainingTracker:
    """Main tracker for emotional model training progress"""
    
    def __init__(self, db_path: str = "emotion_training.db", backup_json: str = "emotion_training_backup.json"):
        self.db_path = Path(db_path)
        self.backup_json = Path(backup_json)
        
        # Initialize database
        self._init_database()
        
        # Load or create backup
        self._load_backup()
        
        logger.info(f"📊 Emotion Training Tracker initialized")
        logger.info(f"   Database: {self.db_path}")
        logger.info(f"   Backup: {self.backup_json}")
    
    def _init_database(self):
        """Initialize SQLite database schema"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
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
                    notes TEXT DEFAULT '',
                    config_hash TEXT DEFAULT '',
                    UNIQUE(model_name, quant_level, pass_type, pass_count, iteration)
                )
            ''')
            
            # Create indexes for better query performance
            conn.execute('CREATE INDEX IF NOT EXISTS idx_model_quant ON training_iterations(model_name, quant_level)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_pass_type ON training_iterations(pass_type, pass_count)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON training_iterations(timestamp)')
            
            conn.commit()
        
        logger.info("✅ Database schema initialized")
    
    def _load_backup(self):
        """Load data from JSON backup if exists"""
        if self.backup_json.exists():
            try:
                with open(self.backup_json, 'r') as f:
                    backup_data = json.load(f)
                
                # Check if we need to restore from backup
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.execute('SELECT COUNT(*) FROM training_iterations')
                    db_count = cursor.fetchone()[0]
                
                backup_count = len(backup_data.get('iterations', []))
                
                if backup_count > db_count:
                    logger.info(f"🔄 Restoring {backup_count - db_count} records from backup")
                    self._restore_from_backup(backup_data)
                
            except Exception as e:
                logger.warning(f"⚠️ Could not load backup: {e}")
    
    def _restore_from_backup(self, backup_data: Dict):
        """Restore data from backup JSON"""
        with sqlite3.connect(self.db_path) as conn:
            for iteration_data in backup_data.get('iterations', []):
                try:
                    # Convert to TrainingIteration object
                    metrics = EmotionalMetrics(**iteration_data['emotional_metrics'])
                    iteration = TrainingIteration(
                        id=iteration_data.get('id', 0),
                        model_name=iteration_data['model_name'],
                        quant_level=iteration_data['quant_level'],
                        pass_type=iteration_data['pass_type'],
                        pass_count=iteration_data['pass_count'],
                        iteration=iteration_data['iteration'],
                        timestamp=iteration_data['timestamp'],
                        model_size_mb=iteration_data['model_size_mb'],
                        emotional_metrics=metrics,
                        notes=iteration_data.get('notes', ''),
                        config_hash=iteration_data.get('config_hash', '')
                    )
                    
                    # Insert if not exists
                    self._insert_iteration(iteration, conn)
                    
                except Exception as e:
                    logger.warning(f"⚠️ Could not restore iteration: {e}")
    
    def _insert_iteration(self, iteration: TrainingIteration, conn: sqlite3.Connection = None):
        """Insert iteration into database"""
        close_conn = conn is None
        if conn is None:
            conn = sqlite3.connect(self.db_path)
        
        try:
            conn.execute('''
                INSERT OR REPLACE INTO training_iterations 
                (model_name, quant_level, pass_type, pass_count, iteration, timestamp, 
                 model_size_mb, response_fluency, emotional_intensity, emotional_match,
                 empathy_score, metaphor_usage, sentiment_accuracy, overall_score, notes, config_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                iteration.model_name, iteration.quant_level, iteration.pass_type,
                iteration.pass_count, iteration.iteration, iteration.timestamp,
                iteration.model_size_mb, iteration.emotional_metrics.response_fluency,
                iteration.emotional_metrics.emotional_intensity, iteration.emotional_metrics.emotional_match,
                iteration.emotional_metrics.empathy_score, iteration.emotional_metrics.metaphor_usage,
                iteration.emotional_metrics.sentiment_accuracy, iteration.emotional_metrics.overall_score(),
                iteration.notes, iteration.config_hash
            ))
            
            if close_conn:
                conn.commit()
                
        finally:
            if close_conn and conn:
                conn.close()
    
    def add_iteration(self, model_name: str, quant_level: QuantLevel, pass_type: PassType,
                     pass_count: int, model_size_mb: float, emotional_metrics: EmotionalMetrics,
                     notes: str = "", config_hash: str = "") -> int:
        """Add a new training iteration"""
        
        # Get next iteration number for this model/pass combination
        iteration_num = self.get_next_iteration(model_name, quant_level.value, pass_type.value, pass_count)
        
        iteration = TrainingIteration(
            id=0,  # Will be auto-assigned
            model_name=model_name,
            quant_level=quant_level.value,
            pass_type=pass_type.value,
            pass_count=pass_count,
            iteration=iteration_num,
            timestamp=datetime.now().isoformat(),
            model_size_mb=model_size_mb,
            emotional_metrics=emotional_metrics,
            notes=notes,
            config_hash=config_hash
        )
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO training_iterations 
                (model_name, quant_level, pass_type, pass_count, iteration, timestamp, 
                 model_size_mb, response_fluency, emotional_intensity, emotional_match,
                 empathy_score, metaphor_usage, sentiment_accuracy, overall_score, notes, config_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                iteration.model_name, iteration.quant_level, iteration.pass_type,
                iteration.pass_count, iteration.iteration, iteration.timestamp,
                iteration.model_size_mb, iteration.emotional_metrics.response_fluency,
                iteration.emotional_metrics.emotional_intensity, iteration.emotional_metrics.emotional_match,
                iteration.emotional_metrics.empathy_score, iteration.emotional_metrics.metaphor_usage,
                iteration.emotional_metrics.sentiment_accuracy, iteration.emotional_metrics.overall_score(),
                iteration.notes, iteration.config_hash
            ))
            iteration_id = cursor.lastrowid or 0
            conn.commit()
        
        # Update backup
        self._update_backup()
        
        logger.info(f"➕ Added iteration {iteration_num} for {model_name} ({quant_level.value}, {pass_type.value})")
        logger.info(f"   Overall Score: {emotional_metrics.overall_score():.3f}")
        logger.info(f"   Model Size: {model_size_mb:.1f}MB")
        
        return iteration_id
    
    def get_next_iteration(self, model_name: str, quant_level: str, pass_type: str, pass_count: int) -> int:
        """Get the next iteration number for a model/pass combination"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                SELECT MAX(iteration) FROM training_iterations 
                WHERE model_name = ? AND quant_level = ? AND pass_type = ? AND pass_count = ?
            ''', (model_name, quant_level, pass_type, pass_count))
            
            max_iteration = cursor.fetchone()[0]
            return (max_iteration + 1) if max_iteration is not None else 1
    
    def increment_iteration(self, model_name: str, quant_level: QuantLevel, pass_type: PassType,
                           pass_count: int, model_size_mb: float, emotional_metrics: EmotionalMetrics,
                           notes: str = "") -> int:
        """Convenience method to increment iteration for a model"""
        return self.add_iteration(model_name, quant_level, pass_type, pass_count, 
                                model_size_mb, emotional_metrics, notes)
    
    def reset_pass(self, model_name: str, quant_level: QuantLevel, pass_type: PassType, pass_count: int) -> bool:
        """Reset (delete) all iterations for a specific pass"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                DELETE FROM training_iterations 
                WHERE model_name = ? AND quant_level = ? AND pass_type = ? AND pass_count = ?
            ''', (model_name, quant_level.value, pass_type.value, pass_count))
            
            deleted_count = cursor.rowcount
            conn.commit()
        
        self._update_backup()
        
        logger.info(f"🗑️ Reset pass: deleted {deleted_count} iterations for {model_name} ({quant_level.value}, {pass_type.value}, pass {pass_count})")
        return deleted_count > 0
    
    def get_iterations(self, model_name: Optional[str] = None, quant_level: Optional[str] = None,
                      pass_type: Optional[str] = None, pass_count: Optional[int] = None,
                      limit: Optional[int] = None) -> List[TrainingIteration]:
        """Get iterations with optional filtering"""
        
        query = 'SELECT * FROM training_iterations WHERE 1=1'
        params = []
        
        if model_name:
            query += ' AND model_name = ?'
            params.append(model_name)
        
        if quant_level:
            query += ' AND quant_level = ?'
            params.append(quant_level)
        
        if pass_type:
            query += ' AND pass_type = ?'
            params.append(pass_type)
        
        if pass_count is not None:
            query += ' AND pass_count = ?'
            params.append(pass_count)
        
        query += ' ORDER BY timestamp DESC'
        
        if limit:
            query += ' LIMIT ?'
            params.append(limit)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query, params)
            rows = cursor.fetchall()
        
        # Convert to TrainingIteration objects
        iterations = []
        for row in rows:
            metrics = EmotionalMetrics(
                response_fluency=row['response_fluency'],
                emotional_intensity=row['emotional_intensity'],
                emotional_match=row['emotional_match'],
                empathy_score=row['empathy_score'],
                metaphor_usage=row['metaphor_usage'],
                sentiment_accuracy=row['sentiment_accuracy']
            )
            
            iteration = TrainingIteration(
                id=row['id'],
                model_name=row['model_name'],
                quant_level=row['quant_level'],
                pass_type=row['pass_type'],
                pass_count=row['pass_count'],
                iteration=row['iteration'],
                timestamp=row['timestamp'],
                model_size_mb=row['model_size_mb'],
                emotional_metrics=metrics,
                notes=row['notes'],
                config_hash=row['config_hash']
            )
            
            iterations.append(iteration)
        
        return iterations
    
    def compare_models(self, model_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """Compare performance across models and quantization levels"""
        
        query = '''
            SELECT model_name, quant_level, pass_type, 
                   AVG(overall_score) as avg_score,
                   MAX(overall_score) as best_score,
                   AVG(model_size_mb) as avg_size,
                   COUNT(*) as iteration_count,
                   AVG(response_fluency) as avg_fluency,
                   AVG(emotional_match) as avg_emotional_match
            FROM training_iterations
        '''
        
        params = []
        if model_names:
            placeholders = ','.join('?' * len(model_names))
            query += f' WHERE model_name IN ({placeholders})'
            params.extend(model_names)
        
        query += ' GROUP BY model_name, quant_level, pass_type ORDER BY avg_score DESC'
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query, params)
            results = cursor.fetchall()
        
        comparison = {
            'timestamp': datetime.now().isoformat(),
            'models_compared': model_names or 'all',
            'results': []
        }
        
        for row in results:
            comparison['results'].append({
                'model_name': row['model_name'],
                'quant_level': row['quant_level'],
                'pass_type': row['pass_type'],
                'avg_score': round(row['avg_score'], 3),
                'best_score': round(row['best_score'], 3),
                'avg_size_mb': round(row['avg_size'], 1),
                'iteration_count': row['iteration_count'],
                'avg_fluency': round(row['avg_fluency'], 3),
                'avg_emotional_match': round(row['avg_emotional_match'], 3),
                'efficiency_ratio': round(row['avg_score'] / (row['avg_size'] / 1000), 3)  # Score per GB
            })
        
        return comparison
    
    def get_progress_visualization_data(self, model_name: str, quant_level: str, 
                                      pass_type: str, pass_count: int) -> Dict[str, Any]:
        """Get data formatted for visualization"""
        
        iterations = self.get_iterations(model_name, quant_level, pass_type, pass_count)
        
        if not iterations:
            return {'error': 'No data found for the specified model/pass combination'}
        
        # Sort by iteration number
        iterations.sort(key=lambda x: x.iteration)
        
        viz_data = {
            'model_info': {
                'model_name': model_name,
                'quant_level': quant_level,
                'pass_type': pass_type,
                'pass_count': pass_count
            },
            'iterations': [],
            'summary': {
                'total_iterations': len(iterations),
                'best_score': 0,
                'latest_score': 0,
                'improvement_trend': 0,
                'avg_model_size': 0
            }
        }
        
        overall_scores = []
        model_sizes = []
        
        for iteration in iterations:
            iteration_data = {
                'iteration': iteration.iteration,
                'timestamp': iteration.timestamp,
                'overall_score': iteration.emotional_metrics.overall_score(),
                'response_fluency': iteration.emotional_metrics.response_fluency,
                'emotional_intensity': iteration.emotional_metrics.emotional_intensity,
                'emotional_match': iteration.emotional_metrics.emotional_match,
                'empathy_score': iteration.emotional_metrics.empathy_score,
                'model_size_mb': iteration.model_size_mb,
                'notes': iteration.notes
            }
            
            viz_data['iterations'].append(iteration_data)
            overall_scores.append(iteration.emotional_metrics.overall_score())
            model_sizes.append(iteration.model_size_mb)
        
        # Calculate summary statistics
        viz_data['summary']['best_score'] = max(overall_scores)
        viz_data['summary']['latest_score'] = overall_scores[-1]
        viz_data['summary']['avg_model_size'] = sum(model_sizes) / len(model_sizes)
        
        # Calculate improvement trend (linear regression slope approximation)
        if len(overall_scores) > 1:
            x_vals = list(range(len(overall_scores)))
            n = len(overall_scores)
            sum_x = sum(x_vals)
            sum_y = sum(overall_scores)
            sum_xy = sum(x * y for x, y in zip(x_vals, overall_scores))
            sum_x2 = sum(x * x for x in x_vals)
            
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
            viz_data['summary']['improvement_trend'] = round(slope, 4)
        
        return viz_data
    
    def export_to_csv(self, output_file: str, model_name: Optional[str] = None) -> str:
        """Export training data to CSV"""
        
        iterations = self.get_iterations(model_name=model_name)
        
        output_path = Path(output_file)
        
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = [
                'id', 'model_name', 'quant_level', 'pass_type', 'pass_count', 'iteration',
                'timestamp', 'model_size_mb', 'response_fluency', 'emotional_intensity',
                'emotional_match', 'empathy_score', 'metaphor_usage', 'sentiment_accuracy',
                'overall_score', 'notes', 'config_hash'
            ]
            
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for iteration in iterations:
                row = {
                    'id': iteration.id,
                    'model_name': iteration.model_name,
                    'quant_level': iteration.quant_level,
                    'pass_type': iteration.pass_type,
                    'pass_count': iteration.pass_count,
                    'iteration': iteration.iteration,
                    'timestamp': iteration.timestamp,
                    'model_size_mb': iteration.model_size_mb,
                    'response_fluency': iteration.emotional_metrics.response_fluency,
                    'emotional_intensity': iteration.emotional_metrics.emotional_intensity,
                    'emotional_match': iteration.emotional_metrics.emotional_match,
                    'empathy_score': iteration.emotional_metrics.empathy_score,
                    'metaphor_usage': iteration.emotional_metrics.metaphor_usage,
                    'sentiment_accuracy': iteration.emotional_metrics.sentiment_accuracy,
                    'overall_score': iteration.emotional_metrics.overall_score(),
                    'notes': iteration.notes,
                    'config_hash': iteration.config_hash
                }
                
                writer.writerow(row)
        
        logger.info(f"📊 Exported {len(iterations)} iterations to {output_path}")
        return str(output_path)
    
    def export_to_json(self, output_file: str, model_name: Optional[str] = None) -> str:
        """Export training data to JSON"""
        
        iterations = self.get_iterations(model_name=model_name)
        
        export_data = {
            'export_info': {
                'timestamp': datetime.now().isoformat(),
                'total_iterations': len(iterations),
                'model_filter': model_name
            },
            'iterations': []
        }
        
        for iteration in iterations:
            iteration_dict = asdict(iteration)
            # Convert EmotionalMetrics to dict
            iteration_dict['emotional_metrics'] = asdict(iteration.emotional_metrics)
            iteration_dict['overall_score'] = iteration.emotional_metrics.overall_score()
            export_data['iterations'].append(iteration_dict)
        
        output_path = Path(output_file)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📊 Exported {len(iterations)} iterations to {output_path}")
        return str(output_path)
    
    def _update_backup(self):
        """Update JSON backup file"""
        try:
            iterations = self.get_iterations()
            
            backup_data = {
                'backup_info': {
                    'timestamp': datetime.now().isoformat(),
                    'total_iterations': len(iterations)
                },
                'iterations': []
            }
            
            for iteration in iterations:
                iteration_dict = asdict(iteration)
                iteration_dict['emotional_metrics'] = asdict(iteration.emotional_metrics)
                backup_data['iterations'].append(iteration_dict)
            
            with open(self.backup_json, 'w', encoding='utf-8') as f:
                json.dump(backup_data, f, indent=2, ensure_ascii=False)
            
        except Exception as e:
            logger.warning(f"⚠️ Could not update backup: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about training progress"""
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            # Basic counts
            cursor = conn.execute('SELECT COUNT(*) as total FROM training_iterations')
            total_iterations = cursor.fetchone()['total']
            
            if total_iterations == 0:
                return {'total_iterations': 0, 'message': 'No training data available'}
            
            # Model statistics
            cursor = conn.execute('''
                SELECT model_name, quant_level, COUNT(*) as count,
                       AVG(overall_score) as avg_score, MAX(overall_score) as best_score
                FROM training_iterations 
                GROUP BY model_name, quant_level
                ORDER BY avg_score DESC
            ''')
            model_stats = cursor.fetchall()
            
            # Pass type statistics
            cursor = conn.execute('''
                SELECT pass_type, COUNT(*) as count, AVG(overall_score) as avg_score
                FROM training_iterations 
                GROUP BY pass_type
            ''')
            pass_stats = cursor.fetchall()
            
            # Recent activity (last 7 days)
            week_ago = (datetime.now() - timedelta(days=7)).isoformat()
            cursor = conn.execute('''
                SELECT COUNT(*) as recent_count 
                FROM training_iterations 
                WHERE timestamp > ?
            ''', (week_ago,))
            recent_activity = cursor.fetchone()['recent_count']
            
            # Best performing models
            cursor = conn.execute('''
                SELECT model_name, quant_level, MAX(overall_score) as best_score,
                       model_size_mb, timestamp
                FROM training_iterations 
                GROUP BY model_name, quant_level
                ORDER BY best_score DESC
                LIMIT 5
            ''')
            top_performers = cursor.fetchall()
        
        stats = {
            'total_iterations': total_iterations,
            'recent_activity_7days': recent_activity,
            'model_statistics': [dict(row) for row in model_stats],
            'pass_statistics': [dict(row) for row in pass_stats],
            'top_performers': [dict(row) for row in top_performers],
            'generated_at': datetime.now().isoformat()
        }
        
        return stats

def create_cli():
    """Create command-line interface"""
    
    parser = argparse.ArgumentParser(description='Emotion Training Tracker CLI')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Add iteration command
    add_parser = subparsers.add_parser('add', help='Add a new training iteration')
    add_parser.add_argument('--model', required=True, help='Model name')
    add_parser.add_argument('--quant', choices=['original', '4bit', '8bit', 'gptq', 'custom'], 
                           required=True, help='Quantization level')
    add_parser.add_argument('--pass-type', choices=['pass_1', 'pass_2'], required=True, 
                           help='Training pass type')
    add_parser.add_argument('--pass-count', type=int, default=1, help='Pass count number')
    add_parser.add_argument('--size', type=float, required=True, help='Model size in MB')
    add_parser.add_argument('--fluency', type=float, required=True, help='Response fluency (0-1)')
    add_parser.add_argument('--intensity', type=float, required=True, help='Emotional intensity (0-1)')
    add_parser.add_argument('--match', type=float, required=True, help='Emotional match (0-1)')
    add_parser.add_argument('--empathy', type=float, required=True, help='Empathy score (0-1)')
    add_parser.add_argument('--metaphor', type=float, default=0.5, help='Metaphor usage (0-1)')
    add_parser.add_argument('--sentiment', type=float, default=0.5, help='Sentiment accuracy (0-1)')
    add_parser.add_argument('--notes', default='', help='Additional notes')
    
    # Reset pass command
    reset_parser = subparsers.add_parser('reset', help='Reset a training pass')
    reset_parser.add_argument('--model', required=True, help='Model name')
    reset_parser.add_argument('--quant', choices=['original', '4bit', '8bit', 'gptq', 'custom'], 
                             required=True, help='Quantization level')
    reset_parser.add_argument('--pass-type', choices=['pass_1', 'pass_2'], required=True, 
                             help='Training pass type')
    reset_parser.add_argument('--pass-count', type=int, default=1, help='Pass count number')
    
    # Compare models command
    compare_parser = subparsers.add_parser('compare', help='Compare model performances')
    compare_parser.add_argument('--models', nargs='*', help='Model names to compare (default: all)')
    compare_parser.add_argument('--output', help='Save comparison to file')
    
    # Export command
    export_parser = subparsers.add_parser('export', help='Export training data')
    export_parser.add_argument('--format', choices=['csv', 'json'], required=True, 
                              help='Export format')
    export_parser.add_argument('--output', required=True, help='Output file path')
    export_parser.add_argument('--model', help='Filter by model name')
    
    # Statistics command
    stats_parser = subparsers.add_parser('stats', help='Show training statistics')
    
    # Visualize command
    viz_parser = subparsers.add_parser('visualize', help='Get visualization data')
    viz_parser.add_argument('--model', required=True, help='Model name')
    viz_parser.add_argument('--quant', required=True, help='Quantization level')
    viz_parser.add_argument('--pass-type', required=True, help='Pass type')
    viz_parser.add_argument('--pass-count', type=int, default=1, help='Pass count')
    viz_parser.add_argument('--output', help='Save visualization data to file')
    
    return parser

def main():
    """Main CLI function"""
    
    parser = create_cli()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Initialize tracker
    tracker = EmotionTrainingTracker()
    
    try:
        if args.command == 'add':
            metrics = EmotionalMetrics(
                response_fluency=args.fluency,
                emotional_intensity=args.intensity,
                emotional_match=args.match,
                empathy_score=args.empathy,
                metaphor_usage=args.metaphor,
                sentiment_accuracy=args.sentiment
            )
            
            iteration_id = tracker.add_iteration(
                model_name=args.model,
                quant_level=QuantLevel(args.quant),
                pass_type=PassType(args.pass_type),
                pass_count=args.pass_count,
                model_size_mb=args.size,
                emotional_metrics=metrics,
                notes=args.notes
            )
            
            print(f"✅ Added iteration with ID: {iteration_id}")
            print(f"   Overall Score: {metrics.overall_score():.3f}")
        
        elif args.command == 'reset':
            success = tracker.reset_pass(
                model_name=args.model,
                quant_level=QuantLevel(args.quant),
                pass_type=PassType(args.pass_type),
                pass_count=args.pass_count
            )
            
            if success:
                print("✅ Pass reset successfully")
            else:
                print("❌ No iterations found to reset")
        
        elif args.command == 'compare':
            comparison = tracker.compare_models(args.models)
            
            print("\n📊 Model Comparison Results")
            print("=" * 80)
            
            for result in comparison['results']:
                print(f"\n🤖 {result['model_name']} ({result['quant_level']}, {result['pass_type']})")
                print(f"   Average Score: {result['avg_score']:.3f}")
                print(f"   Best Score: {result['best_score']:.3f}")
                print(f"   Average Size: {result['avg_size_mb']:.1f}MB")
                print(f"   Iterations: {result['iteration_count']}")
                print(f"   Efficiency: {result['efficiency_ratio']:.3f} score/GB")
            
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(comparison, f, indent=2)
                print(f"\n💾 Comparison saved to {args.output}")
        
        elif args.command == 'export':
            if args.format == 'csv':
                output_path = tracker.export_to_csv(args.output, args.model)
            else:
                output_path = tracker.export_to_json(args.output, args.model)
            
            print(f"✅ Data exported to {output_path}")
        
        elif args.command == 'stats':
            stats = tracker.get_statistics()
            
            print("\n📈 Training Statistics")
            print("=" * 50)
            print(f"Total Iterations: {stats['total_iterations']}")
            print(f"Recent Activity (7 days): {stats['recent_activity_7days']}")
            
            print(f"\n🏆 Top Performers:")
            for performer in stats['top_performers']:
                print(f"   {performer['model_name']} ({performer['quant_level']}): {performer['best_score']:.3f}")
            
            print(f"\n📊 Pass Type Distribution:")
            for pass_stat in stats['pass_statistics']:
                print(f"   {pass_stat['pass_type']}: {pass_stat['count']} iterations, avg {pass_stat['avg_score']:.3f}")
        
        elif args.command == 'visualize':
            viz_data = tracker.get_progress_visualization_data(
                args.model, args.quant, args.pass_type, args.pass_count
            )
            
            if 'error' in viz_data:
                print(f"❌ {viz_data['error']}")
                return 1
            
            print(f"\n📈 Progress Visualization Data for {args.model}")
            print("=" * 60)
            print(f"Total Iterations: {viz_data['summary']['total_iterations']}")
            print(f"Best Score: {viz_data['summary']['best_score']:.3f}")
            print(f"Latest Score: {viz_data['summary']['latest_score']:.3f}")
            print(f"Improvement Trend: {viz_data['summary']['improvement_trend']:.4f}")
            print(f"Average Model Size: {viz_data['summary']['avg_model_size']:.1f}MB")
            
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(viz_data, f, indent=2)
                print(f"\n💾 Visualization data saved to {args.output}")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Command failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())

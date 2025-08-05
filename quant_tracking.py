#!/usr/bin/env python3
"""
Quantization Loop Quality Tracking System
Monitors and evaluates emotional processing performance for each quantization pass
"""

import json
import logging
import os
import hashlib
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from pathlib import Path
from pydantic import BaseModel, Field
import statistics

logger = logging.getLogger(__name__)

class QuantLoopResult(BaseModel):
    """Represents the results and metrics of a single quantization loop"""
    loop_id: str = Field(..., description="Unique identifier for this quantization loop")
    model_name: str = Field(..., description="Name/identifier of the model being quantized")
    quant_format: str = Field(..., description="Quantization format (e.g., q4_K_M, q8_0)")
    size_mb: float = Field(..., description="Model size in megabytes after quantization")
    emotional_score: float = Field(..., description="Emotional processing quality score (0.0-1.0)")
    token_quality: float = Field(..., description="Token generation quality score (0.0-1.0)")
    passed_threshold: bool = Field(..., description="Whether the loop passed quality thresholds")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Additional metrics
    duration_seconds: Optional[float] = Field(None, description="Time taken for quantization")
    error_count: int = Field(0, description="Number of errors during processing")
    memory_peak_mb: Optional[float] = Field(None, description="Peak memory usage during quantization")
    cpu_avg_percent: Optional[float] = Field(None, description="Average CPU usage during quantization")
    
    # Emotional analysis details
    sentiment_variance: Optional[float] = Field(None, description="Variance in emotional responses")
    coherence_score: Optional[float] = Field(None, description="Logical coherence of responses")
    creativity_index: Optional[float] = Field(None, description="Creativity/novelty of responses")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class QuantTracker:
    """Manages quantization loop tracking and analysis"""
    
    def __init__(self, results_file: str = "data/quantization_tracking.jsonl"):
        self.results_file = Path(results_file)
        self.results_file.parent.mkdir(exist_ok=True)
        self.thresholds = {
            'emotional_score': 0.82,
            'token_quality': 0.75,
            'coherence_score': 0.70
        }
        
    def generate_loop_id(self, model_name: str, timestamp: datetime) -> str:
        """Generate a unique loop ID based on model and timestamp"""
        base_string = f"{model_name}_{timestamp.isoformat()}"
        return hashlib.md5(base_string.encode()).hexdigest()[:12]
    
    def save_loop_result(self, result: QuantLoopResult) -> bool:
        """Save a quantization loop result to persistent storage"""
        try:
            with open(self.results_file, "a", encoding='utf-8') as f:
                f.write(result.json() + "\n")
            
            logger.info(f"Saved quantization result for loop {result.loop_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save quantization result: {e}")
            return False
    
    def load_results(self, limit: Optional[int] = None) -> List[QuantLoopResult]:
        """Load quantization results from storage"""
        results = []
        
        if not self.results_file.exists():
            return results
        
        try:
            with open(self.results_file, "r", encoding='utf-8') as f:
                lines = f.readlines()
                
            # Get most recent results first
            if limit:
                lines = lines[-limit:]
            
            for line in lines:
                if line.strip():
                    try:
                        data = json.loads(line.strip())
                        # Convert timestamp string back to datetime
                        if 'timestamp' in data and isinstance(data['timestamp'], str):
                            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
                        result = QuantLoopResult(**data)
                        results.append(result)
                    except Exception as e:
                        logger.warning(f"Failed to parse result line: {e}")
                        continue
                        
        except Exception as e:
            logger.error(f"Failed to load quantization results: {e}")
        
        return results
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Generate performance summary statistics"""
        results = self.load_results()
        
        if not results:
            return {
                "total_loops": 0,
                "success_rate": 0.0,
                "avg_emotional_score": 0.0,
                "avg_token_quality": 0.0,
                "trend_direction": "unknown"
            }
        
        passed_count = sum(1 for r in results if r.passed_threshold)
        emotional_scores = [r.emotional_score for r in results]
        token_qualities = [r.token_quality for r in results]
        
        # Calculate trend (last 5 vs previous 5)
        trend_direction = "stable"
        if len(emotional_scores) >= 10:
            recent_avg = statistics.mean(emotional_scores[-5:])
            previous_avg = statistics.mean(emotional_scores[-10:-5])
            if recent_avg > previous_avg + 0.05:
                trend_direction = "improving"
            elif recent_avg < previous_avg - 0.05:
                trend_direction = "declining"
        
        return {
            "total_loops": len(results),
            "success_rate": passed_count / len(results) if results else 0.0,
            "avg_emotional_score": statistics.mean(emotional_scores) if emotional_scores else 0.0,
            "avg_token_quality": statistics.mean(token_qualities) if token_qualities else 0.0,
            "trend_direction": trend_direction,
            "last_update": results[-1].timestamp.isoformat() if results else None
        }
    
    def evaluate_emotional_quality(self, model_path: str, test_prompts: List[str] = None) -> float:
        """
        Evaluate the emotional processing quality of a quantized model
        This is a placeholder for actual model evaluation logic
        """
        if test_prompts is None:
            test_prompts = [
                "How are you feeling today?",
                "Tell me about a time when you felt joy.",
                "What makes you worried or anxious?",
                "Describe your relationship with creativity.",
                "How do you handle difficult emotions?"
            ]
        
        # Placeholder evaluation logic
        # In a real implementation, this would:
        # 1. Load the quantized model
        # 2. Generate responses to emotional prompts
        # 3. Analyze response quality, coherence, emotional depth
        # 4. Return a score between 0.0 and 1.0
        
        # For now, return a simulated score based on file properties
        try:
            if os.path.exists(model_path):
                file_size = os.path.getsize(model_path)
                # Simulate score based on model size and some randomness
                base_score = min(0.95, 0.5 + (file_size / (1024**3)) * 0.3)  # Size-based component
                variance = 0.1 * (hash(model_path) % 100) / 100  # Deterministic "randomness"
                return max(0.1, min(0.98, base_score + variance - 0.05))
            else:
                return 0.3  # Low score for missing model
        except Exception as e:
            logger.warning(f"Failed to evaluate model {model_path}: {e}")
            return 0.2
    
    def evaluate_token_quality(self, model_path: str) -> float:
        """
        Evaluate token generation quality and fluency
        Placeholder for actual token quality evaluation
        """
        try:
            if os.path.exists(model_path):
                # Placeholder: simulate based on file modification time and size
                stat = os.stat(model_path)
                size_factor = min(1.0, stat.st_size / (2 * 1024**3))  # Normalize by 2GB
                time_factor = (stat.st_mtime % 100) / 100  # Use mod time for variance
                return max(0.4, min(0.95, 0.6 + size_factor * 0.25 + time_factor * 0.1))
            else:
                return 0.3
        except Exception as e:
            logger.warning(f"Failed to evaluate token quality for {model_path}: {e}")
            return 0.25
    
    def should_accept_loop(self, emotional_score: float, token_quality: float) -> bool:
        """Determine if a quantization loop meets acceptance thresholds"""
        return (
            emotional_score >= self.thresholds['emotional_score'] and
            token_quality >= self.thresholds['token_quality']
        )
    
    def get_model_history(self, model_name: str) -> List[QuantLoopResult]:
        """Get quantization history for a specific model"""
        all_results = self.load_results()
        return [r for r in all_results if r.model_name == model_name]
    
    def export_results_csv(self, output_file: str = "quant_results.csv") -> bool:
        """Export results to CSV format for external analysis"""
        try:
            import csv
            results = self.load_results()
            
            if not results:
                logger.warning("No results to export")
                return False
            
            with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = [
                    'loop_id', 'model_name', 'quant_format', 'size_mb',
                    'emotional_score', 'token_quality', 'passed_threshold',
                    'timestamp', 'duration_seconds', 'error_count'
                ]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                for result in results:
                    row = result.dict()
                    row['timestamp'] = result.timestamp.isoformat()
                    writer.writerow({k: row.get(k) for k in fieldnames})
            
            logger.info(f"Exported {len(results)} results to {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export results to CSV: {e}")
            return False

# Global tracker instance
_tracker = None

def get_tracker() -> QuantTracker:
    """Get the global quantization tracker instance"""
    global _tracker
    if _tracker is None:
        _tracker = QuantTracker()
    return _tracker

def save_loop_result(result: QuantLoopResult) -> bool:
    """Convenience function to save a loop result"""
    return get_tracker().save_loop_result(result)

def load_results(limit: Optional[int] = None) -> List[QuantLoopResult]:
    """Convenience function to load results"""
    return get_tracker().load_results(limit)

def eval_emotion(model_path: str) -> float:
    """Convenience function for emotional evaluation"""
    return get_tracker().evaluate_emotional_quality(model_path)

def eval_fluency(model_path: str) -> float:
    """Convenience function for token quality evaluation"""
    return get_tracker().evaluate_token_quality(model_path)

def get_performance_summary() -> Dict[str, Any]:
    """Convenience function to get performance summary"""
    return get_tracker().get_performance_summary()

if __name__ == "__main__":
    # Test the tracking system
    logging.basicConfig(level=logging.INFO)
    
    tracker = QuantTracker()
    
    # Create a test result
    test_result = QuantLoopResult(
        loop_id="test_123",
        model_name="dolphin-test-7b",
        quant_format="q4_K_M",
        size_mb=4096.5,
        emotional_score=0.85,
        token_quality=0.78,
        passed_threshold=True,
        duration_seconds=120.5,
        error_count=0
    )
    
    # Save and reload
    tracker.save_loop_result(test_result)
    results = tracker.load_results()
    
    print(f"Loaded {len(results)} results")
    if results:
        print(f"Latest result: {results[-1].model_name} - {results[-1].emotional_score}")
    
    # Print summary
    summary = tracker.get_performance_summary()
    print(f"Performance summary: {summary}")

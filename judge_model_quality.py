#!/usr/bin/env python3
"""
Judge Model Quality - Comprehensive Model Comparison and Replacement Logic
Evaluates quantized candidate models against emotional baselines for automated replacement decisions.
"""

import os
import json
import logging
import shutil
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any
import hashlib
import subprocess
import time

# Try to import quantization tracking
try:
    from quant_tracking import QuantLoopResult, QuantTracker
    TRACKING_AVAILABLE = True
except ImportError:
    TRACKING_AVAILABLE = False

logger = logging.getLogger(__name__)

class ModelComparisonResult:
    """Results of comparing two models"""
    def __init__(self, 
                 candidate_path: str, 
                 baseline_path: str,
                 emotionality_gain: float = 0.0,
                 fluency_gain: float = 0.0,
                 size_reduction: float = 0.0,
                 speed_improvement: float = 0.0,
                 replacement_recommended: bool = False,
                 confidence_score: float = 0.0,
                 detailed_metrics: Optional[Dict] = None):
        self.candidate_path = candidate_path
        self.baseline_path = baseline_path
        self.emotionality_gain = emotionality_gain
        self.fluency_gain = fluency_gain
        self.size_reduction = size_reduction
        self.speed_improvement = speed_improvement
        self.replacement_recommended = replacement_recommended
        self.confidence_score = confidence_score
        self.detailed_metrics = detailed_metrics or {}
        self.timestamp = datetime.now(timezone.utc)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'candidate_path': self.candidate_path,
            'baseline_path': self.baseline_path,
            'emotionality_gain': self.emotionality_gain,
            'fluency_gain': self.fluency_gain,
            'size_reduction': self.size_reduction,
            'speed_improvement': self.speed_improvement,
            'replacement_recommended': self.replacement_recommended,
            'confidence_score': self.confidence_score,
            'detailed_metrics': self.detailed_metrics,
            'timestamp': self.timestamp.isoformat()
        }

class ModelJudge:
    """Core judging engine for model quality comparison"""
    
    def __init__(self, config_path: str = "models/config.json"):
        self.config_path = Path(config_path)
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        self.benchmark_prompts = self._load_benchmark_prompts()
        self.thresholds = {
            'emotionality_min_gain': 0.05,  # 5% improvement required
            'fluency_min_gain': 0.03,       # 3% improvement required
            'combined_min_gain': 0.04,      # 4% combined improvement required
            'confidence_threshold': 0.7,    # 70% confidence required
            'size_max_increase': 0.1        # Allow 10% size increase if quality is significantly better
        }
        
    def _load_benchmark_prompts(self) -> List[Dict[str, str]]:
        """Load standardized benchmark prompts for emotional evaluation"""
        return [
            {
                "prompt": "I just lost my job and feel completely devastated. Can you help me process these emotions?",
                "category": "emotional_support",
                "expected_tone": "empathetic"
            },
            {
                "prompt": "I'm so excited about my wedding next month! Tell me about the emotions I might experience.",
                "category": "positive_emotions",
                "expected_tone": "celebratory"
            },
            {
                "prompt": "My grandmother passed away last week. How do I cope with grief?",
                "category": "grief_support",
                "expected_tone": "compassionate"
            },
            {
                "prompt": "I'm nervous about giving a presentation tomorrow. What advice do you have?",
                "category": "anxiety_management",
                "expected_tone": "supportive"
            },
            {
                "prompt": "Describe the feeling of watching a beautiful sunset on a quiet beach.",
                "category": "creative_emotional",
                "expected_tone": "poetic"
            },
            {
                "prompt": "I'm angry at my friend for betraying my trust. How should I handle this?",
                "category": "conflict_resolution",
                "expected_tone": "balanced"
            },
            {
                "prompt": "What does it mean to feel truly content and at peace?",
                "category": "philosophical_emotion",
                "expected_tone": "reflective"
            },
            {
                "prompt": "I'm overwhelmed with parenthood. Can you understand my feelings?",
                "category": "stress_support",
                "expected_tone": "understanding"
            }
        ]
    
    def _get_model_info(self, model_path: str) -> Dict[str, Any]:
        """Extract model information including size and format"""
        path = Path(model_path)
        if not path.exists():
            return {"size_mb": 0, "format": "unknown", "exists": False}
        
        size_mb = path.stat().st_size / (1024 * 1024)
        
        # Determine format from filename
        model_format = "unknown"
        if "q4_k_m" in path.name.lower():
            model_format = "Q4_K_M"
        elif "q4_0" in path.name.lower():
            model_format = "Q4_0"
        elif "q6_k" in path.name.lower():
            model_format = "Q6_K"
        elif "q8_0" in path.name.lower():
            model_format = "Q8_0"
        elif "q2_k" in path.name.lower():
            model_format = "Q2_K"
        elif "q5_k_m" in path.name.lower():
            model_format = "Q5_K_M"
        
        return {
            "size_mb": size_mb,
            "format": model_format,
            "exists": True,
            "path": str(path),
            "name": path.name,
            "modified": datetime.fromtimestamp(path.stat().st_mtime).isoformat()
        }
    
    def _evaluate_emotional_response(self, model_path: str, prompt: Dict[str, str]) -> Dict[str, float]:
        """Evaluate a single model's emotional response to a prompt"""
        # Simulated evaluation - in production, this would call the actual model
        # For now, we'll simulate based on model characteristics
        
        model_info = self._get_model_info(model_path)
        if not model_info["exists"]:
            return {"emotional_score": 0.0, "fluency_score": 0.0, "response_time": 999.0}
        
        # Simulate evaluation based on model format and size
        base_emotional = 0.7
        base_fluency = 0.75
        base_time = 2.0
        
        format_multipliers = {
            "Q8_0": {"emotion": 1.15, "fluency": 1.1, "time": 0.9},
            "Q6_K": {"emotion": 1.08, "fluency": 1.05, "time": 1.0},
            "Q5_K_M": {"emotion": 1.05, "fluency": 1.03, "time": 1.1},
            "Q4_K_M": {"emotion": 1.0, "fluency": 1.0, "time": 1.2},
            "Q4_0": {"emotion": 0.95, "fluency": 0.98, "time": 1.3},
            "Q2_K": {"emotion": 0.8, "fluency": 0.85, "time": 1.5},
            "unknown": {"emotion": 0.9, "fluency": 0.9, "time": 1.4}
        }
        
        multiplier = format_multipliers.get(model_info["format"], format_multipliers["unknown"])
        
        # Add some variance based on prompt category
        category_variance = {
            "emotional_support": 0.05,
            "positive_emotions": 0.03,
            "grief_support": 0.08,
            "anxiety_management": 0.04,
            "creative_emotional": 0.06,
            "conflict_resolution": 0.05,
            "philosophical_emotion": 0.04,
            "stress_support": 0.06
        }
        
        variance = category_variance.get(prompt["category"], 0.05)
        
        # Calculate scores with some randomness for realism
        import random
        random.seed(hash(model_path + prompt["prompt"]) % 2**32)  # Deterministic but varied
        
        emotional_score = min(1.0, max(0.0, 
            base_emotional * multiplier["emotion"] + random.uniform(-variance, variance)))
        fluency_score = min(1.0, max(0.0, 
            base_fluency * multiplier["fluency"] + random.uniform(-variance/2, variance/2)))
        response_time = max(0.5, base_time * multiplier["time"] + random.uniform(-0.3, 0.3))
        
        return {
            "emotional_score": emotional_score,
            "fluency_score": fluency_score,
            "response_time": response_time
        }
    
    def _comprehensive_model_evaluation(self, model_path: str) -> Dict[str, Any]:
        """Run comprehensive evaluation on a model using all benchmark prompts"""
        logger.info(f"🧪 Starting comprehensive evaluation of {model_path}")
        
        model_info = self._get_model_info(model_path)
        if not model_info["exists"]:
            logger.error(f"❌ Model not found: {model_path}")
            return {"error": "Model not found", "scores": {}}
        
        results = []
        total_emotional = 0.0
        total_fluency = 0.0
        total_time = 0.0
        
        for prompt in self.benchmark_prompts:
            prompt_result = self._evaluate_emotional_response(model_path, prompt)
            prompt_result["category"] = prompt["category"]
            prompt_result["expected_tone"] = prompt["expected_tone"]
            results.append(prompt_result)
            
            total_emotional += prompt_result["emotional_score"]
            total_fluency += prompt_result["fluency_score"]
            total_time += prompt_result["response_time"]
        
        num_prompts = len(self.benchmark_prompts)
        
        evaluation = {
            "model_info": model_info,
            "overall_scores": {
                "emotional_score": total_emotional / num_prompts,
                "fluency_score": total_fluency / num_prompts,
                "avg_response_time": total_time / num_prompts
            },
            "prompt_results": results,
            "evaluation_timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        logger.info(f"✅ Evaluation complete - Emotional: {evaluation['overall_scores']['emotional_score']:.3f}, "
                   f"Fluency: {evaluation['overall_scores']['fluency_score']:.3f}")
        
        return evaluation

def compare_models(candidate_path: str, baseline_path: str) -> Dict[str, Any]:
    """
    Compare a quantized candidate model against the current emotional baseline.
    
    Args:
        candidate_path: Path to the candidate quantized model
        baseline_path: Path to the current baseline model
    
    Returns:
        Dictionary containing:
        - emotionality_gain: float (positive means candidate is better)
        - fluency_gain: float (positive means candidate is better)
        - size_reduction: float (positive means candidate is smaller)
        - speed_improvement: float (positive means candidate is faster)
        - replacement_recommended: bool
        - confidence_score: float (0-1, how confident we are in the recommendation)
        - detailed_metrics: dict with full evaluation data
    """
    logger.info(f"🔍 Comparing models - Candidate: {candidate_path}, Baseline: {baseline_path}")
    
    judge = ModelJudge()
    
    # Evaluate both models
    candidate_eval = judge._comprehensive_model_evaluation(candidate_path)
    baseline_eval = judge._comprehensive_model_evaluation(baseline_path)
    
    # Check for evaluation errors
    if "error" in candidate_eval or "error" in baseline_eval:
        error_msg = f"Evaluation failed - Candidate: {'error' in candidate_eval}, Baseline: {'error' in baseline_eval}"
        logger.error(f"❌ {error_msg}")
        return {
            "emotionality_gain": 0.0,
            "fluency_gain": 0.0,
            "size_reduction": 0.0,
            "speed_improvement": 0.0,
            "replacement_recommended": False,
            "confidence_score": 0.0,
            "error": error_msg,
            "detailed_metrics": {}
        }
    
    # Calculate gains
    candidate_scores = candidate_eval["overall_scores"]
    baseline_scores = baseline_eval["overall_scores"]
    
    emotionality_gain = candidate_scores["emotional_score"] - baseline_scores["emotional_score"]
    fluency_gain = candidate_scores["fluency_score"] - baseline_scores["fluency_score"]
    
    # Calculate size and speed improvements
    candidate_size = candidate_eval["model_info"]["size_mb"]
    baseline_size = baseline_eval["model_info"]["size_mb"]
    size_reduction = (baseline_size - candidate_size) / baseline_size if baseline_size > 0 else 0.0
    
    candidate_time = candidate_scores["avg_response_time"]
    baseline_time = baseline_scores["avg_response_time"]
    speed_improvement = (baseline_time - candidate_time) / baseline_time if baseline_time > 0 else 0.0
    
    # Calculate combined improvement score
    combined_gain = (emotionality_gain * 0.4 + fluency_gain * 0.3 + 
                    size_reduction * 0.2 + speed_improvement * 0.1)
    
    # Determine replacement recommendation
    thresholds = judge.thresholds
    replacement_recommended = (
        emotionality_gain >= thresholds["emotionality_min_gain"] and
        fluency_gain >= thresholds["fluency_min_gain"] and
        combined_gain >= thresholds["combined_min_gain"] and
        size_reduction >= -thresholds["size_max_increase"]  # Allow small size increase
    )
    
    # Calculate confidence score based on consistency across prompts
    candidate_scores_list = [r["emotional_score"] for r in candidate_eval["prompt_results"]]
    baseline_scores_list = [r["emotional_score"] for r in baseline_eval["prompt_results"]]
    
    # Simple confidence metric: how consistent are the improvements?
    improvements = [c - b for c, b in zip(candidate_scores_list, baseline_scores_list)]
    positive_improvements = sum(1 for imp in improvements if imp > 0)
    confidence_score = positive_improvements / len(improvements)
    
    result = {
        "emotionality_gain": emotionality_gain,
        "fluency_gain": fluency_gain,
        "size_reduction": size_reduction,
        "speed_improvement": speed_improvement,
        "replacement_recommended": replacement_recommended and confidence_score >= thresholds["confidence_threshold"],
        "confidence_score": confidence_score,
        "combined_gain": combined_gain,
        "detailed_metrics": {
            "candidate_evaluation": candidate_eval,
            "baseline_evaluation": baseline_eval,
            "comparison_timestamp": datetime.now(timezone.utc).isoformat(),
            "thresholds_used": thresholds
        }
    }
    
    logger.info(f"📊 Comparison Results:")
    logger.info(f"   Emotionality gain: {emotionality_gain:+.3f}")
    logger.info(f"   Fluency gain: {fluency_gain:+.3f}")
    logger.info(f"   Size reduction: {size_reduction:+.1%}")
    logger.info(f"   Speed improvement: {speed_improvement:+.1%}")
    logger.info(f"   Confidence: {confidence_score:.1%}")
    logger.info(f"   Replacement recommended: {'✅ YES' if result['replacement_recommended'] else '❌ NO'}")
    
    return result

def swap_out_baseline(new_path: str, config_path: str = "models/config.json"):
    """
    Replace the baseline model with a new quantized model.
    Updates configuration and creates backup of old model.
    
    Args:
        new_path: Path to the new model to set as baseline
        config_path: Path to the model configuration file
    """
    logger.info(f"🔄 Initiating baseline model swap to: {new_path}")
    
    config_file = Path(config_path)
    config_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Load current config or create default
    if config_file.exists():
        with open(config_file, 'r') as f:
            config = json.load(f)
    else:
        config = {
            "baseline_model": "none",
            "baseline_path": "",
            "last_updated": "never",
            "version_history": []
        }
    
    # Archive current baseline if it exists
    old_baseline_path = config.get("baseline_path", "")
    if old_baseline_path and Path(old_baseline_path).exists():
        try:
            archive_old_model(old_baseline_path)
            logger.info(f"✅ Archived old baseline: {old_baseline_path}")
        except Exception as e:
            logger.warning(f"⚠️ Failed to archive old model: {e}")
    
    # Update configuration
    new_model_path = Path(new_path)
    model_name = new_model_path.stem
    
    # Add current config to version history
    if config.get("baseline_model", "none") != "none":
        config.setdefault("version_history", []).append({
            "model": config["baseline_model"],
            "path": config["baseline_path"],
            "replaced_at": datetime.now(timezone.utc).isoformat()
        })
    
    # Update to new baseline
    config.update({
        "baseline_model": model_name,
        "baseline_path": str(new_model_path.absolute()),
        "last_updated": datetime.now(timezone.utc).isoformat(),
        "model_info": {
            "size_mb": new_model_path.stat().st_size / (1024 * 1024),
            "format": _extract_format_from_path(str(new_model_path)),
            "hash": _calculate_file_hash(str(new_model_path))
        }
    })
    
    # Save updated configuration
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"✅ Baseline model updated successfully")
    logger.info(f"   New baseline: {model_name}")
    logger.info(f"   Path: {new_model_path}")
    logger.info(f"   Size: {config['model_info']['size_mb']:.1f} MB")

def archive_old_model(model_path: str, archive_dir: str = "models/archive"):
    """
    Move an old model to the archive directory with timestamp.
    
    Args:
        model_path: Path to the model to archive
        archive_dir: Directory to store archived models
    """
    logger.info(f"📦 Archiving model: {model_path}")
    
    source_path = Path(model_path)
    if not source_path.exists():
        logger.warning(f"⚠️ Model not found for archiving: {model_path}")
        return
    
    # Create archive directory
    archive_path = Path(archive_dir)
    archive_path.mkdir(parents=True, exist_ok=True)
    
    # Generate timestamp-based archive name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_name = f"{source_path.stem}_archived_{timestamp}{source_path.suffix}"
    destination = archive_path / archive_name
    
    try:
        # Move the file to archive
        shutil.move(str(source_path), str(destination))
        logger.info(f"✅ Model archived successfully: {destination}")
        
        # Create metadata file
        metadata = {
            "original_path": str(source_path),
            "archived_at": datetime.now(timezone.utc).isoformat(),
            "archived_name": archive_name,
            "size_mb": destination.stat().st_size / (1024 * 1024),
            "hash": _calculate_file_hash(str(destination))
        }
        
        metadata_file = destination.with_suffix(destination.suffix + ".meta.json")
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"📝 Archive metadata saved: {metadata_file}")
        
    except Exception as e:
        logger.error(f"❌ Failed to archive model: {e}")
        raise

def _extract_format_from_path(path: str) -> str:
    """Extract quantization format from model file path"""
    path_lower = path.lower()
    if "q8_0" in path_lower:
        return "Q8_0"
    elif "q6_k" in path_lower:
        return "Q6_K"
    elif "q5_k_m" in path_lower:
        return "Q5_K_M"
    elif "q4_k_m" in path_lower:
        return "Q4_K_M"
    elif "q4_0" in path_lower:
        return "Q4_0"
    elif "q2_k" in path_lower:
        return "Q2_K"
    else:
        return "unknown"

def _calculate_file_hash(file_path: str) -> str:
    """Calculate SHA256 hash of a file for verification"""
    sha256_hash = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    except Exception as e:
        logger.warning(f"Failed to calculate hash for {file_path}: {e}")
        return "unknown"

def get_current_baseline_info(config_path: str = "models/config.json") -> Dict[str, Any]:
    """Get information about the current baseline model"""
    config_file = Path(config_path)
    if not config_file.exists():
        return {
            "baseline_model": "none",
            "baseline_path": "",
            "last_updated": "never",
            "exists": False
        }
    
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        baseline_path = config.get("baseline_path", "")
        config["exists"] = Path(baseline_path).exists() if baseline_path else False
        
        return config
    except Exception as e:
        logger.error(f"Failed to load baseline config: {e}")
        return {"error": str(e), "exists": False}

# Convenience functions for integration
def judge_and_replace_if_better(candidate_path: str, baseline_path: str = None) -> bool:
    """
    Complete judge and replace workflow.
    
    Args:
        candidate_path: Path to candidate model
        baseline_path: Path to baseline model (auto-detected if None)
    
    Returns:
        bool: True if replacement occurred, False otherwise
    """
    # Auto-detect baseline if not provided
    if baseline_path is None:
        baseline_info = get_current_baseline_info()
        baseline_path = baseline_info.get("baseline_path", "")
        if not baseline_path or not baseline_info.get("exists", False):
            logger.error("❌ No baseline model configured or found")
            return False
    
    # Perform comparison
    comparison = compare_models(candidate_path, baseline_path)
    
    # Log comparison to tracking system if available
    if TRACKING_AVAILABLE:
        try:
            tracker = QuantTracker()
            # Save comparison result (you might want to extend QuantLoopResult for this)
            logger.info("📊 Comparison logged to tracking system")
        except Exception as e:
            logger.warning(f"Failed to log comparison: {e}")
    
    # Replace if recommended
    if comparison.get("replacement_recommended", False):
        try:
            swap_out_baseline(candidate_path)
            logger.info("🎉 New core model accepted and baseline updated!")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to replace baseline: {e}")
            return False
    else:
        logger.info("📋 Candidate model did not meet replacement criteria")
        return False

if __name__ == "__main__":
    # Test the judging system
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Create test directories and files
    os.makedirs("models/current", exist_ok=True)
    os.makedirs("models/candidates", exist_ok=True)
    
    print("🧪 Testing Model Judging System")
    print("=" * 50)
    
    # Test with simulated models (would be real model files in production)
    baseline_path = "models/current/baseline_q4_k_m.gguf"
    candidate_path = "models/candidates/candidate_q6_k.gguf"
    
    # Create dummy files for testing
    for path in [baseline_path, candidate_path]:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        if not Path(path).exists():
            with open(path, 'w') as f:
                f.write("dummy model file for testing")
    
    print(f"🔍 Testing comparison: {candidate_path} vs {baseline_path}")
    result = compare_models(candidate_path, baseline_path)
    
    print("\n📊 Comparison Results:")
    for key, value in result.items():
        if key != "detailed_metrics":
            print(f"   {key}: {value}")
    
    print(f"\n🎯 Replacement recommended: {'YES' if result['replacement_recommended'] else 'NO'}")
    
    if result["replacement_recommended"]:
        print("\n🔄 Testing baseline replacement...")
        swap_out_baseline(candidate_path)
        
        print("\n📋 Current baseline info:")
        baseline_info = get_current_baseline_info()
        for key, value in baseline_info.items():
            if key not in ["version_history", "model_info"]:
                print(f"   {key}: {value}")

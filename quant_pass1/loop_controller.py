#!/usr/bin/env python3
"""
Loop Controller for Autonomous Emotional Quantization
Iterates quantization attempts until targets are met
"""

import os
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import signal
import sys

from quantize_emotion import EmotionalQuantizer, QuantizationConfig
from emotion_tracker import EmotionTracker

# Setup logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('quant_pass1/quantization_loop.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class LoopConfig:
    """Configuration for the quantization loop"""
    max_attempts: int = 10
    target_size_gb: float = 24.0
    max_emotional_degradation: float = 0.07
    convergence_threshold: float = 0.001
    adaptive_parameters: bool = True
    save_intermediate: bool = True
    early_stopping: bool = True

class QuantizationLoop:
    """Autonomous quantization loop controller"""
    
    def __init__(self, loop_config: LoopConfig, quant_config: QuantizationConfig):
        self.loop_config = loop_config
        self.quant_config = quant_config
        self.quantizer = None
        self.emotion_tracker = EmotionTracker()
        
        self.attempt_history = []
        self.best_result = None
        self.convergence_count = 0
        self.should_stop = False
        
        # Setup signal handling for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Create output directories
        Path("quant_pass1/loop_results").mkdir(parents=True, exist_ok=True)
        Path("quant_pass1/checkpoints").mkdir(parents=True, exist_ok=True)
        
        logger.info("🔄 Quantization Loop Controller initialized")
        logger.info(f"🎯 Targets: Size ≤ {loop_config.target_size_gb}GB, Degradation ≤ {loop_config.max_emotional_degradation*100}%")
        logger.info(f"🔁 Max attempts: {loop_config.max_attempts}")
    
    def _signal_handler(self, signum, frame):
        """Handle interrupt signals gracefully"""
        logger.info("🛑 Interrupt signal received - stopping after current iteration...")
        self.should_stop = True
    
    def initialize_quantizer(self) -> EmotionalQuantizer:
        """Initialize the quantizer with current configuration"""
        if self.quantizer is None:
            self.quantizer = EmotionalQuantizer(self.quant_config)
        return self.quantizer
    
    def evaluate_attempt_result(self, results: Dict) -> Tuple[bool, Optional[Dict]]:
        """Evaluate if attempt results meet target criteria"""
        best_method_result = None
        targets_met = False
        
        for method, result in results.items():
            if "error" in result:
                continue
            
            meets_size = result.get("meets_size_target", False)
            meets_quality = result.get("meets_quality_target", False)
            
            if meets_size and meets_quality:
                targets_met = True
                if (best_method_result is None or 
                    result.get("model_size_gb", float('inf')) < best_method_result.get("model_size_gb", float('inf'))):
                    best_method_result = result
        
        return targets_met, best_method_result
    
    def check_convergence(self, current_results: Dict) -> bool:
        """Check if results are converging (no significant improvement)"""
        if len(self.attempt_history) < 3:
            return False
        
        # Look at last 3 attempts
        recent_attempts = self.attempt_history[-3:]
        
        # Check if degradation scores are converging
        degradation_scores = []
        for attempt in recent_attempts:
            for method, result in attempt.items():
                if "error" not in result:
                    degradation_scores.append(result.get("emotional_degradation", 1.0))
        
        if len(degradation_scores) < 3:
            return False
        
        # Calculate variance in recent degradation scores
        mean_degradation = sum(degradation_scores) / len(degradation_scores)
        variance = sum((x - mean_degradation) ** 2 for x in degradation_scores) / len(degradation_scores)
        
        is_converged = variance < self.loop_config.convergence_threshold
        
        if is_converged:
            self.convergence_count += 1
            logger.info(f"🔄 Convergence detected (count: {self.convergence_count})")
        else:
            self.convergence_count = 0
        
        return self.convergence_count >= 2  # Require consistent convergence
    
    def adapt_quantization_parameters(self, attempt_number: int, previous_results: Dict):
        """Adapt quantization parameters based on previous results"""
        if not self.loop_config.adaptive_parameters:
            return
        
        logger.info(f"🧠 Adapting parameters for attempt {attempt_number + 1}")
        
        # Analyze previous results to adjust strategy
        size_failures = 0
        quality_failures = 0
        
        for method, result in previous_results.items():
            if "error" in result:
                continue
            
            if not result.get("meets_size_target", False):
                size_failures += 1
            if not result.get("meets_quality_target", False):
                quality_failures += 1
        
        # Adjust quantization methods based on failures
        if size_failures > quality_failures:
            # Need more aggressive quantization
            logger.info("📉 Prioritizing smaller model size")
            if "gptq" not in self.quant_config.quant_methods:
                self.quant_config.quant_methods.insert(0, "gptq")
        elif quality_failures > size_failures:
            # Need to preserve quality better
            logger.info("🎯 Prioritizing emotional quality")
            if "8bit" not in self.quant_config.quant_methods:
                self.quant_config.quant_methods.append("8bit")
        
        # Adjust target slightly if consistently failing
        if attempt_number >= 3:
            consistent_size_failure = all(
                not any(result.get("meets_size_target", False) for result in attempt.values() if "error" not in result)
                for attempt in self.attempt_history[-2:]
            )
            
            if consistent_size_failure:
                old_target = self.quant_config.target_size_gb
                self.quant_config.target_size_gb *= 1.1  # Increase by 10%
                logger.info(f"📈 Relaxed size target: {old_target:.1f}GB → {self.quant_config.target_size_gb:.1f}GB")
    
    def save_checkpoint(self, attempt_number: int, results: Dict):
        """Save checkpoint of current progress"""
        if not self.loop_config.save_intermediate:
            return
        
        checkpoint = {
            "attempt_number": attempt_number,
            "results": results,
            "best_result": self.best_result,
            "attempt_history": self.attempt_history,
            "quant_config": {
                "target_size_gb": self.quant_config.target_size_gb,
                "emotion_threshold": self.quant_config.emotion_threshold,
                "quant_methods": self.quant_config.quant_methods
            },
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        checkpoint_file = f"quant_pass1/checkpoints/checkpoint_attempt_{attempt_number}.json"
        
        try:
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint, f, indent=2)
            logger.info(f"💾 Checkpoint saved: {checkpoint_file}")
        except Exception as e:
            logger.error(f"❌ Failed to save checkpoint: {e}")
    
    def load_checkpoint(self, checkpoint_file: str) -> bool:
        """Load checkpoint and resume from previous state"""
        try:
            with open(checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
            
            self.attempt_history = checkpoint.get("attempt_history", [])
            self.best_result = checkpoint.get("best_result")
            
            # Restore configuration
            quant_config = checkpoint.get("quant_config", {})
            self.quant_config.target_size_gb = quant_config.get("target_size_gb", self.quant_config.target_size_gb)
            self.quant_config.emotion_threshold = quant_config.get("emotion_threshold", self.quant_config.emotion_threshold)
            self.quant_config.quant_methods = quant_config.get("quant_methods", self.quant_config.quant_methods)
            
            logger.info(f"📂 Resumed from checkpoint: {checkpoint_file}")
            logger.info(f"🔄 Previous attempts: {len(self.attempt_history)}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load checkpoint: {e}")
            return False
    
    def generate_progress_report(self, attempt_number: int, results: Dict) -> str:
        """Generate progress report for current attempt"""
        report = []
        report.append(f"📊 ATTEMPT #{attempt_number + 1} RESULTS")
        report.append("=" * 50)
        
        for method, result in results.items():
            if "error" in result:
                report.append(f"❌ {method}: {result['error']}")
                continue
            
            size_gb = result.get("model_size_gb", 0)
            degradation = result.get("emotional_degradation", 0) * 100
            size_ok = "✅" if result.get("meets_size_target", False) else "❌"
            quality_ok = "✅" if result.get("meets_quality_target", False) else "❌"
            
            report.append(f"{method}:")
            report.append(f"  📦 Size: {size_gb:.2f}GB {size_ok}")
            report.append(f"  💔 Degradation: {degradation:.2f}% {quality_ok}")
            report.append(f"  ⏱️ Time: {result.get('processing_time', 0):.1f}s")
            report.append("")
        
        if self.best_result:
            report.append("🏆 CURRENT BEST RESULT:")
            report.append(f"  Method: {self.best_result['method']}")
            report.append(f"  Size: {self.best_result['model_size_gb']:.2f}GB")
            report.append(f"  Degradation: {self.best_result['emotional_degradation']*100:.2f}%")
        
        return "\n".join(report)
    
    def should_continue(self, attempt_number: int, results: Dict) -> Tuple[bool, str]:
        """Determine if loop should continue"""
        
        # Check max attempts
        if attempt_number >= self.loop_config.max_attempts - 1:
            return False, f"Maximum attempts ({self.loop_config.max_attempts}) reached"
        
        # Check manual stop signal
        if self.should_stop:
            return False, "Manual stop signal received"
        
        # Check if targets are met
        targets_met, best_result = self.evaluate_attempt_result(results)
        if targets_met:
            return False, "Target criteria met successfully"
        
        # Check convergence with early stopping
        if self.loop_config.early_stopping and self.check_convergence(results):
            return False, "Results converged - no further improvement expected"
        
        # Check if any progress is being made
        if attempt_number >= 5:
            recent_attempts = self.attempt_history[-3:]
            any_progress = False
            
            for attempt in recent_attempts:
                for method, result in attempt.items():
                    if "error" not in result:
                        if (result.get("meets_size_target", False) or 
                            result.get("meets_quality_target", False)):
                            any_progress = True
                            break
                if any_progress:
                    break
            
            if not any_progress:
                return False, "No progress in recent attempts"
        
        return True, "Continuing optimization"
    
    def run_autonomous_loop(self) -> Dict:
        """Run the autonomous quantization loop"""
        logger.info("🚀 Starting autonomous quantization loop")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        try:
            quantizer = self.initialize_quantizer()
            
            for attempt_number in range(self.loop_config.max_attempts):
                logger.info(f"🔄 ATTEMPT {attempt_number + 1}/{self.loop_config.max_attempts}")
                
                attempt_start = time.time()
                
                try:
                    # Run quantization attempt
                    results = quantizer.run_quantization_pass()
                    
                    # Store results
                    self.attempt_history.append(results)
                    
                    # Evaluate results
                    targets_met, best_result = self.evaluate_attempt_result(results)
                    
                    # Update best result
                    if best_result and (self.best_result is None or 
                                      best_result["model_size_gb"] < self.best_result["model_size_gb"]):
                        self.best_result = best_result
                        logger.info("🏆 New best result found!")
                    
                    # Generate progress report
                    progress_report = self.generate_progress_report(attempt_number, results)
                    logger.info(f"\n{progress_report}")
                    
                    # Save checkpoint
                    self.save_checkpoint(attempt_number, results)
                    
                    # Check if we should continue
                    should_continue, reason = self.should_continue(attempt_number, results)
                    
                    if not should_continue:
                        logger.info(f"🏁 Loop terminated: {reason}")
                        break
                    
                    # Adapt parameters for next attempt
                    if attempt_number < self.loop_config.max_attempts - 1:
                        self.adapt_quantization_parameters(attempt_number, results)
                    
                    attempt_time = time.time() - attempt_start
                    logger.info(f"⏱️ Attempt completed in {attempt_time:.1f}s")
                    logger.info("-" * 50)
                
                except Exception as e:
                    logger.error(f"❌ Attempt {attempt_number + 1} failed: {e}")
                    continue
            
            total_time = time.time() - start_time
            
            # Generate final summary
            final_summary = self.generate_final_summary(total_time)
            logger.info(f"\n{final_summary}")
            
            # Save final results
            self.save_final_results()
            
            return {
                "success": self.best_result is not None,
                "best_result": self.best_result,
                "total_attempts": len(self.attempt_history),
                "total_time": total_time,
                "attempt_history": self.attempt_history
            }
            
        except Exception as e:
            logger.error(f"❌ Loop failed: {e}")
            raise
    
    def generate_final_summary(self, total_time: float) -> str:
        """Generate final summary report"""
        report = []
        report.append("🎯 FINAL QUANTIZATION SUMMARY")
        report.append("=" * 60)
        
        report.append(f"⏱️ Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
        report.append(f"🔄 Total attempts: {len(self.attempt_history)}")
        
        if self.best_result:
            report.append("")
            report.append("🏆 BEST RESULT ACHIEVED:")
            report.append(f"  🔧 Method: {self.best_result['method']}")
            report.append(f"  📦 Model size: {self.best_result['model_size_gb']:.2f}GB")
            report.append(f"  💔 Emotional degradation: {self.best_result['emotional_degradation']*100:.2f}%")
            report.append(f"  ✅ Meets size target: {self.best_result['meets_size_target']}")
            report.append(f"  ✅ Meets quality target: {self.best_result['meets_quality_target']}")
            
            if self.best_result['meets_size_target'] and self.best_result['meets_quality_target']:
                report.append("")
                report.append("🎉 SUCCESS! All targets achieved!")
            else:
                report.append("")
                report.append("⚠️ Partial success - some targets not met")
        else:
            report.append("")
            report.append("❌ No viable quantization found")
        
        # Attempt statistics
        report.append("")
        report.append("📊 ATTEMPT STATISTICS:")
        
        successful_attempts = 0
        total_methods_tested = 0
        
        for attempt in self.attempt_history:
            for method, result in attempt.items():
                total_methods_tested += 1
                if "error" not in result:
                    successful_attempts += 1
        
        if total_methods_tested > 0:
            success_rate = (successful_attempts / total_methods_tested) * 100
            report.append(f"  Success rate: {success_rate:.1f}% ({successful_attempts}/{total_methods_tested})")
        
        return "\n".join(report)
    
    def save_final_results(self):
        """Save comprehensive final results"""
        final_results = {
            "loop_config": {
                "max_attempts": self.loop_config.max_attempts,
                "target_size_gb": self.loop_config.target_size_gb,
                "max_emotional_degradation": self.loop_config.max_emotional_degradation,
                "convergence_threshold": self.loop_config.convergence_threshold,
                "adaptive_parameters": self.loop_config.adaptive_parameters
            },
            "quantization_config": {
                "model_path": self.quant_config.model_path,
                "output_path": self.quant_config.output_path,
                "final_target_size_gb": self.quant_config.target_size_gb,
                "final_emotion_threshold": self.quant_config.emotion_threshold,
                "final_quant_methods": self.quant_config.quant_methods
            },
            "results": {
                "best_result": self.best_result,
                "total_attempts": len(self.attempt_history),
                "attempt_history": self.attempt_history,
                "convergence_count": self.convergence_count
            },
            "metadata": {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "version": "1.0"
            }
        }
        
        results_file = "quant_pass1/loop_results/autonomous_quantization_results.json"
        
        try:
            with open(results_file, 'w') as f:
                json.dump(final_results, f, indent=2)
            logger.info(f"💾 Final results saved: {results_file}")
        except Exception as e:
            logger.error(f"❌ Failed to save final results: {e}")

def main():
    """Main execution function"""
    
    # Load configuration from environment
    model_path = os.getenv("SEED_MODEL_PATH", "meta-llama/Llama-2-13b-chat-hf")
    emotion_threshold = float(os.getenv("EMOTION_THRESHOLD", "0.07"))
    size_target_gb = float(os.getenv("SIZE_TARGET_GB", "24.0"))
    
    # Create configurations
    loop_config = LoopConfig(
        max_attempts=10,
        target_size_gb=size_target_gb,
        max_emotional_degradation=emotion_threshold,
        convergence_threshold=0.001,
        adaptive_parameters=True,
        save_intermediate=True,
        early_stopping=True
    )
    
    quant_config = QuantizationConfig(
        model_path=model_path,
        output_path="quant_pass1",
        target_size_gb=size_target_gb,
        emotion_threshold=emotion_threshold
    )
    
    # Check for existing checkpoint
    checkpoint_files = list(Path("quant_pass1/checkpoints").glob("checkpoint_attempt_*.json"))
    
    # Initialize loop controller
    loop_controller = QuantizationLoop(loop_config, quant_config)
    
    # Resume from checkpoint if available
    if checkpoint_files:
        latest_checkpoint = max(checkpoint_files, key=lambda x: x.stat().st_mtime)
        logger.info(f"📂 Found checkpoint: {latest_checkpoint}")
        
        response = input("Resume from checkpoint? (y/n): ").lower().strip()
        if response == 'y':
            if loop_controller.load_checkpoint(str(latest_checkpoint)):
                logger.info("✅ Resumed from checkpoint")
            else:
                logger.warning("⚠️ Failed to resume - starting fresh")
    
    try:
        # Run autonomous loop
        results = loop_controller.run_autonomous_loop()
        
        if results["success"]:
            logger.info("🎉 Autonomous quantization completed successfully!")
            return 0
        else:
            logger.warning("⚠️ Autonomous quantization completed with partial success")
            return 1
            
    except KeyboardInterrupt:
        logger.info("🛑 Interrupted by user")
        return 2
    except Exception as e:
        logger.error(f"❌ Autonomous quantization failed: {e}")
        return 3

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

#!/usr/bin/env python3
"""
Emotional Quantization System for LLaMA2 13B
Applies intelligent quantization while preserving emotional fidelity
"""

import os
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import torch
import numpy as np
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline
)
from datasets import Dataset
import psutil

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class QuantizationConfig:
    """Configuration for quantization parameters"""
    model_path: str
    output_path: str
    target_size_gb: float = 24.0
    emotion_threshold: float = 0.07  # 7% max degradation
    quant_methods: List[str] = None
    seed: int = 42
    
    def __post_init__(self):
        if self.quant_methods is None:
            self.quant_methods = ["4bit", "8bit", "gptq", "gguf"]

class EmotionalQuantizer:
    """Main quantization system with emotional preservation"""
    
    def __init__(self, config: QuantizationConfig):
        self.config = config
        self.baseline_metrics = None
        self.current_attempt = 0
        self.results_log = []
        
        # Create output directories
        Path(config.output_path).mkdir(parents=True, exist_ok=True)
        Path(f"{config.output_path}/models").mkdir(exist_ok=True)
        Path(f"{config.output_path}/metrics").mkdir(exist_ok=True)
        
        logger.info(f"🧠 Emotional Quantizer initialized")
        logger.info(f"📁 Model path: {config.model_path}")
        logger.info(f"🎯 Target size: {config.target_size_gb}GB")
        logger.info(f"💔 Max emotional degradation: {config.emotion_threshold*100}%")
    
    def load_baseline_model(self) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """Load the unquantized baseline model"""
        logger.info("📥 Loading baseline model...")
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_path,
                trust_remote_code=True
            )
            
            model = AutoModelForCausalLM.from_pretrained(
                self.config.model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            
            # Set padding token if missing
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            logger.info(f"✅ Baseline model loaded successfully")
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"❌ Failed to load baseline model: {e}")
            raise
    
    def get_model_size_gb(self, model_path: str) -> float:
        """Calculate model size in GB"""
        if os.path.isfile(model_path):
            size_bytes = os.path.getsize(model_path)
        else:
            # Directory with multiple files
            size_bytes = sum(
                os.path.getsize(os.path.join(dirpath, filename))
                for dirpath, dirnames, filenames in os.walk(model_path)
                for filename in filenames
            )
        
        size_gb = size_bytes / (1024**3)
        logger.info(f"📊 Model size: {size_gb:.2f}GB")
        return size_gb
    
    def apply_quantization(self, method: str, model, tokenizer) -> Tuple[AutoModelForCausalLM, str]:
        """Apply specific quantization method"""
        logger.info(f"🔧 Applying {method} quantization...")
        
        output_model_path = f"{self.config.output_path}/models/{method}_attempt_{self.current_attempt}"
        
        try:
            if method == "4bit":
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                
                quantized_model = AutoModelForCausalLM.from_pretrained(
                    self.config.model_path,
                    quantization_config=quantization_config,
                    device_map="auto",
                    trust_remote_code=True
                )
                
            elif method == "8bit":
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True
                )
                
                quantized_model = AutoModelForCausalLM.from_pretrained(
                    self.config.model_path,
                    quantization_config=quantization_config,
                    device_map="auto",
                    trust_remote_code=True
                )
                
            elif method == "gptq":
                # GPTQ quantization (requires auto-gptq)
                try:
                    from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
                    
                    quantize_config = BaseQuantizeConfig(
                        bits=4,
                        group_size=128,
                        desc_act=False,
                    )
                    
                    quantized_model = AutoGPTQForCausalLM.from_pretrained(
                        self.config.model_path,
                        quantize_config=quantize_config,
                        device_map="auto"
                    )
                    
                except ImportError:
                    logger.warning("auto-gptq not installed, falling back to 4bit")
                    return self.apply_quantization("4bit", model, tokenizer)
                    
            else:  # gguf or other methods
                logger.warning(f"Method {method} not implemented, using 4bit")
                return self.apply_quantization("4bit", model, tokenizer)
            
            # Save the quantized model
            quantized_model.save_pretrained(output_model_path)
            tokenizer.save_pretrained(output_model_path)
            
            logger.info(f"✅ {method} quantization completed")
            return quantized_model, output_model_path
            
        except Exception as e:
            logger.error(f"❌ {method} quantization failed: {e}")
            raise
    
    def load_evaluation_set(self) -> List[Dict]:
        """Load emotional evaluation prompts"""
        eval_file = Path("quant_pass1/emotional_eval_set.jsonl")
        
        try:
            with open(eval_file, 'r', encoding='utf-8') as f:
                eval_set = [json.loads(line) for line in f]
            
            logger.info(f"📋 Loaded {len(eval_set)} evaluation prompts")
            return eval_set
            
        except FileNotFoundError:
            logger.error("❌ Emotional evaluation set not found!")
            return []
    
    def generate_response(self, model, tokenizer, prompt: str, max_length: int = 200) -> str:
        """Generate response from model"""
        try:
            # Create pipeline for text generation
            generator = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            
            # Generate response
            response = generator(
                prompt,
                max_length=max_length,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )
            
            # Extract generated text (remove the prompt)
            generated_text = response[0]['generated_text']
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):].strip()
            
            return generated_text
            
        except Exception as e:
            logger.error(f"❌ Response generation failed: {e}")
            return ""
    
    def evaluate_emotional_fidelity(self, model, tokenizer, eval_set: List[Dict]) -> Dict:
        """Evaluate emotional fidelity of quantized model"""
        from emotion_tracker import EmotionTracker
        
        logger.info("🧪 Evaluating emotional fidelity...")
        
        tracker = EmotionTracker()
        results = {
            "total_prompts": len(eval_set),
            "successful_generations": 0,
            "avg_emotion_score": 0.0,
            "avg_sentiment_score": 0.0,
            "avg_metaphor_density": 0.0,
            "responses": []
        }
        
        for i, prompt_data in enumerate(eval_set):
            try:
                prompt = prompt_data["prompt"]
                expected_category = prompt_data.get("category", "general")
                
                # Generate response
                response = self.generate_response(model, tokenizer, prompt)
                
                if response:
                    # Analyze emotional content
                    emotion_metrics = tracker.analyze_emotional_content(response)
                    
                    results["responses"].append({
                        "prompt": prompt,
                        "response": response,
                        "category": expected_category,
                        "metrics": emotion_metrics
                    })
                    
                    results["successful_generations"] += 1
                    results["avg_emotion_score"] += emotion_metrics.get("emotion_score", 0)
                    results["avg_sentiment_score"] += emotion_metrics.get("sentiment_score", 0)
                    results["avg_metaphor_density"] += emotion_metrics.get("metaphor_density", 0)
                
                # Progress logging
                if (i + 1) % 10 == 0:
                    logger.info(f"📊 Evaluated {i + 1}/{len(eval_set)} prompts")
                    
            except Exception as e:
                logger.error(f"❌ Evaluation failed for prompt {i}: {e}")
        
        # Calculate averages
        if results["successful_generations"] > 0:
            results["avg_emotion_score"] /= results["successful_generations"]
            results["avg_sentiment_score"] /= results["successful_generations"]
            results["avg_metaphor_density"] /= results["successful_generations"]
        
        logger.info(f"📈 Evaluation complete: {results['successful_generations']}/{results['total_prompts']} successful")
        return results
    
    def calculate_degradation(self, baseline_metrics: Dict, current_metrics: Dict) -> float:
        """Calculate emotional degradation percentage"""
        baseline_score = (
            baseline_metrics["avg_emotion_score"] * 0.4 +
            baseline_metrics["avg_sentiment_score"] * 0.4 +
            baseline_metrics["avg_metaphor_density"] * 0.2
        )
        
        current_score = (
            current_metrics["avg_emotion_score"] * 0.4 +
            current_metrics["avg_sentiment_score"] * 0.4 +
            current_metrics["avg_metaphor_density"] * 0.2
        )
        
        if baseline_score == 0:
            return 0.0
        
        degradation = (baseline_score - current_score) / baseline_score
        return max(0.0, degradation)  # Ensure non-negative
    
    def run_quantization_pass(self) -> Dict:
        """Run complete quantization pass"""
        logger.info(f"🚀 Starting quantization pass #{self.current_attempt + 1}")
        
        # Load evaluation set
        eval_set = self.load_evaluation_set()
        if not eval_set:
            raise ValueError("No evaluation set available")
        
        # Load baseline model if needed
        if self.baseline_metrics is None:
            logger.info("📊 Establishing baseline metrics...")
            baseline_model, tokenizer = self.load_baseline_model()
            self.baseline_metrics = self.evaluate_emotional_fidelity(baseline_model, tokenizer, eval_set)
            
            # Save baseline metrics
            with open(f"{self.config.output_path}/metrics/baseline_metrics.json", 'w') as f:
                json.dump(self.baseline_metrics, f, indent=2)
            
            # Clear baseline model from memory
            del baseline_model
            torch.cuda.empty_cache()
        
        results = {}
        
        for method in self.config.quant_methods:
            try:
                start_time = time.time()
                
                # Load fresh model for quantization
                model, tokenizer = self.load_baseline_model()
                
                # Apply quantization
                quantized_model, model_path = self.apply_quantization(method, model, tokenizer)
                
                # Check model size
                model_size = self.get_model_size_gb(model_path)
                
                # Evaluate emotional fidelity
                quant_metrics = self.evaluate_emotional_fidelity(quantized_model, tokenizer, eval_set)
                
                # Calculate degradation
                degradation = self.calculate_degradation(self.baseline_metrics, quant_metrics)
                
                # Store results
                result = {
                    "method": method,
                    "attempt": self.current_attempt,
                    "model_size_gb": model_size,
                    "emotional_degradation": degradation,
                    "meets_size_target": model_size <= self.config.target_size_gb,
                    "meets_quality_target": degradation <= self.config.emotion_threshold,
                    "processing_time": time.time() - start_time,
                    "metrics": quant_metrics
                }
                
                results[method] = result
                
                # Log results
                logger.info(f"📊 {method} Results:")
                logger.info(f"   📦 Size: {model_size:.2f}GB (target: {self.config.target_size_gb}GB)")
                logger.info(f"   💔 Degradation: {degradation*100:.2f}% (max: {self.config.emotion_threshold*100}%)")
                logger.info(f"   ✅ Meets targets: Size={result['meets_size_target']}, Quality={result['meets_quality_target']}")
                
                # Save individual result
                with open(f"{self.config.output_path}/metrics/{method}_attempt_{self.current_attempt}.json", 'w') as f:
                    json.dump(result, f, indent=2)
                
                # Clean up memory
                del model, quantized_model
                torch.cuda.empty_cache()
                
            except Exception as e:
                logger.error(f"❌ {method} quantization failed: {e}")
                results[method] = {
                    "method": method,
                    "attempt": self.current_attempt,
                    "error": str(e),
                    "meets_size_target": False,
                    "meets_quality_target": False
                }
        
        self.current_attempt += 1
        self.results_log.append(results)
        
        return results
    
    def find_best_result(self, results: Dict) -> Optional[Dict]:
        """Find best quantization result that meets both targets"""
        best_result = None
        best_score = float('inf')
        
        for method, result in results.items():
            if result.get("meets_size_target") and result.get("meets_quality_target"):
                # Score based on size and quality (smaller size and lower degradation is better)
                score = result["model_size_gb"] + (result["emotional_degradation"] * 100)
                
                if score < best_score:
                    best_score = score
                    best_result = result
        
        return best_result
    
    def save_final_report(self, final_result: Dict):
        """Save comprehensive final report"""
        report = {
            "quantization_config": {
                "model_path": self.config.model_path,
                "target_size_gb": self.config.target_size_gb,
                "emotion_threshold": self.config.emotion_threshold,
                "methods_tested": self.config.quant_methods
            },
            "baseline_metrics": self.baseline_metrics,
            "final_result": final_result,
            "all_attempts": self.results_log,
            "total_attempts": self.current_attempt,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(f"{self.config.output_path}/final_quantization_report.json", 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"📄 Final report saved to {self.config.output_path}/final_quantization_report.json")

def main():
    """Main execution function"""
    
    # Load environment variables
    model_path = os.getenv("SEED_MODEL_PATH", "meta-llama/Llama-2-13b-chat-hf")
    emotion_threshold = float(os.getenv("EMOTION_THRESHOLD", "0.07"))
    size_target_gb = float(os.getenv("SIZE_TARGET_GB", "24.0"))
    
    # Create configuration
    config = QuantizationConfig(
        model_path=model_path,
        output_path="quant_pass1",
        target_size_gb=size_target_gb,
        emotion_threshold=emotion_threshold
    )
    
    # Initialize quantizer
    quantizer = EmotionalQuantizer(config)
    
    logger.info("🧠 Starting Emotional Quantization Pass 1")
    logger.info("=" * 60)
    
    try:
        # Run quantization pass
        results = quantizer.run_quantization_pass()
        
        # Find best result
        best_result = quantizer.find_best_result(results)
        
        if best_result:
            logger.info("🎉 SUCCESS! Found quantization that meets both targets:")
            logger.info(f"   🔧 Method: {best_result['method']}")
            logger.info(f"   📦 Size: {best_result['model_size_gb']:.2f}GB")
            logger.info(f"   💔 Degradation: {best_result['emotional_degradation']*100:.2f}%")
            
            quantizer.save_final_report(best_result)
            
        else:
            logger.warning("⚠️ No quantization met both size and quality targets")
            logger.info("📊 Results summary:")
            for method, result in results.items():
                if "error" not in result:
                    logger.info(f"   {method}: {result['model_size_gb']:.2f}GB, {result['emotional_degradation']*100:.2f}% degradation")
            
            # Save report anyway
            quantizer.save_final_report(results)
    
    except Exception as e:
        logger.error(f"❌ Quantization pass failed: {e}")
        raise
    
    logger.info("🏁 Quantization Pass 1 Complete")

if __name__ == "__main__":
    main()

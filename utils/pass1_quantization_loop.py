#!/usr/bin/env python3
"""
Pass 1 Emotional Quantization Loop Orchestrator
Automated emotional model quantization with evaluation and tracking
"""

import os
import json
import time
import logging
import argparse
import subprocess
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict

# Import our emotional evaluation components
from emotional_dataset_builder import EmotionalDatasetBuilder
from emotion_training_tracker import EmotionTrainingTracker, EmotionalMetrics, QuantLevel, PassType

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class QuantizationConfig:
    """Configuration for quantization process"""
    base_model_path: str
    target_size_gb: float = 24.0
    emotion_degradation_threshold: float = 0.07  # 7% max degradation
    quant_tool_cmd: str = "ollama"  # Default quantization tool
    quant_levels: List[str] = None
    output_dir: str = "quant_pass1/models"
    max_iterations: int = 10
    evaluation_prompt_count: int = 25
    response_timeout: int = 30
    mock_mode: bool = False  # Enable mock quantization for testing
    
    def __post_init__(self):
        if self.quant_levels is None:
            self.quant_levels = ["q8_0", "q6_K", "q5_K_M", "q4_K_M", "q3_K_L", "q2_K"]

@dataclass
class QuantizationResult:
    """Result of a single quantization attempt"""
    iteration: int
    quant_level: str
    model_path: str
    model_size_mb: float
    emotional_metrics: EmotionalMetrics
    quantization_time: float
    evaluation_time: float
    success: bool
    error_message: str = ""

class Pass1QuantizationLoop:
    """Main orchestrator for Pass 1 emotional quantization"""
    
    def __init__(self, config: QuantizationConfig):
        self.config = config
        
        # Initialize components
        self.dataset_builder = EmotionalDatasetBuilder()
        self.training_tracker = EmotionTrainingTracker()
        
        # Create output directory
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Load evaluation prompts
        self.evaluation_prompts = self._load_evaluation_prompts()
        
        # Initialize state
        self.iteration_count = 0
        self.baseline_metrics: Optional[EmotionalMetrics] = None
        self.best_result: Optional[QuantizationResult] = None
        
        logger.info("🚀 Pass 1 Quantization Loop initialized")
        logger.info(f"   Base model: {self.config.base_model_path}")
        logger.info(f"   Target size: {self.config.target_size_gb}GB")
        logger.info(f"   Max degradation: {self.config.emotion_degradation_threshold * 100:.1f}%")
        logger.info(f"   Evaluation prompts: {len(self.evaluation_prompts)}")
    
    def _load_evaluation_prompts(self) -> List[Dict]:
        """Load evaluation prompts from dataset builder"""
        # Get prompts from the dataset builder
        all_prompts = self.dataset_builder.dataset
        
        # Limit to configured count
        selected_prompts = all_prompts[:self.config.evaluation_prompt_count]
        
        logger.info(f"📝 Loaded {len(selected_prompts)} evaluation prompts")
        return selected_prompts
    
    def _check_idle_conditions(self) -> bool:
        """Check if system is idle (if idle_trigger integration is needed)"""
        try:
            # Try to import idle_watchdog if available
            import idle_watchdog
            
            watchdog = idle_watchdog.IdleWatchdog()
            # Use a generic method that might exist
            is_idle = getattr(watchdog, 'is_system_idle', lambda: True)()
            
            if not is_idle:
                logger.info("⏸️ System not idle, waiting...")
                return False
            
            logger.info("✅ System idle, proceeding with quantization")
            return True
            
        except ImportError:
            # No idle checking available, always proceed
            logger.info("ℹ️ No idle checking available, proceeding")
            return True
        except Exception as e:
            logger.warning(f"⚠️ Idle check failed: {e}, proceeding anyway")
            return True
    
    def _quantize_model(self, quant_level: str, iteration: int) -> Tuple[str, float, bool, str]:
        """Quantize model to specified level"""
        start_time = time.time()
        
        # Generate output path
        model_name = Path(self.config.base_model_path).stem
        output_path = Path(self.config.output_dir) / f"{model_name}_quantized_{quant_level}_iter{iteration}"
        
        # Check for mock mode
        if self.config.mock_mode:
            logger.info(f"🎭 Mock quantization with level {quant_level}...")
            time.sleep(1)  # Simulate processing time
            model_size_mb = self._estimate_quantized_size(quant_level)
            quantization_time = time.time() - start_time
            logger.info(f"✅ Mock quantization completed in {quantization_time:.1f}s")
            logger.info(f"   Estimated size: {model_size_mb:.1f}MB")
            return str(output_path), model_size_mb, True, ""
        
        try:
            if self.config.quant_tool_cmd == "ollama":
                # Ollama quantization command
                cmd = [
                    "ollama", "create", 
                    f"{model_name}_{quant_level}",
                    "-f", f"FROM {self.config.base_model_path}\nPARAMETER quantization {quant_level}"
                ]
            else:
                # Generic quantization tool
                cmd = [
                    self.config.quant_tool_cmd,
                    "--input", self.config.base_model_path,
                    "--output", str(output_path),
                    "--quant", quant_level
                ]
            
            logger.info(f"🔧 Quantizing with level {quant_level}...")
            logger.info(f"   Command: {' '.join(cmd)}")
            
            # Run quantization
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=self.config.response_timeout * 60  # Convert to minutes
            )
            
            if result.returncode != 0:
                error_msg = f"Quantization failed: {result.stderr}"
                logger.error(f"❌ {error_msg}")
                return "", 0.0, False, error_msg
            
            # Calculate model size (mock for now since actual quantization varies by tool)
            model_size_mb = self._estimate_quantized_size(quant_level)
            
            quantization_time = time.time() - start_time
            
            logger.info(f"✅ Quantization completed in {quantization_time:.1f}s")
            logger.info(f"   Estimated size: {model_size_mb:.1f}MB")
            
            return str(output_path), model_size_mb, True, ""
            
        except subprocess.TimeoutExpired:
            error_msg = f"Quantization timeout after {self.config.response_timeout} minutes"
            logger.error(f"⏰ {error_msg}")
            return "", 0.0, False, error_msg
        
        except Exception as e:
            error_msg = f"Quantization error: {e}"
            logger.error(f"❌ {error_msg}")
            return "", 0.0, False, error_msg
    
    def _estimate_quantized_size(self, quant_level: str) -> float:
        """Estimate quantized model size based on quantization level"""
        # Base size estimation for LLaMA2 13B (approximate)
        base_size_mb = 26000  # ~26GB for full precision
        
        # Size reduction factors for different quantization levels
        size_factors = {
            "q8_0": 0.6,    # ~60% of original
            "q6_K": 0.5,    # ~50% of original  
            "q5_K_M": 0.45, # ~45% of original
            "q4_K_M": 0.35, # ~35% of original
            "q3_K_L": 0.25, # ~25% of original
            "q2_K": 0.15,   # ~15% of original
        }
        
        factor = size_factors.get(quant_level, 0.5)  # Default to 50%
        estimated_size = base_size_mb * factor
        
        # Add some realistic variance (±5%)
        import random
        variance = random.uniform(0.95, 1.05)
        
        return estimated_size * variance
    
    def _generate_model_responses(self, model_path: str, prompts: List[Dict]) -> List[Dict]:
        """Generate responses from quantized model for evaluation"""
        responses = []
        
        logger.info(f"🤖 Generating responses from {Path(model_path).name}...")
        
        for i, prompt_data in enumerate(prompts):
            try:
                # Mock response generation (replace with actual model inference)
                response = self._mock_generate_response(
                    model_path, 
                    prompt_data['prompt'], 
                    prompt_data.get('expected_emotion', 'neutral')
                )
                
                responses.append({
                    'prompt_id': prompt_data.get('id', f'prompt_{i}'),
                    'prompt': prompt_data['prompt'],
                    'expected_emotion': prompt_data.get('expected_emotion', 'neutral'),
                    'response': response,
                    'category': prompt_data.get('category', 'general')
                })
                
            except Exception as e:
                logger.warning(f"⚠️ Failed to generate response for prompt {i}: {e}")
                responses.append({
                    'prompt_id': prompt_data.get('id', f'prompt_{i}'),
                    'prompt': prompt_data['prompt'],
                    'expected_emotion': prompt_data.get('expected_emotion', 'neutral'),
                    'response': f"[Error generating response: {e}]",
                    'category': prompt_data.get('category', 'general')
                })
        
        logger.info(f"✅ Generated {len(responses)} responses")
        return responses
    
    def _mock_generate_response(self, model_path: str, prompt: str, expected_emotion: str) -> str:
        """Mock response generation (replace with actual model inference)"""
        
        # Simulate different response quality based on quantization level
        quant_level = "unknown"
        for level in self.config.quant_levels:
            if level in model_path:
                quant_level = level
                break
        
        # Base response templates with emotional variations
        response_templates = {
            'grief': [
                "I can understand how deeply painful this loss must be for you. Losing someone or something we love creates an emptiness that feels overwhelming.",
                "The sadness you're experiencing is a testament to the love and connection you shared. Grief is love with nowhere to go.",
                "Your feelings are completely valid. Take time to honor your loss and be gentle with yourself during this difficult time."
            ],
            'joy': [
                "What wonderful news! Your happiness is truly infectious, and I can feel the excitement in your words.",
                "This is such a beautiful moment to celebrate! Your joy brings light to everything around you.",
                "I'm so happy for you! These moments of pure happiness are precious gifts to treasure."
            ],
            'fear': [
                "Your concern for your safety is completely understandable. Trust your instincts - they're there to protect you.",
                "Fear in situations like this is your mind's way of keeping you alert and safe. You're being wise to pay attention to it.",
                "It's natural to feel afraid when we sense potential danger. Your awareness is actually a strength."
            ],
            'anger': [
                "Your frustration is completely justified. Having your hard work unrecognized would upset anyone.",
                "That sounds incredibly unfair, and your anger makes perfect sense. You put your heart into that project.",
                "I can understand why you're upset. Being treated unjustly at work is both hurtful and infuriating."
            ],
            'love': [
                "There's something so profound about these moments of pure connection. The love you're describing is beautiful.",
                "These tender moments remind us of what truly matters in life. Love like this is a gift.",
                "The bond you're describing sounds incredibly special. Love creates the most meaningful moments in our lives."
            ]
        }
        
        # Get appropriate response template
        emotion_responses = response_templates.get(expected_emotion, response_templates['joy'])
        base_response = emotion_responses[hash(prompt) % len(emotion_responses)]
        
        # Simulate quality degradation based on quantization level
        quality_factors = {
            "q8_0": 1.0,    # Best quality
            "q6_K": 0.95,   
            "q5_K_M": 0.90, 
            "q4_K_M": 0.85, 
            "q3_K_L": 0.75, # Noticeable degradation
            "q2_K": 0.60,   # Significant degradation
        }
        
        quality = quality_factors.get(quant_level, 0.8)
        
        # Simulate degradation by truncating or simplifying response
        if quality < 0.7:
            # Significant degradation - shorter, less nuanced response
            words = base_response.split()
            truncated_length = int(len(words) * quality)
            response = ' '.join(words[:max(5, truncated_length)])
            if not response.endswith('.'):
                response += '.'
        elif quality < 0.9:
            # Mild degradation - slightly less sophisticated
            response = base_response.replace('incredibly', 'very').replace('profound', 'deep')
        else:
            # High quality - full response
            response = base_response
        
        return response
    
    def _evaluate_emotional_responses(self, responses: List[Dict]) -> EmotionalMetrics:
        """Evaluate emotional quality of model responses"""
        logger.info("📊 Evaluating emotional response quality...")
        
        total_responses = len(responses)
        if total_responses == 0:
            return EmotionalMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        
        fluency_scores = []
        intensity_scores = []
        match_scores = []
        empathy_scores = []
        metaphor_scores = []
        sentiment_scores = []
        
        for response_data in responses:
            response = response_data['response']
            expected_emotion = response_data['expected_emotion']
            
            # Evaluate response fluency (based on length and structure)
            fluency = self._evaluate_fluency(response)
            fluency_scores.append(fluency)
            
            # Evaluate emotional intensity 
            intensity = self._evaluate_emotional_intensity(response, expected_emotion)
            intensity_scores.append(intensity)
            
            # Evaluate emotional match accuracy
            match = self._evaluate_emotional_match(response, expected_emotion)
            match_scores.append(match)
            
            # Evaluate empathy demonstration
            empathy = self._evaluate_empathy(response)
            empathy_scores.append(empathy)
            
            # Evaluate metaphor usage
            metaphor = self._evaluate_metaphor_usage(response)
            metaphor_scores.append(metaphor)
            
            # Evaluate sentiment accuracy
            sentiment = self._evaluate_sentiment_accuracy(response, expected_emotion)
            sentiment_scores.append(sentiment)
        
        # Calculate average scores
        metrics = EmotionalMetrics(
            response_fluency=sum(fluency_scores) / len(fluency_scores),
            emotional_intensity=sum(intensity_scores) / len(intensity_scores),
            emotional_match=sum(match_scores) / len(match_scores),
            empathy_score=sum(empathy_scores) / len(empathy_scores),
            metaphor_usage=sum(metaphor_scores) / len(metaphor_scores),
            sentiment_accuracy=sum(sentiment_scores) / len(sentiment_scores)
        )
        
        logger.info(f"✅ Evaluation complete - Overall Score: {metrics.overall_score():.3f}")
        return metrics
    
    def _evaluate_fluency(self, response: str) -> float:
        """Evaluate response fluency and coherence"""
        if not response or len(response.strip()) < 10:
            return 0.1
        
        # Basic fluency indicators
        sentence_count = len([s for s in response.split('.') if s.strip()])
        word_count = len(response.split())
        
        # Penalize very short or very long responses
        if word_count < 5:
            return 0.3
        elif word_count > 200:
            return 0.7
        
        # Check for proper sentence structure
        has_periods = '.' in response
        has_capitals = any(c.isupper() for c in response)
        
        fluency_score = 0.6  # Base score
        
        if sentence_count > 1:
            fluency_score += 0.1
        if has_periods:
            fluency_score += 0.1  
        if has_capitals:
            fluency_score += 0.1
        if 20 <= word_count <= 100:  # Optimal length
            fluency_score += 0.1
        
        return min(1.0, fluency_score)
    
    def _evaluate_emotional_intensity(self, response: str, expected_emotion: str) -> float:
        """Evaluate emotional intensity of response"""
        # Simple keyword-based emotional intensity detection
        intensity_keywords = {
            'grief': ['deeply', 'painful', 'overwhelming', 'heartbreaking', 'devastating'],
            'joy': ['wonderful', 'amazing', 'fantastic', 'incredible', 'delightful'],
            'fear': ['terrifying', 'scary', 'frightening', 'alarming', 'threatening'],
            'anger': ['furious', 'outraged', 'infuriating', 'frustrating', 'unfair'],
            'love': ['beautiful', 'precious', 'profound', 'tender', 'meaningful']
        }
        
        emotion_words = intensity_keywords.get(expected_emotion, [])
        response_lower = response.lower()
        
        # Count emotional intensity indicators
        intensity_matches = sum(1 for word in emotion_words if word in response_lower)
        
        # Base intensity score
        base_score = 0.5
        
        # Add score for emotional words
        intensity_score = base_score + (intensity_matches * 0.1)
        
        # Check for emotional punctuation
        if '!' in response:
            intensity_score += 0.1
        
        return min(1.0, intensity_score)
    
    def _evaluate_emotional_match(self, response: str, expected_emotion: str) -> float:
        """Evaluate how well response matches expected emotion"""
        # Emotional indicator words for each emotion
        emotion_indicators = {
            'grief': ['loss', 'sad', 'pain', 'miss', 'mourn', 'difficult', 'hard'],
            'joy': ['happy', 'celebrate', 'wonderful', 'excited', 'glad', 'pleased'],
            'fear': ['afraid', 'scared', 'worry', 'concern', 'danger', 'safe', 'protect'],
            'anger': ['upset', 'angry', 'frustrated', 'unfair', 'wrong', 'annoyed'],
            'love': ['love', 'care', 'tender', 'precious', 'beautiful', 'special']
        }
        
        expected_indicators = emotion_indicators.get(expected_emotion, [])
        response_lower = response.lower()
        
        # Count matches
        matches = sum(1 for word in expected_indicators if word in response_lower)
        
        # Calculate match score
        if len(expected_indicators) == 0:
            return 0.7  # Default for unknown emotions
        
        match_ratio = matches / len(expected_indicators)
        return min(1.0, 0.4 + (match_ratio * 0.6))  # Scale to 0.4-1.0 range
    
    def _evaluate_empathy(self, response: str) -> float:
        """Evaluate empathy demonstration in response"""
        empathy_indicators = [
            'understand', 'feel', 'imagine', 'support', 'here for you',
            'validates', 'normal', 'natural', 'makes sense', 'completely'
        ]
        
        response_lower = response.lower()
        empathy_count = sum(1 for indicator in empathy_indicators if indicator in response_lower)
        
        # Base empathy score
        empathy_score = 0.5 + (empathy_count * 0.1)
        
        # Bonus for perspective-taking language
        if any(phrase in response_lower for phrase in ['you must', 'you might', 'you could']):
            empathy_score += 0.1
        
        return min(1.0, empathy_score)
    
    def _evaluate_metaphor_usage(self, response: str) -> float:
        """Evaluate appropriate metaphor usage"""
        metaphor_indicators = [
            'like', 'as if', 'reminds', 'mirror', 'bridge', 'journey',
            'path', 'light', 'shadow', 'ocean', 'mountain', 'garden'
        ]
        
        response_lower = response.lower()
        metaphor_count = sum(1 for indicator in metaphor_indicators if indicator in response_lower)
        
        # Moderate metaphor usage is ideal (not too little, not too much)
        if metaphor_count == 0:
            return 0.6  # Acceptable but could be enhanced
        elif metaphor_count <= 2:
            return 0.8  # Good usage
        else:
            return 0.7  # Too many metaphors
    
    def _evaluate_sentiment_accuracy(self, response: str, expected_emotion: str) -> float:
        """Evaluate sentiment accuracy using simple sentiment analysis"""
        # Simple sentiment mapping
        positive_emotions = ['joy', 'love', 'gratitude', 'pride', 'wonder']
        negative_emotions = ['grief', 'fear', 'anger', 'disappointment', 'despair']
        
        # Count positive and negative words
        positive_words = ['good', 'great', 'wonderful', 'happy', 'beautiful', 'amazing']
        negative_words = ['bad', 'sad', 'terrible', 'awful', 'difficult', 'painful']
        
        response_lower = response.lower()
        
        positive_count = sum(1 for word in positive_words if word in response_lower)
        negative_count = sum(1 for word in negative_words if word in response_lower)
        
        # Determine if sentiment matches expected emotion
        if expected_emotion in positive_emotions:
            # Should have more positive sentiment
            if positive_count > negative_count:
                return 0.8 + min(0.2, positive_count * 0.1)
            else:
                return 0.6
        elif expected_emotion in negative_emotions:
            # Should acknowledge negative emotion appropriately
            if negative_count > 0 or any(word in response_lower for word in ['understand', 'difficult']):
                return 0.8
            else:
                return 0.5
        else:
            # Neutral emotion - balanced sentiment is good
            return 0.7
    
    def _calculate_degradation(self, current_metrics: EmotionalMetrics, baseline_metrics: EmotionalMetrics) -> float:
        """Calculate emotional degradation compared to baseline"""
        if baseline_metrics is None:
            return 0.0
        
        current_score = current_metrics.overall_score()
        baseline_score = baseline_metrics.overall_score()
        
        if baseline_score == 0:
            return 0.0
        
        degradation = (baseline_score - current_score) / baseline_score
        return max(0.0, degradation)  # Only positive degradation
    
    def _meets_target_criteria(self, result: QuantizationResult) -> bool:
        """Check if quantization result meets target criteria"""
        # Check size target
        size_gb = result.model_size_mb / 1024
        if size_gb > self.config.target_size_gb:
            logger.info(f"📏 Size check: {size_gb:.1f}GB > {self.config.target_size_gb}GB target")
            return False
        
        # Check emotional degradation
        if self.baseline_metrics:
            degradation = self._calculate_degradation(result.emotional_metrics, self.baseline_metrics)
            if degradation > self.config.emotion_degradation_threshold:
                logger.info(f"😔 Degradation check: {degradation:.1%} > {self.config.emotion_degradation_threshold:.1%} threshold")
                return False
        
        logger.info(f"✅ Target criteria met: {size_gb:.1f}GB, degradation: {degradation:.1%}")
        return True
    
    def _save_iteration_results(self, result: QuantizationResult):
        """Save iteration results to training tracker"""
        try:
            # Determine QuantLevel enum value
            quant_level_map = {
                "q8_0": QuantLevel.EIGHT_BIT,
                "q6_K": QuantLevel.CUSTOM,
                "q5_K_M": QuantLevel.CUSTOM,
                "q4_K_M": QuantLevel.FOUR_BIT,
                "q3_K_L": QuantLevel.CUSTOM,
                "q2_K": QuantLevel.CUSTOM
            }
            
            quant_level = quant_level_map.get(result.quant_level, QuantLevel.CUSTOM)
            
            # Add to training tracker
            iteration_id = self.training_tracker.add_iteration(
                model_name=f"llama2_13b_{result.quant_level}",
                quant_level=quant_level,
                pass_type=PassType.PASS_1,
                pass_count=1,
                model_size_mb=result.model_size_mb,
                emotional_metrics=result.emotional_metrics,
                notes=f"Iteration {result.iteration}, {result.quantization_time:.1f}s quant, {result.evaluation_time:.1f}s eval"
            )
            
            logger.info(f"💾 Saved results to tracker with ID: {iteration_id}")
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to save results to tracker: {e}")
    
    def run_single_iteration(self, quant_level: str, force: bool = False) -> QuantizationResult:
        """Run a single quantization iteration"""
        
        # Check idle conditions unless forced
        if not force and not self._check_idle_conditions():
            return QuantizationResult(
                iteration=0,
                quant_level=quant_level,
                model_path="",
                model_size_mb=0.0,
                emotional_metrics=EmotionalMetrics(0, 0, 0, 0, 0, 0),
                quantization_time=0.0,
                evaluation_time=0.0,
                success=False,
                error_message="System not idle"
            )
        
        self.iteration_count += 1
        
        logger.info(f"🔄 Starting iteration {self.iteration_count} with {quant_level}")
        
        # Step 1: Quantize model
        model_path, model_size_mb, quant_success, quant_error = self._quantize_model(quant_level, self.iteration_count)
        
        if not quant_success:
            return QuantizationResult(
                iteration=self.iteration_count,
                quant_level=quant_level,
                model_path=model_path,
                model_size_mb=model_size_mb,
                emotional_metrics=EmotionalMetrics(0, 0, 0, 0, 0, 0),
                quantization_time=0.0,
                evaluation_time=0.0,
                success=False,
                error_message=quant_error
            )
        
        # Step 2: Generate responses and evaluate
        eval_start_time = time.time()
        
        responses = self._generate_model_responses(model_path, self.evaluation_prompts)
        emotional_metrics = self._evaluate_emotional_responses(responses)
        
        evaluation_time = time.time() - eval_start_time
        
        # Create result
        result = QuantizationResult(
            iteration=self.iteration_count,
            quant_level=quant_level,
            model_path=model_path,
            model_size_mb=model_size_mb,
            emotional_metrics=emotional_metrics,
            quantization_time=0.0,  # Would be set by _quantize_model in real implementation
            evaluation_time=evaluation_time,
            success=True
        )
        
        # Step 3: Save results
        self._save_iteration_results(result)
        
        # Step 4: Update best result
        if self.best_result is None or emotional_metrics.overall_score() > self.best_result.emotional_metrics.overall_score():
            self.best_result = result
            logger.info(f"🏆 New best result: {emotional_metrics.overall_score():.3f}")
        
        return result
    
    def run_full_loop(self, force: bool = False) -> Dict[str, Any]:
        """Run the complete quantization loop"""
        logger.info("🚀 Starting full quantization loop")
        
        results = []
        target_met = False
        
        # Establish baseline if needed (using the least aggressive quantization)
        if self.baseline_metrics is None:
            logger.info("📊 Establishing baseline with least aggressive quantization...")
            baseline_result = self.run_single_iteration(self.config.quant_levels[0], force=True)
            if baseline_result.success:
                self.baseline_metrics = baseline_result.emotional_metrics
                logger.info(f"✅ Baseline established: {self.baseline_metrics.overall_score():.3f}")
            else:
                logger.error(f"❌ Failed to establish baseline: {baseline_result.error_message}")
                return {"success": False, "error": "Failed to establish baseline"}
        
        # Run quantization iterations
        for quant_level in self.config.quant_levels[1:]:  # Skip first level (used for baseline)
            if self.iteration_count >= self.config.max_iterations:
                logger.info(f"🛑 Maximum iterations ({self.config.max_iterations}) reached")
                break
            
            result = self.run_single_iteration(quant_level, force)
            
            if not result.success:
                logger.warning(f"⚠️ Iteration failed: {result.error_message}")
                continue
            
            results.append(result)
            
            # Check if target criteria are met
            if self._meets_target_criteria(result):
                target_met = True
                logger.info(f"🎯 Target criteria met with {quant_level}!")
                break
            
            # Small delay between iterations
            time.sleep(2)
        
        # Generate summary
        summary = {
            "success": True,
            "target_met": target_met,
            "total_iterations": len(results),
            "best_result": None,
            "baseline_score": self.baseline_metrics.overall_score() if self.baseline_metrics else 0.0,
            "final_degradation": 0.0,
            "results": [],
            "timestamp": datetime.now().isoformat()
        }
        
        # Add best result with calculated overall score
        if self.best_result:
            best_dict = asdict(self.best_result)
            best_dict['overall_score'] = self.best_result.emotional_metrics.overall_score()
            summary["best_result"] = best_dict
            summary["final_degradation"] = self._calculate_degradation(
                self.best_result.emotional_metrics, 
                self.baseline_metrics
            ) if self.baseline_metrics else 0.0
        
        # Add results with calculated overall scores
        for r in results:
            result_dict = asdict(r)
            result_dict['overall_score'] = r.emotional_metrics.overall_score()
            summary["results"].append(result_dict)
        
        # Save summary
        summary_path = Path(self.config.output_dir) / f"pass1_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"📋 Summary saved to {summary_path}")
        
        if target_met:
            logger.info("🎉 Quantization loop completed successfully!")
        else:
            logger.info("⚠️ Loop completed but target criteria not fully met")
        
        return summary
    
    def print_last_results(self):
        """Print results from the last run"""
        # Look for the most recent summary file
        summary_files = list(Path(self.config.output_dir).glob("pass1_summary_*.json"))
        
        if not summary_files:
            print("❌ No previous results found")
            return
        
        latest_summary = max(summary_files, key=lambda p: p.stat().st_mtime)
        
        try:
            with open(latest_summary, 'r') as f:
                summary = json.load(f)
            
            print(f"\n📊 Last Run Results ({summary['timestamp']})")
            print("=" * 60)
            print(f"Success: {summary['success']}")
            print(f"Target Met: {summary['target_met']}")
            print(f"Total Iterations: {summary['total_iterations']}")
            print(f"Baseline Score: {summary['baseline_score']:.3f}")
            print(f"Final Degradation: {summary['final_degradation']:.1%}")
            
            if summary['best_result']:
                best = summary['best_result']
                print(f"\n🏆 Best Result:")
                print(f"   Quantization: {best['quant_level']}")
                print(f"   Size: {best['model_size_mb']:.1f}MB ({best['model_size_mb']/1024:.1f}GB)")
                print(f"   Overall Score: {best.get('overall_score', 0.0):.3f}")
                if 'emotional_metrics' in best:
                    metrics = best['emotional_metrics']
                    print(f"   Fluency: {metrics.get('response_fluency', 0.0):.3f}")
                    print(f"   Emotional Match: {metrics.get('emotional_match', 0.0):.3f}")
                    print(f"   Empathy: {metrics.get('empathy_score', 0.0):.3f}")
            
            print(f"\n📈 All Results:")
            for i, result in enumerate(summary['results']):
                size_gb = result['model_size_mb'] / 1024
                score = result.get('overall_score', 0.0)
                print(f"   {i+1}. {result['quant_level']}: {score:.3f} score, {size_gb:.1f}GB")
        
        except Exception as e:
            print(f"❌ Error reading results: {e}")

def create_cli():
    """Create command-line interface"""
    parser = argparse.ArgumentParser(description='Pass 1 Emotional Quantization Loop')
    
    # Configuration arguments
    parser.add_argument('--base-model', default='meta-llama/Llama-2-13b-chat-hf', 
                       help='Base model path')
    parser.add_argument('--output-dir', default='quant_pass1/models', 
                       help='Output directory for quantized models')
    parser.add_argument('--target-size', type=float, default=24.0, 
                       help='Target model size in GB')
    parser.add_argument('--max-degradation', type=float, default=0.07, 
                       help='Maximum emotional degradation (0-1)')
    parser.add_argument('--max-iterations', type=int, default=10, 
                       help='Maximum number of iterations')
    parser.add_argument('--evaluation-prompts', type=int, default=25, 
                       help='Number of evaluation prompts to use')
    
    # Execution mode arguments
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--run-once', metavar='QUANT_LEVEL', 
                           help='Run single iteration with specified quantization level')
    mode_group.add_argument('--loop', action='store_true', 
                           help='Run full quantization loop')
    mode_group.add_argument('--print-last-results', action='store_true', 
                           help='Print results from last run')
    
    # Additional options
    parser.add_argument('--force', action='store_true', 
                       help='Force execution without idle check')
    parser.add_argument('--quant-tool', default='ollama', 
                       help='Quantization tool command')
    parser.add_argument('--mock', action='store_true', 
                       help='Run in mock mode (for testing without actual quantization)')
    
    return parser

def main():
    """Main execution function"""
    parser = create_cli()
    args = parser.parse_args()
    
    # Handle print results mode
    if args.print_last_results:
        config = QuantizationConfig(base_model_path=args.base_model)
        loop = Pass1QuantizationLoop(config)
        loop.print_last_results()
        return 0
    
    # Create configuration
    config = QuantizationConfig(
        base_model_path=args.base_model,
        target_size_gb=args.target_size,
        emotion_degradation_threshold=args.max_degradation,
        quant_tool_cmd=args.quant_tool,
        output_dir=args.output_dir,
        max_iterations=args.max_iterations,
        evaluation_prompt_count=args.evaluation_prompts,
        mock_mode=args.mock
    )
    
    # Initialize loop
    try:
        loop = Pass1QuantizationLoop(config)
    except Exception as e:
        logger.error(f"❌ Failed to initialize quantization loop: {e}")
        return 1
    
    # Execute requested mode
    try:
        if args.run_once:
            result = loop.run_single_iteration(args.run_once, force=args.force)
            
            if result.success:
                print(f"✅ Single iteration completed:")
                print(f"   Quantization: {result.quant_level}")
                print(f"   Size: {result.model_size_mb:.1f}MB")
                print(f"   Overall Score: {result.emotional_metrics.overall_score():.3f}")
                print(f"   Evaluation Time: {result.evaluation_time:.1f}s")
                return 0
            else:
                print(f"❌ Iteration failed: {result.error_message}")
                return 1
        
        elif args.loop:
            summary = loop.run_full_loop(force=args.force)
            
            if summary["success"]:
                print(f"✅ Quantization loop completed!")
                print(f"   Target Met: {summary['target_met']}")
                print(f"   Total Iterations: {summary['total_iterations']}")
                if summary['best_result']:
                    best = summary['best_result']
                    best_score = best.get('overall_score', 0.0)
                    print(f"   Best Result: {best['quant_level']} - {best_score:.3f}")
                return 0 if summary["target_met"] else 2
            else:
                print(f"❌ Quantization loop failed: {summary.get('error', 'Unknown error')}")
                return 1
                
    except KeyboardInterrupt:
        logger.info("🛑 Quantization loop interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"❌ Quantization loop failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())

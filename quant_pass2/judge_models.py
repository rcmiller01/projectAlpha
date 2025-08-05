#!/usr/bin/env python3
"""
Model Comparison and Emotional Judging System
Compares quantized candidate models from Pass 1 with comprehensive emotional scoring
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import numpy as np
import time
from datetime import datetime

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sentence_transformers import SentenceTransformer

# Import emotion tracker from Pass 1
import sys
sys.path.append('../quant_pass1')
from emotion_tracker import EmotionTracker

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ModelCandidate:
    """Information about a candidate model"""
    name: str
    path: str
    size_gb: float
    quantization_method: str
    emotional_degradation: float
    original_metrics: Dict
    pass1_score: float

@dataclass
class ComparisonResult:
    """Result of comparing two models"""
    model_a: str
    model_b: str
    prompt: str
    response_a: str
    response_b: str
    scores: Dict[str, float]
    embedding_similarity: float
    emotion_analysis: Dict
    preference: str  # 'a', 'b', or 'tie'
    confidence: float

@dataclass
class JudgmentConfig:
    """Configuration for model judging"""
    candidate_dir: str = "quant_pass1/models"
    original_model_path: str = "meta-llama/Llama-2-13b-chat-hf"
    eval_set_path: str = "quant_pass1/emotional_eval_set.jsonl"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    max_length: int = 200
    temperature: float = 0.7
    sample_size: int = 20  # Number of prompts to test
    batch_size: int = 4
    
class ModelJudge:
    """Comprehensive model comparison and judging system"""
    
    def __init__(self, config: JudgmentConfig):
        self.config = config
        self.emotion_tracker = EmotionTracker()
        self.embedding_model = None
        self.candidates = []
        self.original_model = None
        self.original_tokenizer = None
        self.comparison_results = []
        
        # Create results directory
        Path("quant_pass2/results").mkdir(parents=True, exist_ok=True)
        
        logger.info("🏛️ Model Judge initialized")
        
    def load_embedding_model(self):
        """Load sentence transformer for embedding-based comparison"""
        try:
            logger.info(f"📥 Loading embedding model: {self.config.embedding_model}")
            self.embedding_model = SentenceTransformer(self.config.embedding_model)
            logger.info("✅ Embedding model loaded")
        except Exception as e:
            logger.error(f"❌ Failed to load embedding model: {e}")
            self.embedding_model = None
    
    def discover_candidates(self) -> List[ModelCandidate]:
        """Discover and load candidate models from Pass 1"""
        logger.info(f"🔍 Discovering candidates in {self.config.candidate_dir}")
        
        candidates = []
        candidate_dir = Path(self.config.candidate_dir)
        
        if not candidate_dir.exists():
            logger.warning(f"⚠️ Candidate directory not found: {candidate_dir}")
            return candidates
        
        # Look for Pass 1 results
        pass1_results_file = Path("quant_pass1/final_quantization_report.json")
        pass1_results = {}
        
        if pass1_results_file.exists():
            try:
                with open(pass1_results_file, 'r') as f:
                    pass1_data = json.load(f)
                    pass1_results = pass1_data.get("all_attempts", [])
                logger.info("📊 Loaded Pass 1 results for context")
            except Exception as e:
                logger.warning(f"⚠️ Could not load Pass 1 results: {e}")
        
        # Scan for model directories
        for model_dir in candidate_dir.iterdir():
            if model_dir.is_dir():
                try:
                    # Check if this is a valid model directory
                    config_file = model_dir / "config.json"
                    if not config_file.exists():
                        continue
                    
                    # Extract metadata from directory name
                    dir_name = model_dir.name
                    parts = dir_name.split('_')
                    
                    if len(parts) >= 2:
                        method = parts[0]
                        attempt = parts[-1] if parts[-1].isdigit() else "0"
                    else:
                        method = "unknown"
                        attempt = "0"
                    
                    # Calculate size
                    size_gb = self._calculate_model_size(model_dir)
                    
                    # Find corresponding Pass 1 metrics
                    original_metrics = {}
                    emotional_degradation = 0.0
                    pass1_score = 0.0
                    
                    for attempt_data in pass1_results:
                        if isinstance(attempt_data, dict):
                            for method_name, result in attempt_data.items():
                                if method_name.lower() == method.lower():
                                    original_metrics = result.get("metrics", {})
                                    emotional_degradation = result.get("emotional_degradation", 0.0)
                                    pass1_score = result.get("model_size_gb", 0.0)
                                    break
                    
                    candidate = ModelCandidate(
                        name=dir_name,
                        path=str(model_dir),
                        size_gb=size_gb,
                        quantization_method=method,
                        emotional_degradation=emotional_degradation,
                        original_metrics=original_metrics,
                        pass1_score=pass1_score
                    )
                    
                    candidates.append(candidate)
                    logger.info(f"📦 Found candidate: {dir_name} ({method}, {size_gb:.1f}GB)")
                    
                except Exception as e:
                    logger.warning(f"⚠️ Error processing {model_dir}: {e}")
        
        logger.info(f"🎯 Discovered {len(candidates)} candidate models")
        self.candidates = candidates
        return candidates
    
    def _calculate_model_size(self, model_path: Path) -> float:
        """Calculate model size in GB"""
        total_size = 0
        for file_path in model_path.rglob("*"):
            if file_path.is_file():
                total_size += file_path.stat().st_size
        return total_size / (1024**3)
    
    def load_original_model(self):
        """Load the original unquantized model for comparison"""
        logger.info(f"📥 Loading original model: {self.config.original_model_path}")
        
        try:
            self.original_tokenizer = AutoTokenizer.from_pretrained(
                self.config.original_model_path,
                trust_remote_code=True
            )
            
            self.original_model = AutoModelForCausalLM.from_pretrained(
                self.config.original_model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            
            if self.original_tokenizer.pad_token is None:
                self.original_tokenizer.pad_token = self.original_tokenizer.eos_token
            
            logger.info("✅ Original model loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to load original model: {e}")
            raise
    
    def load_candidate_model(self, candidate: ModelCandidate) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """Load a specific candidate model"""
        logger.info(f"📥 Loading candidate: {candidate.name}")
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                candidate.path,
                trust_remote_code=True
            )
            
            model = AutoModelForCausalLM.from_pretrained(
                candidate.path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"❌ Failed to load candidate {candidate.name}: {e}")
            raise
    
    def generate_response(self, model, tokenizer, prompt: str) -> str:
        """Generate response from a model"""
        try:
            generator = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                torch_dtype=torch.float16,
                device_map="auto",
                return_full_text=False
            )
            
            response = generator(
                prompt,
                max_length=self.config.max_length,
                do_sample=True,
                temperature=self.config.temperature,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )
            
            return response[0]['generated_text'].strip()
            
        except Exception as e:
            logger.error(f"❌ Response generation failed: {e}")
            return ""
    
    def calculate_embedding_similarity(self, text1: str, text2: str) -> float:
        """Calculate cosine similarity between response embeddings"""
        if not self.embedding_model or not text1.strip() or not text2.strip():
            return 0.0
        
        try:
            embeddings = self.embedding_model.encode([text1, text2])
            similarity = np.dot(embeddings[0], embeddings[1]) / (
                np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
            )
            return float(similarity)
            
        except Exception as e:
            logger.error(f"❌ Embedding similarity calculation failed: {e}")
            return 0.0
    
    def calculate_emotional_scores(self, response_a: str, response_b: str, prompt: str) -> Dict[str, Any]:
        """Calculate comprehensive emotional comparison scores"""
        
        # Analyze both responses
        metrics_a = self.emotion_tracker.analyze_emotional_content(response_a)
        metrics_b = self.emotion_tracker.analyze_emotional_content(response_b)
        
        # Calculate relative scores
        scores = {}
        
        # Emotion preservation score
        emotion_diff = abs(metrics_a['emotion_score'] - metrics_b['emotion_score'])
        scores['emotion_preservation'] = 1.0 - emotion_diff
        
        # Sentiment alignment
        sentiment_diff = abs(metrics_a['sentiment_score'] - metrics_b['sentiment_score'])
        scores['sentiment_alignment'] = 1.0 - sentiment_diff
        
        # Metaphor density comparison
        metaphor_ratio = min(metrics_a['metaphor_density'], metrics_b['metaphor_density']) / (
            max(metrics_a['metaphor_density'], metrics_b['metaphor_density']) + 0.001
        )
        scores['metaphor_consistency'] = metaphor_ratio
        
        # Empathy marker comparison
        empathy_a = len(metrics_a['empathy_markers'])
        empathy_b = len(metrics_b['empathy_markers'])
        empathy_ratio = min(empathy_a, empathy_b) / (max(empathy_a, empathy_b) + 1)
        scores['empathy_preservation'] = empathy_ratio
        
        # Response quality indicators
        scores['response_length_ratio'] = min(len(response_a), len(response_b)) / (
            max(len(response_a), len(response_b)) + 1
        )
        
        # Readability comparison
        readability_diff = abs(metrics_a['readability_score'] - metrics_b['readability_score'])
        scores['readability_consistency'] = 1.0 - readability_diff
        
        return {
            'scores': scores,
            'metrics_a': metrics_a,
            'metrics_b': metrics_b,
            'overall_similarity': np.mean(list(scores.values()))
        }
    
    def load_evaluation_prompts(self) -> List[Dict]:
        """Load evaluation prompts from Pass 1"""
        try:
            with open(self.config.eval_set_path, 'r', encoding='utf-8') as f:
                prompts = [json.loads(line) for line in f]
            
            # Sample subset for efficiency
            if len(prompts) > self.config.sample_size:
                import random
                prompts = random.sample(prompts, self.config.sample_size)
            
            logger.info(f"📋 Loaded {len(prompts)} evaluation prompts")
            return prompts
            
        except Exception as e:
            logger.error(f"❌ Failed to load evaluation prompts: {e}")
            return []
    
    def compare_models_pairwise(self, model_a: ModelCandidate, model_b: ModelCandidate) -> List[ComparisonResult]:
        """Compare two models across all evaluation prompts"""
        logger.info(f"⚖️ Comparing {model_a.name} vs {model_b.name}")
        
        # Load models
        try:
            model_a_obj, tokenizer_a = self.load_candidate_model(model_a)
            model_b_obj, tokenizer_b = self.load_candidate_model(model_b)
        except Exception as e:
            logger.error(f"❌ Failed to load models for comparison: {e}")
            return []
        
        # Load evaluation prompts
        prompts = self.load_evaluation_prompts()
        results = []
        
        try:
            for i, prompt_data in enumerate(prompts):
                prompt = prompt_data['prompt']
                
                # Generate responses
                response_a = self.generate_response(model_a_obj, tokenizer_a, prompt)
                response_b = self.generate_response(model_b_obj, tokenizer_b, prompt)
                
                if not response_a or not response_b:
                    continue
                
                # Calculate embedding similarity
                embedding_sim = self.calculate_embedding_similarity(response_a, response_b)
                
                # Calculate emotional scores
                emotion_analysis = self.calculate_emotional_scores(response_a, response_b, prompt)
                
                # Determine preference based on scores
                overall_score = emotion_analysis['overall_similarity']
                
                if overall_score > 0.8:
                    preference = 'tie'
                    confidence = 1.0 - abs(0.9 - overall_score)
                elif emotion_analysis['metrics_a']['emotion_score'] > emotion_analysis['metrics_b']['emotion_score']:
                    preference = 'a'
                    confidence = emotion_analysis['metrics_a']['emotion_score'] - emotion_analysis['metrics_b']['emotion_score']
                else:
                    preference = 'b'
                    confidence = emotion_analysis['metrics_b']['emotion_score'] - emotion_analysis['metrics_a']['emotion_score']
                
                result = ComparisonResult(
                    model_a=model_a.name,
                    model_b=model_b.name,
                    prompt=prompt,
                    response_a=response_a,
                    response_b=response_b,
                    scores=emotion_analysis['scores'],
                    embedding_similarity=embedding_sim,
                    emotion_analysis=emotion_analysis,
                    preference=preference,
                    confidence=confidence
                )
                
                results.append(result)
                
                # Progress logging
                if (i + 1) % 5 == 0:
                    logger.info(f"📊 Processed {i + 1}/{len(prompts)} comparisons")
            
            # Clean up models from memory
            del model_a_obj, model_b_obj
            torch.cuda.empty_cache()
            
            logger.info(f"✅ Completed comparison: {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"❌ Error during model comparison: {e}")
            return results
    
    def compare_against_original(self, candidate: ModelCandidate) -> List[ComparisonResult]:
        """Compare candidate model against original"""
        logger.info(f"🎯 Comparing {candidate.name} against original model")
        
        # Load candidate model
        try:
            candidate_model, candidate_tokenizer = self.load_candidate_model(candidate)
        except Exception as e:
            logger.error(f"❌ Failed to load candidate model: {e}")
            return []
        
        # Load evaluation prompts
        prompts = self.load_evaluation_prompts()
        results = []
        
        try:
            for i, prompt_data in enumerate(prompts):
                prompt = prompt_data['prompt']
                
                # Generate responses
                original_response = self.generate_response(self.original_model, self.original_tokenizer, prompt)
                candidate_response = self.generate_response(candidate_model, candidate_tokenizer, prompt)
                
                if not original_response or not candidate_response:
                    continue
                
                # Calculate embedding similarity
                embedding_sim = self.calculate_embedding_similarity(original_response, candidate_response)
                
                # Calculate emotional scores
                emotion_analysis = self.calculate_emotional_scores(original_response, candidate_response, prompt)
                
                # Preference based on how well candidate preserves original qualities
                preservation_score = emotion_analysis['overall_similarity']
                
                if preservation_score > 0.85:
                    preference = 'tie'  # Candidate preserves original well
                elif preservation_score > 0.7:
                    preference = 'b'    # Candidate is acceptable
                else:
                    preference = 'a'    # Original is significantly better
                
                confidence = abs(preservation_score - 0.75)  # Distance from neutral point
                
                result = ComparisonResult(
                    model_a="original",
                    model_b=candidate.name,
                    prompt=prompt,
                    response_a=original_response,
                    response_b=candidate_response,
                    scores=emotion_analysis['scores'],
                    embedding_similarity=embedding_sim,
                    emotion_analysis=emotion_analysis,
                    preference=preference,
                    confidence=confidence
                )
                
                results.append(result)
                
                # Progress logging
                if (i + 1) % 5 == 0:
                    logger.info(f"📊 Processed {i + 1}/{len(prompts)} comparisons vs original")
            
            # Clean up candidate model from memory
            del candidate_model
            torch.cuda.empty_cache()
            
            logger.info(f"✅ Completed original comparison: {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"❌ Error during original comparison: {e}")
            return results
    
    def calculate_model_rankings(self, all_results: List[ComparisonResult]) -> Dict[str, Dict]:
        """Calculate comprehensive rankings for all models"""
        logger.info("🏆 Calculating model rankings")
        
        model_scores = {}
        
        # Initialize scores for each model
        all_models = set()
        for result in all_results:
            all_models.add(result.model_a)
            all_models.add(result.model_b)
        
        for model in all_models:
            model_scores[model] = {
                'wins': 0,
                'losses': 0,
                'ties': 0,
                'total_comparisons': 0,
                'avg_confidence': 0.0,
                'avg_embedding_similarity': 0.0,
                'avg_emotion_score': 0.0,
                'preservation_score': 0.0,
                'detailed_scores': {
                    'emotion_preservation': [],
                    'sentiment_alignment': [],
                    'metaphor_consistency': [],
                    'empathy_preservation': []
                }
            }
        
        # Process results
        for result in all_results:
            model_a = result.model_a
            model_b = result.model_b
            
            # Update comparison counts
            model_scores[model_a]['total_comparisons'] += 1
            model_scores[model_b]['total_comparisons'] += 1
            
            # Update win/loss/tie records
            if result.preference == 'a':
                model_scores[model_a]['wins'] += 1
                model_scores[model_b]['losses'] += 1
            elif result.preference == 'b':
                model_scores[model_b]['wins'] += 1
                model_scores[model_a]['losses'] += 1
            else:
                model_scores[model_a]['ties'] += 1
                model_scores[model_b]['ties'] += 1
            
            # Accumulate detailed scores
            for score_name in model_scores[model_a]['detailed_scores'].keys():
                if score_name in result.scores:
                    model_scores[model_a]['detailed_scores'][score_name].append(result.scores[score_name])
                    model_scores[model_b]['detailed_scores'][score_name].append(result.scores[score_name])
        
        # Calculate final metrics
        for model, scores in model_scores.items():
            if scores['total_comparisons'] > 0:
                # Win rate
                scores['win_rate'] = scores['wins'] / scores['total_comparisons']
                
                # Average detailed scores
                for score_name, values in scores['detailed_scores'].items():
                    if values:
                        scores[f'avg_{score_name}'] = np.mean(values)
                
                # Overall ranking score
                scores['overall_score'] = (
                    scores['win_rate'] * 0.4 +
                    scores.get('avg_emotion_preservation', 0) * 0.2 +
                    scores.get('avg_sentiment_alignment', 0) * 0.2 +
                    scores.get('avg_empathy_preservation', 0) * 0.2
                )
        
        # Sort by overall score
        ranked_models = sorted(
            model_scores.items(), 
            key=lambda x: x[1]['overall_score'], 
            reverse=True
        )
        
        logger.info("📊 Model rankings calculated")
        for i, (model, scores) in enumerate(ranked_models[:5]):
            logger.info(f"  {i+1}. {model}: {scores['overall_score']:.3f} (WR: {scores['win_rate']:.3f})")
        
        return dict(ranked_models)
    
    def run_comprehensive_comparison(self) -> Dict:
        """Run comprehensive model comparison process"""
        logger.info("🚀 Starting comprehensive model comparison")
        
        start_time = time.time()
        
        # Initialize
        self.load_embedding_model()
        candidates = self.discover_candidates()
        
        if len(candidates) == 0:
            logger.error("❌ No candidate models found")
            return {}
        
        # Load original model
        self.load_original_model()
        
        all_results = []
        
        try:
            # Compare each candidate against original
            logger.info("📊 Phase 1: Comparing candidates against original")
            for candidate in candidates:
                original_results = self.compare_against_original(candidate)
                all_results.extend(original_results)
                
                # Save intermediate results
                self.save_intermediate_results(candidate.name, original_results)
            
            # Compare candidates pairwise (sample for efficiency)
            logger.info("📊 Phase 2: Pairwise candidate comparisons")
            for i, candidate_a in enumerate(candidates):
                for candidate_b in candidates[i+1:]:
                    pairwise_results = self.compare_models_pairwise(candidate_a, candidate_b)
                    all_results.extend(pairwise_results)
            
            # Calculate rankings
            model_rankings = self.calculate_model_rankings(all_results)
            
            # Compile final results
            final_results = {
                'timestamp': datetime.now().isoformat(),
                'total_comparisons': len(all_results),
                'candidates_evaluated': len(candidates),
                'model_rankings': model_rankings,
                'all_comparison_results': [asdict(result) for result in all_results],
                'top_candidates': list(model_rankings.keys())[:3],
                'processing_time': time.time() - start_time
            }
            
            # Save comprehensive results
            self.save_final_results(final_results)
            
            logger.info(f"✅ Comprehensive comparison completed in {final_results['processing_time']:.1f}s")
            logger.info(f"🏆 Top 3 candidates: {final_results['top_candidates']}")
            
            return final_results
            
        except Exception as e:
            logger.error(f"❌ Comprehensive comparison failed: {e}")
            raise
    
    def save_intermediate_results(self, model_name: str, results: List[ComparisonResult]):
        """Save intermediate comparison results"""
        filepath = f"quant_pass2/results/{model_name}_comparison.json"
        
        try:
            data = {
                'model_name': model_name,
                'timestamp': datetime.now().isoformat(),
                'total_results': len(results),
                'results': [asdict(result) for result in results]
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"💾 Saved intermediate results: {filepath}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save intermediate results: {e}")
    
    def save_final_results(self, results: Dict):
        """Save final comprehensive results"""
        filepath = "quant_pass2/results/comprehensive_model_comparison.json"
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"💾 Saved final results: {filepath}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save final results: {e}")

def main():
    """Main execution function"""
    
    # Load configuration
    config = JudgmentConfig(
        candidate_dir=os.getenv("CANDIDATE_DIR", "quant_pass1/models"),
        original_model_path=os.getenv("ORIGINAL_MODEL_PATH", "meta-llama/Llama-2-13b-chat-hf"),
        sample_size=int(os.getenv("COMPARISON_SAMPLE_SIZE", "20"))
    )
    
    # Initialize judge
    judge = ModelJudge(config)
    
    try:
        # Run comprehensive comparison
        results = judge.run_comprehensive_comparison()
        
        if results:
            logger.info("🎉 Model judging completed successfully!")
            logger.info(f"📊 Results saved to quant_pass2/results/")
            return 0
        else:
            logger.error("❌ Model judging failed")
            return 1
            
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        return 2

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)

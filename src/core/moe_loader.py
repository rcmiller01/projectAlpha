"""
Mixture-of-Experts (MoE) Loader for ProjectAlpha
Dynamic loading, routing, and unloading of specialized SLiM experts
Optimized for CPU/RAM-based systems with memory constraints
"""

import time
import threading
import logging
from typing import Dict, List, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
import json
import os
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class ExpertInfo:
    """Metadata for a SLiM expert model"""
    key: str
    path: str
    size_gb: float
    domain: str
    description: str
    priority: int = 0
    dependencies: List[str] = field(default_factory=list)
    last_used: float = 0.0
    load_count: int = 0
    success_rate: float = 1.0

@dataclass
class ExpertInstance:
    """Runtime instance of a loaded expert"""
    model: Any
    info: ExpertInfo
    loaded_at: float
    last_used: float
    thread_safe: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock)

class IntentClassifier:
    """Classifies user prompts to determine relevant expert weights"""
    
    def __init__(self):
        self.domain_keywords = {
            'logic_high': ['analyze', 'reason', 'prove', 'logic', 'deduce', 'infer'],
            'logic_code': ['code', 'program', 'debug', 'function', 'algorithm', 'syntax'],
            'logic_proof': ['theorem', 'proof', 'mathematical', 'formal', 'verify'],
            'logic_fallback': ['think', 'consider', 'evaluate', 'assess'],
            'emote_valence': ['feel', 'emotion', 'mood', 'sentiment', 'empathy'],
            'emote_arousal': ['excited', 'calm', 'energetic', 'relaxed', 'intense'],
            'creative_metaphor': ['creative', 'metaphor', 'story', 'imagine', 'artistic'],
            'creative_write': ['write', 'compose', 'draft', 'author', 'narrative'],
            'planning_temporal': ['plan', 'schedule', 'organize', 'timeline', 'sequence'],
            'planning_strategic': ['strategy', 'approach', 'method', 'tactics', 'goal'],
            'ritual_symbolic': ['ritual', 'symbol', 'ceremony', 'tradition', 'meaning'],
            'memory_recall': ['remember', 'recall', 'history', 'past', 'previous']
        }
        
    def score_intent(self, prompt: str, context: Optional[Dict] = None) -> Dict[str, float]:
        """
        Score prompt against expert domains
        Returns: {expert_key: confidence_score}
        """
        prompt_lower = prompt.lower()
        scores = {}
        
        # Keyword-based scoring
        for domain, keywords in self.domain_keywords.items():
            score = sum(1 for keyword in keywords if keyword in prompt_lower)
            if score > 0:
                scores[domain] = min(score / len(keywords), 1.0)
        
        # Context-based adjustments
        if context:
            agent_type = context.get('agentType', 'general')
            if agent_type == 'deduction':
                scores['logic_high'] = scores.get('logic_high', 0) + 0.3
            elif agent_type == 'metaphor':
                scores['creative_metaphor'] = scores.get('creative_metaphor', 0) + 0.3
            elif agent_type == 'planner':
                scores['planning_temporal'] = scores.get('planning_temporal', 0) + 0.3
            elif agent_type == 'ritual':
                scores['ritual_symbolic'] = scores.get('ritual_symbolic', 0) + 0.3
        
        # Ensure fallback has minimum score
        if not scores or max(scores.values()) < 0.1:
            scores['logic_fallback'] = 0.5
            
        # Normalize scores
        total = sum(scores.values())
        if total > 0:
            scores = {k: v/total for k, v in scores.items()}
            
        return scores

class MoELoader:
    """
    Mixture-of-Experts Loader
    Manages dynamic loading/unloading of SLiM expert models
    """
    
    def __init__(self, model_registry: Dict[str, ExpertInfo], ram_limit_gb: float = 16.0):
        self.registry = {k: v for k, v in model_registry.items()}
        self.loaded: Dict[str, ExpertInstance] = {}
        self.ram_used = 0.0
        self.ram_limit = ram_limit_gb
        self.lock = threading.RLock()
        self.executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="MoE")
        self.intent_classifier = IntentClassifier()
        
        # Performance metrics
        self.stats = {
            'loads': 0,
            'unloads': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'inference_count': 0,
            'total_inference_time': 0.0
        }
        
        logger.info(f"MoELoader initialized with {len(self.registry)} experts, {ram_limit_gb}GB RAM limit")
    
    def get_expert(self, key: str) -> Optional[ExpertInstance]:
        """
        Get expert instance, loading if necessary
        Thread-safe with LRU eviction
        """
        with self.lock:
            if key not in self.registry:
                logger.warning(f"Expert '{key}' not found in registry")
                return None
                
            # Cache hit
            if key in self.loaded:
                instance = self.loaded[key]
                instance.last_used = time.time()
                instance.info.last_used = time.time()
                self.stats['cache_hits'] += 1
                return instance
            
            # Cache miss - need to load
            self.stats['cache_misses'] += 1
            return self._load_expert(key)
    
    def _load_expert(self, key: str) -> Optional[ExpertInstance]:
        """Load expert model into RAM"""
        info = self.registry[key]
        
        # Ensure RAM availability
        self._ensure_ram_for(info.size_gb)
        
        try:
            # Load model (this would integrate with your actual model loading)
            logger.info(f"Loading expert '{key}' from {info.path}")
            model = self._load_model_from_path(info.path)
            
            # Create instance
            instance = ExpertInstance(
                model=model,
                info=info,
                loaded_at=time.time(),
                last_used=time.time()
            )
            
            # Update tracking
            self.loaded[key] = instance
            self.ram_used += info.size_gb
            info.load_count += 1
            info.last_used = time.time()
            self.stats['loads'] += 1
            
            logger.info(f"Expert '{key}' loaded successfully. RAM usage: {self.ram_used:.1f}/{self.ram_limit}GB")
            return instance
            
        except Exception as e:
            logger.error(f"Failed to load expert '{key}': {e}")
            return None
    
    def _load_model_from_path(self, path: str) -> Any:
        """
        Load actual model from file path
        This is a placeholder - integrate with your model loading system
        """
        # TODO: Integrate with your actual model loading (HuggingFace, ONNX, etc.)
        logger.info(f"Loading model from {path}")
        
        # Simulated model loading
        class MockModel:
            def __init__(self, path):
                self.path = path
                self.loaded_at = time.time()
            
            def generate(self, prompt, **kwargs):
                return f"Response from {self.path} for: {prompt[:50]}..."
        
        return MockModel(path)
    
    def _ensure_ram_for(self, size_needed: float):
        """Ensure sufficient RAM by unloading LRU experts"""
        while self.ram_used + size_needed > self.ram_limit and self.loaded:
            # Find least recently used expert
            lru_key = min(self.loaded.keys(), 
                         key=lambda k: self.loaded[k].last_used)
            
            logger.info(f"RAM limit exceeded, unloading LRU expert: {lru_key}")
            self._unload_expert(lru_key)
    
    def _unload_expert(self, key: str):
        """Unload expert from RAM"""
        if key not in self.loaded:
            return
            
        instance = self.loaded[key]
        
        # Cleanup model resources
        if hasattr(instance.model, 'cleanup'):
            instance.model.cleanup()
        
        # Update tracking
        del self.loaded[key]
        self.ram_used -= instance.info.size_gb
        self.stats['unloads'] += 1
        
        logger.info(f"Expert '{key}' unloaded. RAM usage: {self.ram_used:.1f}/{self.ram_limit}GB")
    
    def run_experts(self, prompt: str, context: Optional[Dict] = None, max_experts: int = 3) -> Dict[str, Any]:
        """
        Run selected experts on prompt
        Returns: {expert_key: {response, confidence, latency}}
        """
        start_time = time.time()
        
        # Get intent weights
        intent_weights = self.intent_classifier.score_intent(prompt, context)
        
        # Select top experts
        selected_experts = sorted(intent_weights.items(), 
                                key=lambda x: x[1], 
                                reverse=True)[:max_experts]
        
        if not selected_experts:
            logger.warning("No experts selected for prompt")
            return {}
        
        logger.info(f"Selected experts: {[k for k, _ in selected_experts]}")
        
        # Run experts in parallel
        futures = {}
        results = {}
        
        for expert_key, weight in selected_experts:
            instance = self.get_expert(expert_key)
            if instance:
                future = self.executor.submit(
                    self._run_single_expert, 
                    instance, prompt, context, weight
                )
                futures[expert_key] = future
        
        # Collect results
        for expert_key, future in futures.items():
            try:
                result = future.result(timeout=30.0)  # 30s timeout
                results[expert_key] = result
            except Exception as e:
                logger.error(f"Expert '{expert_key}' failed: {e}")
                results[expert_key] = {
                    'response': None,
                    'error': str(e),
                    'confidence': 0.0,
                    'latency': 0.0
                }
        
        # Update stats
        total_time = time.time() - start_time
        self.stats['inference_count'] += 1
        self.stats['total_inference_time'] += total_time
        
        logger.info(f"MoE inference completed in {total_time:.2f}s with {len(results)} experts")
        return results
    
    def _run_single_expert(self, instance: ExpertInstance, prompt: str, 
                          context: Optional[Dict], weight: float) -> Dict[str, Any]:
        """Run inference on single expert"""
        expert_start = time.time()
        
        try:
            with instance.lock:  # Thread safety for model inference
                response = instance.model.generate(prompt, context=context)
                
            latency = time.time() - expert_start
            
            # Update success rate
            instance.info.success_rate = (instance.info.success_rate * 0.9) + (1.0 * 0.1)
            
            return {
                'response': response,
                'confidence': weight * instance.info.success_rate,
                'latency': latency,
                'expert_info': {
                    'key': instance.info.key,
                    'domain': instance.info.domain,
                    'description': instance.info.description
                }
            }
            
        except Exception as e:
            # Update failure rate
            instance.info.success_rate = (instance.info.success_rate * 0.9) + (0.0 * 0.1)
            raise e
    
    def get_loaded_experts(self) -> List[str]:
        """Get list of currently loaded expert keys"""
        with self.lock:
            return list(self.loaded.keys())
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics"""
        with self.lock:
            return {
                'used_gb': self.ram_used,
                'limit_gb': self.ram_limit,
                'usage_percent': (self.ram_used / self.ram_limit) * 100,
                'loaded_count': len(self.loaded)
            }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        avg_inference_time = (self.stats['total_inference_time'] / 
                             max(self.stats['inference_count'], 1))
        
        return {
            **self.stats,
            'avg_inference_time': avg_inference_time,
            'cache_hit_rate': (self.stats['cache_hits'] / 
                              max(self.stats['cache_hits'] + self.stats['cache_misses'], 1)),
            'memory_usage': self.get_memory_usage()
        }
    
    def preload_experts(self, expert_keys: List[str]):
        """Preload specified experts into RAM"""
        logger.info(f"Preloading experts: {expert_keys}")
        for key in expert_keys:
            self.get_expert(key)
    
    def clear_cache(self):
        """Unload all experts from RAM"""
        with self.lock:
            expert_keys = list(self.loaded.keys())
            for key in expert_keys:
                self._unload_expert(key)
        logger.info("All experts unloaded from cache")
    
    def save_stats(self, filepath: str):
        """Save performance statistics to file"""
        stats = self.get_performance_stats()
        with open(filepath, 'w') as f:
            json.dump(stats, f, indent=2)
        logger.info(f"Stats saved to {filepath}")
    
    def shutdown(self):
        """Cleanup resources"""
        logger.info("Shutting down MoELoader...")
        self.executor.shutdown(wait=True)
        self.clear_cache()
        logger.info("MoELoader shutdown complete")

# Factory function for creating model registry
def create_model_registry(models_config: Dict) -> Dict[str, ExpertInfo]:
    """
    Create model registry from configuration
    Expected config format:
    {
        "expert_key": {
            "path": "/path/to/model",
            "size_gb": 2.5,
            "domain": "logic",
            "description": "High-level logical reasoning",
            "priority": 1
        }
    }
    """
    registry = {}
    
    for key, config in models_config.items():
        expert_info = ExpertInfo(
            key=key,
            path=config['path'],
            size_gb=config['size_gb'],
            domain=config['domain'],
            description=config['description'],
            priority=config.get('priority', 0),
            dependencies=config.get('dependencies', [])
        )
        registry[key] = expert_info
    
    return registry

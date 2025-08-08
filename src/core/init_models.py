"""
Model Initialization Utility - Dynamic model loading for ProjectAlpha

This module provides utilities for dynamically loading AI models based on
environment variables with fallback defaults. Supports multiple backends
including Ollama, llama.cpp, and mock models for testing.

Key Features:
- Environment variable-based model configuration
- Automatic fallback to default models
- Support for multiple model backends
- Standardized model interface
- Comprehensive logging and error handling
- Mixture-of-Experts (MoE) registry and configuration

Author: ProjectAlpha Team
Compatible with: CoreConductor, SLiM agents, HRM stack, MoELoader
"""

import os
import logging
import hashlib
import json
from pathlib import Path
from typing import Optional, Dict, Any, Protocol
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MoE Expert Registry - Define available expert models
MOE_EXPERT_REGISTRY = {
    "logic_high": {
        "path": os.getenv("LOGIC_HIGH_MODEL_PATH", "models/logic_high.gguf"),
        "size_gb": float(os.getenv("LOGIC_HIGH_SIZE_GB", "2.1")),
        "domain": "logical_reasoning",
        "description": "High-level logical reasoning and deduction",
        "priority": 1,
        "dependencies": []
    },
    "logic_code": {
        "path": os.getenv("LOGIC_CODE_MODEL_PATH", "models/logic_code.gguf"),
        "size_gb": float(os.getenv("LOGIC_CODE_SIZE_GB", "1.8")),
        "domain": "code_analysis",
        "description": "Code analysis, debugging, and programming logic",
        "priority": 1,
        "dependencies": []
    },
    "logic_proof": {
        "path": os.getenv("LOGIC_PROOF_MODEL_PATH", "models/logic_proof.gguf"),
        "size_gb": float(os.getenv("LOGIC_PROOF_SIZE_GB", "2.3")),
        "domain": "mathematical_proof",
        "description": "Mathematical proofs and formal verification",
        "priority": 2,
        "dependencies": []
    },
    "logic_fallback": {
        "path": os.getenv("LOGIC_FALLBACK_MODEL_PATH", "models/logic_fallback.gguf"),
        "size_gb": float(os.getenv("LOGIC_FALLBACK_SIZE_GB", "1.5")),
        "domain": "general_reasoning",
        "description": "General purpose logical reasoning fallback",
        "priority": 0,
        "dependencies": []
    },
    "emote_valence": {
        "path": os.getenv("EMOTE_VALENCE_MODEL_PATH", "models/emote_valence.gguf"),
        "size_gb": float(os.getenv("EMOTE_VALENCE_SIZE_GB", "1.7")),
        "domain": "emotional_valence",
        "description": "Emotional valence detection and generation",
        "priority": 1,
        "dependencies": []
    },
    "emote_arousal": {
        "path": os.getenv("EMOTE_AROUSAL_MODEL_PATH", "models/emote_arousal.gguf"),
        "size_gb": float(os.getenv("EMOTE_AROUSAL_SIZE_GB", "1.6")),
        "domain": "emotional_arousal",
        "description": "Emotional arousal and intensity modulation",
        "priority": 1,
        "dependencies": []
    },
    "creative_metaphor": {
        "path": os.getenv("CREATIVE_METAPHOR_MODEL_PATH", "models/creative_metaphor.gguf"),
        "size_gb": float(os.getenv("CREATIVE_METAPHOR_SIZE_GB", "2.0")),
        "domain": "creative_metaphor",
        "description": "Metaphorical and creative expression",
        "priority": 1,
        "dependencies": []
    },
    "creative_write": {
        "path": os.getenv("CREATIVE_WRITE_MODEL_PATH", "models/creative_write.gguf"),
        "size_gb": float(os.getenv("CREATIVE_WRITE_SIZE_GB", "2.2")),
        "domain": "creative_writing",
        "description": "Creative writing and narrative generation",
        "priority": 2,
        "dependencies": []
    },
    "planning_temporal": {
        "path": os.getenv("PLANNING_TEMPORAL_MODEL_PATH", "models/planning_temporal.gguf"),
        "size_gb": float(os.getenv("PLANNING_TEMPORAL_SIZE_GB", "1.9")),
        "domain": "temporal_planning",
        "description": "Temporal reasoning and scheduling",
        "priority": 1,
        "dependencies": []
    },
    "planning_strategic": {
        "path": os.getenv("PLANNING_STRATEGIC_MODEL_PATH", "models/planning_strategic.gguf"),
        "size_gb": float(os.getenv("PLANNING_STRATEGIC_SIZE_GB", "2.1")),
        "domain": "strategic_planning",
        "description": "Strategic planning and decision making",
        "priority": 1,
        "dependencies": []
    },
    "ritual_symbolic": {
        "path": os.getenv("RITUAL_SYMBOLIC_MODEL_PATH", "models/ritual_symbolic.gguf"),
        "size_gb": float(os.getenv("RITUAL_SYMBOLIC_SIZE_GB", "1.8")),
        "domain": "symbolic_ritual",
        "description": "Symbolic ritual and ceremonial logic",
        "priority": 2,
        "dependencies": []
    },
    "memory_recall": {
        "path": os.getenv("MEMORY_RECALL_MODEL_PATH", "models/memory_recall.gguf"),
        "size_gb": float(os.getenv("MEMORY_RECALL_SIZE_GB", "2.0")),
        "domain": "memory_recall",
        "description": "Memory recall and historical context",
        "priority": 1,
        "dependencies": []
    }
}

# Model verification and integrity checking
MODEL_CHECKSUMS_FILE = "config/model_checksums.json"

def calculate_file_hash(file_path: str, algorithm: str = "sha256") -> Optional[str]:
    """
    Calculate hash checksum for a model file.

    Args:
        file_path: Path to the model file
        algorithm: Hashing algorithm (sha256, md5, sha1)

    Returns:
        Hex digest of the file hash, or None if error
    """
    try:
        if not os.path.exists(file_path):
            logger.warning(f"Model file not found for checksum: {file_path}")
            return None

        hash_func = getattr(hashlib, algorithm.lower())()

        with open(file_path, 'rb') as f:
            # Read in chunks to handle large model files
            chunk_size = 64 * 1024  # 64KB chunks
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                hash_func.update(chunk)

        return hash_func.hexdigest()

    except Exception as e:
        logger.error(f"Error calculating hash for {file_path}: {e}")
        return None

def verify_model_integrity(model_path: str, expected_checksum: Optional[str] = None) -> Dict[str, Any]:
    """
    Verify model file integrity using checksum verification.

    Args:
        model_path: Path to the model file
        expected_checksum: Expected checksum (loads from config if None)

    Returns:
        Verification result dictionary
    """
    result = {
        "file_path": model_path,
        "exists": False,
        "size_bytes": 0,
        "size_mb": 0.0,
        "checksum": None,
        "verification_status": "unknown",
        "expected_checksum": expected_checksum,
        "timestamp": None
    }

    try:
        if not os.path.exists(model_path):
            result["verification_status"] = "file_not_found"
            return result

        # Get file stats
        stat = os.stat(model_path)
        result["exists"] = True
        result["size_bytes"] = stat.st_size
        result["size_mb"] = round(stat.st_size / (1024 * 1024), 2)
        result["timestamp"] = stat.st_mtime

        # Calculate current checksum
        current_checksum = calculate_file_hash(model_path)
        result["checksum"] = current_checksum

        if not current_checksum:
            result["verification_status"] = "checksum_error"
            return result

        # Load expected checksum if not provided
        if expected_checksum is None:
            expected_checksum = load_expected_checksum(model_path)

        if expected_checksum:
            result["expected_checksum"] = expected_checksum
            if current_checksum == expected_checksum:
                result["verification_status"] = "verified"
            else:
                result["verification_status"] = "checksum_mismatch"
                logger.warning(f"Checksum mismatch for {model_path}: "
                             f"expected {expected_checksum[:12]}..., got {current_checksum[:12]}...")
        else:
            result["verification_status"] = "no_expected_checksum"
            logger.info(f"No expected checksum found for {model_path}, storing current: {current_checksum[:12]}...")
            store_model_checksum(model_path, current_checksum)

        return result

    except Exception as e:
        logger.error(f"Error verifying model integrity for {model_path}: {e}")
        result["verification_status"] = "error"
        result["error"] = str(e)
        return result

def load_expected_checksum(model_path: str) -> Optional[str]:
    """Load expected checksum for a model from the checksums file."""
    try:
        if os.path.exists(MODEL_CHECKSUMS_FILE):
            with open(MODEL_CHECKSUMS_FILE, 'r') as f:
                checksums = json.load(f)

            # Normalize path for lookup
            normalized_path = os.path.normpath(model_path)
            return checksums.get(normalized_path)

    except Exception as e:
        logger.error(f"Error loading expected checksum: {e}")

    return None

def store_model_checksum(model_path: str, checksum: str) -> bool:
    """Store model checksum in the checksums file."""
    try:
        # Load existing checksums
        checksums = {}
        if os.path.exists(MODEL_CHECKSUMS_FILE):
            with open(MODEL_CHECKSUMS_FILE, 'r') as f:
                checksums = json.load(f)

        # Normalize path and store
        normalized_path = os.path.normpath(model_path)
        checksums[normalized_path] = checksum

        # Ensure directory exists
        os.makedirs(os.path.dirname(MODEL_CHECKSUMS_FILE), exist_ok=True)

        # Save updated checksums
        with open(MODEL_CHECKSUMS_FILE, 'w') as f:
            json.dump(checksums, f, indent=2)

        logger.info(f"Stored checksum for {model_path}: {checksum[:12]}...")
        return True

    except Exception as e:
        logger.error(f"Error storing checksum: {e}")
        return False

def get_model_info(model: ModelInterface) -> Dict[str, Any]:
    """Get comprehensive model information including verification status."""
    info = {
        "model_type": type(model).__name__,
        "model_name": getattr(model, 'model_name', 'unknown'),
        "available": getattr(model, 'available', True),
        "verification": None
    }

    # Add verification info if model has a file path
    if hasattr(model, 'model_path') and model.model_path:
        verification = verify_model_integrity(model.model_path)
        info["verification"] = verification
        info["file_size_mb"] = verification.get("size_mb", 0)
        info["integrity_status"] = verification.get("verification_status", "unknown")

    return info

class ModelInterface(Protocol):
    """Protocol defining the standard model interface"""

    def generate(self, prompt: str, context: Optional[str] = None, **kwargs) -> str:
        """Generate response from the model"""
        ...

class MockModel:
    """Mock model implementation for testing and fallback"""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.call_count = 0
        logger.info(f"MockModel initialized: {model_name}")

    def generate(self, prompt: str, context: Optional[str] = None, **kwargs) -> str:
        """Generate a mock response"""
        self.call_count += 1

        # Create contextual mock responses based on prompt content
        prompt_lower = prompt.lower()

        if "strategic" in prompt_lower or "plan" in prompt_lower:
            response = f"[MOCK STRATEGIC RESPONSE from {self.model_name}] Based on the strategic context, I recommend analyzing the current situation and developing a comprehensive action plan."
        elif "logic" in prompt_lower or "analyze" in prompt_lower:
            response = f"[MOCK LOGICAL RESPONSE from {self.model_name}] From a logical perspective, we should examine the premises and draw evidence-based conclusions."
        elif "emotion" in prompt_lower or "feel" in prompt_lower:
            response = f"[MOCK EMOTIONAL RESPONSE from {self.model_name}] Understanding the emotional context is crucial for meaningful interactions."
        elif "creative" in prompt_lower or "imagine" in prompt_lower:
            response = f"[MOCK CREATIVE RESPONSE from {self.model_name}] Let's explore innovative possibilities and think outside conventional boundaries."
        else:
            response = f"[MOCK RESPONSE from {self.model_name}] This is a simulated response to: {prompt[:50]}..."

        if context:
            response += f" [Context considered: {context[:30]}...]"

        logger.debug(f"MockModel {self.model_name} generated response (call #{self.call_count})")
        return response

class OllamaModel:
    """Ollama model implementation"""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.available = self._check_ollama_availability()

        if self.available:
            logger.info(f"OllamaModel initialized: {model_name}")
        else:
            logger.warning(f"Ollama not available, model {model_name} will use fallback")

    def _check_ollama_availability(self) -> bool:
        """Check if Ollama is available"""
        try:
            import subprocess
            result = subprocess.run(['ollama', 'list'],
                                  capture_output=True, text=True, timeout=5)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
            return False

    def generate(self, prompt: str, context: Optional[str] = None, **kwargs) -> str:
        """Generate response using Ollama"""
        if not self.available:
            # Fallback to mock if Ollama unavailable
            mock = MockModel(f"Mock-{self.model_name}")
            return mock.generate(prompt, context, **kwargs)

        try:
            import subprocess

            # Prepare the full prompt
            full_prompt = prompt
            if context:
                full_prompt = f"Context: {context}\n\nPrompt: {prompt}"

            # Call Ollama
            cmd = ['ollama', 'run', self.model_name, full_prompt]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

            if result.returncode == 0:
                response = result.stdout.strip()
                logger.debug(f"OllamaModel {self.model_name} generated response")
                return response
            else:
                logger.error(f"Ollama error: {result.stderr}")
                # Fallback to mock
                mock = MockModel(f"Fallback-{self.model_name}")
                return mock.generate(prompt, context, **kwargs)

        except Exception as e:
            logger.error(f"Error running Ollama model {self.model_name}: {e}")
            # Fallback to mock
            mock = MockModel(f"Error-Fallback-{self.model_name}")
            return mock.generate(prompt, context, **kwargs)

class LlamaCppModel:
    """llama.cpp model implementation"""

    def __init__(self, model_name: str, model_path: Optional[str] = None):
        self.model_name = model_name
        self.model_path = model_path or self._find_model_path(model_name)
        self.available = self._check_llamacpp_availability()

        if self.available:
            logger.info(f"LlamaCppModel initialized: {model_name} at {self.model_path}")
        else:
            logger.warning(f"llama.cpp not available, model {model_name} will use fallback")

    def _find_model_path(self, model_name: str) -> Optional[str]:
        """Find model file path"""
        # Common model directories
        search_paths = [
            os.path.expanduser("~/models"),
            os.path.expanduser("~/.cache/huggingface"),
            "./models",
            "/opt/models"
        ]

        for base_path in search_paths:
            if os.path.exists(base_path):
                for root, dirs, files in os.walk(base_path):
                    for file in files:
                        if model_name in file and file.endswith(('.gguf', '.bin', '.ggml')):
                            return os.path.join(root, file)
        return None

    def _check_llamacpp_availability(self) -> bool:
        """Check if llama.cpp is available"""
        try:
            import subprocess
            # Check for llama.cpp executable
            result = subprocess.run(['llama.cpp', '--help'],
                                  capture_output=True, text=True, timeout=5)
            return result.returncode == 0 and self.model_path is not None
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
            return False

    def generate(self, prompt: str, context: Optional[str] = None, **kwargs) -> str:
        """Generate response using llama.cpp"""
        if not self.available:
            # Fallback to mock
            mock = MockModel(f"Mock-{self.model_name}")
            return mock.generate(prompt, context, **kwargs)

        try:
            import subprocess

            # Prepare the full prompt
            full_prompt = prompt
            if context:
                full_prompt = f"Context: {context}\n\nPrompt: {prompt}"

            # Call llama.cpp
            cmd = ['llama.cpp', '-m', self.model_path, '-p', full_prompt, '-n', '256']
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

            if result.returncode == 0:
                response = result.stdout.strip()
                logger.debug(f"LlamaCppModel {self.model_name} generated response")
                return response
            else:
                logger.error(f"llama.cpp error: {result.stderr}")
                # Fallback to mock
                mock = MockModel(f"Fallback-{self.model_name}")
                return mock.generate(prompt, context, **kwargs)

        except Exception as e:
            logger.error(f"Error running llama.cpp model {self.model_name}: {e}")
            # Fallback to mock
            mock = MockModel(f"Error-Fallback-{self.model_name}")
            return mock.generate(prompt, context, **kwargs)

def load_model(env_var: str, default_model: str, backend: str = "auto") -> ModelInterface:
    """
    Load a model based on environment variable or default.

    Args:
        env_var: Environment variable name to check for model specification
        default_model: Default model name if environment variable is not set
        backend: Model backend to use ("ollama", "llamacpp", "mock", "auto")

    Returns:
        Model instance implementing the ModelInterface protocol
    """
    model_name = os.getenv(env_var, default_model)
    backend_override = os.getenv(f"{env_var}_BACKEND", backend)

    logger.info(f"Loading model for {env_var}: {model_name} (backend: {backend_override})")

    # Auto-detect backend if not specified
    if backend_override == "auto":
        # Try Ollama first, then llama.cpp, then mock
        ollama_model = OllamaModel(model_name)
        if ollama_model.available:
            return ollama_model

        llamacpp_model = LlamaCppModel(model_name)
        if llamacpp_model.available:
            return llamacpp_model

        logger.info(f"No available backends found, using MockModel for {model_name}")
        return MockModel(model_name)

    # Use specified backend
    elif backend_override == "ollama":
        return OllamaModel(model_name)

    elif backend_override == "llamacpp":
        return LlamaCppModel(model_name)

    elif backend_override == "mock":
        return MockModel(model_name)

    else:
        logger.warning(f"Unknown backend '{backend_override}', falling back to mock")
        return MockModel(model_name)

class MoEModelAdapter:
    """Adapter to make MoELoader compatible with ModelInterface protocol"""

    def __init__(self, moe_loader):
        self.moe_loader = moe_loader
        self.model_name = "MoE-Expert-Router"

    def generate(self, prompt: str, context: Optional[str] = None, **kwargs) -> str:
        """
        Generate response using MoE expert routing.

        Args:
            prompt: Input prompt
            context: Optional context for expert selection
            **kwargs: Additional parameters

        Returns:
            Generated response from appropriate expert(s)
        """
        try:
            # Use MoE system to score intent and run experts
            expert_weights = self.moe_loader.intent_classifier.score_intent(prompt)

            # Convert context string to dict if provided
            context_dict = {"context": context} if context else None

            # Execute experts based on weights
            responses = self.moe_loader.run_experts(prompt, context_dict)

            if responses:
                # Return summary of responses
                expert_count = len(responses)
                best_response = list(responses.values())[0] if responses else "No response"
                return f"[MoE Response from {expert_count} expert(s)] {best_response}"
            else:
                return f"[MoE Fallback] No suitable experts available for: {prompt[:50]}..."

        except Exception as e:
            logger.error(f"MoE adapter error: {e}")
            return f"[MoE Error] Failed to process request: {str(e)}"


def initialize_moe_system(max_ram_gb: float = 8.0):
    """
    Initialize the Mixture-of-Experts system with expert registry.

    Args:
        max_ram_gb: Maximum RAM allocation for experts in GB

    Returns:
        MoELoader instance if successful, None otherwise
    """
    try:
        from .moe_loader import MoELoader, ExpertInfo

        # Convert registry to ExpertInfo objects
        model_registry = {}
        for expert_id, config in MOE_EXPERT_REGISTRY.items():
            model_registry[expert_id] = ExpertInfo(
                key=expert_id,
                path=config["path"],
                size_gb=config["size_gb"],
                domain=config["domain"],
                description=config["description"],
                priority=config["priority"],
                dependencies=config["dependencies"]
            )

        # Initialize MoE system
        moe_loader = MoELoader(
            model_registry=model_registry,
            ram_limit_gb=max_ram_gb
        )

        logger.info(f"MoE system initialized with {len(model_registry)} experts, max RAM: {max_ram_gb}GB")
        return moe_loader

    except ImportError as e:
        logger.warning(f"MoE system not available: {e}")
        return None
    except Exception as e:
        logger.error(f"Failed to initialize MoE system: {e}")
        return None


def get_moe_configuration() -> Dict[str, Any]:
    """
    Get MoE system configuration from environment variables.

    Returns:
        Configuration dictionary with MoE settings
    """
    return {
        "max_ram_gb": float(os.getenv("MOE_MAX_RAM_GB", "8.0")),
        "cache_strategy": os.getenv("MOE_CACHE_STRATEGY", "lru"),
        "parallel_experts": int(os.getenv("MOE_PARALLEL_EXPERTS", "2")),
        "classification_threshold": float(os.getenv("MOE_CLASSIFICATION_THRESHOLD", "0.7")),
        "fallback_enabled": os.getenv("MOE_FALLBACK_ENABLED", "true").lower() == "true",
        "expert_timeout": int(os.getenv("MOE_EXPERT_TIMEOUT", "30")),
        "load_async": os.getenv("MOE_LOAD_ASYNC", "true").lower() == "true"
    }


def load_conductor_models(use_moe: bool = True) -> Dict[str, Any]:
    """
    Load all standard conductor models with role-specific defaults.
    Optionally initializes MoE system for dynamic expert loading.

    Args:
        use_moe: Whether to use Mixture-of-Experts system

    Returns:
        Dictionary with model instances and MoE system if enabled
    """
    logger.info("Loading conductor model suite...")

    # Standard models
    models = {
        "conductor": load_model("CONDUCTOR_MODEL", "gpt-oss-20b"),
        "logic": load_model("LOGIC_MODEL", "deepseek-coder:1.3b"),
        "emotion": load_model("EMOTION_MODEL", "mistral:7b"),
        "creative": load_model("CREATIVE_MODEL", "mixtral:8x7b")
    }

    # Initialize MoE system if requested
    if use_moe:
        moe_config = get_moe_configuration()
        moe_system = initialize_moe_system(moe_config["max_ram_gb"])
        if moe_system:
            models["moe_loader"] = MoEModelAdapter(moe_system)
            logger.info("MoE system integrated with conductor suite")
        else:
            logger.warning("MoE system initialization failed, using standard models only")

    logger.info(f"Conductor model suite loaded: {list(models.keys())}")
    return models

def load_slim_models() -> Dict[str, ModelInterface]:
    """
    Load all SLiM agent models for hemispheric processing.

    Returns:
        Dictionary mapping SLiM role names to model instances
    """
    logger.info("Loading SLiM agent model suite...")

    models = {
        # Main conductor
        "conductor": load_model("CONDUCTOR_MODEL", "gpt-oss-20b"),

        # Left Brain (Logic) SLiMs - 4 agents
        "logic_high": load_model("LOGIC_HIGH_MODEL", "phi4-mini-reasoning:3.8b"),
        "logic_code": load_model("LOGIC_CODE_MODEL", "qwen2.5-coder:3b"),
        "logic_proof": load_model("LOGIC_PROOF_MODEL", "deepseek-r1:1.5b"),
        "logic_fallback": load_model("LOGIC_FALLBACK_MODEL", "granite3.3:2b"),

        # Right Brain (Emotion & Creativity) SLiMs - 4 agents
        "emotion_valence": load_model("EMOTION_VALENCE_MODEL", "gemma3:1b"),
        "emotion_narrative": load_model("EMOTION_NARRATIVE_MODEL", "phi3:3.8b"),
        "emotion_uncensored": load_model("EMOTION_UNCENSORED_MODEL", "artifish/llama3.2-uncensored:latest"),
        "emotion_creative": load_model("EMOTION_CREATIVE_MODEL", "dolphin-phi:latest")
    }

    logger.info(f"SLiM agent model suite loaded: {list(models.keys())}")
    return models

def load_all_models() -> Dict[str, ModelInterface]:
    """
    Load complete model suite including conductor and all SLiM agents.

    Returns:
        Dictionary mapping all role names to model instances
    """
    logger.info("Loading complete model suite (Conductor + SLiMs)...")

    models = {
        # Main conductor
        "conductor": load_model("CONDUCTOR_MODEL", "gpt-oss-20b"),

        # Legacy role mappings for backward compatibility
        "logic": load_model("LOGIC_MODEL", "deepseek-coder:1.3b"),
        "emotion": load_model("EMOTION_MODEL", "mistral:7b"),
        "creative": load_model("CREATIVE_MODEL", "mixtral:8x7b"),

        # Left Brain (Logic) SLiMs - 4 agents
        "logic_high": load_model("LOGIC_HIGH_MODEL", "phi4-mini-reasoning:3.8b"),
        "logic_code": load_model("LOGIC_CODE_MODEL", "qwen2.5-coder:3b"),
        "logic_proof": load_model("LOGIC_PROOF_MODEL", "deepseek-r1:1.5b"),
        "logic_fallback": load_model("LOGIC_FALLBACK_MODEL", "granite3.3:2b"),

        # Right Brain (Emotion & Creativity) SLiMs - 4 agents
        "emotion_valence": load_model("EMOTION_VALENCE_MODEL", "gemma3:1b"),
        "emotion_narrative": load_model("EMOTION_NARRATIVE_MODEL", "phi3:3.8b"),
        "emotion_uncensored": load_model("EMOTION_UNCENSORED_MODEL", "artifish/llama3.2-uncensored:latest"),
        "emotion_creative": load_model("EMOTION_CREATIVE_MODEL", "dolphin-phi:latest")
    }

    logger.info(f"Complete model suite loaded: {list(models.keys())}")
    return models

def get_model_info(model: ModelInterface) -> Dict[str, Any]:
    """Get information about a loaded model"""
    info = {
        "type": type(model).__name__,
        "available": True
    }

    if hasattr(model, 'model_name'):
        info["model_name"] = getattr(model, 'model_name')

    if hasattr(model, 'available'):
        info["available"] = getattr(model, 'available')

    if hasattr(model, 'call_count'):
        info["call_count"] = getattr(model, 'call_count')

    return info

# Example usage and testing
def example_usage():
    """Example usage of the model loading system"""

    print("Model Loading System Example")
    print("=" * 40)

    # Load individual models
    print("\n1. Loading individual models:")
    conductor_model = load_model("CONDUCTOR_MODEL", "llama3.1:8b")
    logic_model = load_model("LOGIC_MODEL", "deepseek-coder:1.3b")

    print(f"   Conductor model: {get_model_info(conductor_model)}")
    print(f"   Logic model: {get_model_info(logic_model)}")

    # Load complete conductor suite
    print("\n2. Loading conductor model suite:")
    models = load_conductor_models()

    for role, model in models.items():
        info = get_model_info(model)
        print(f"   {role}: {info['type']} ({info.get('model_name', 'unknown')})")

    # Test model generation
    print("\n3. Testing model generation:")

    test_prompts = {
        "conductor": "What strategic approach should we take for this decision?",
        "logic": "Analyze the logical implications of this choice",
        "emotion": "How might the user feel about this response?",
        "creative": "Generate an innovative solution to this problem"
    }

    for role, prompt in test_prompts.items():
        if role in models:
            response = models[role].generate(prompt)
            print(f"   {role}: {response[:80]}...")

if __name__ == "__main__":
    example_usage()

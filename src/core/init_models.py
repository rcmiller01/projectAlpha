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

Author: ProjectAlpha Team
Compatible with: CoreConductor, SLiM agents, HRM stack
"""

import os
import logging
from typing import Optional, Dict, Any, Protocol
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

def load_conductor_models() -> Dict[str, ModelInterface]:
    """
    Load all standard conductor models with role-specific defaults.
    
    Returns:
        Dictionary mapping role names to model instances
    """
    logger.info("Loading conductor model suite...")
    
    models = {
        "conductor": load_model("CONDUCTOR_MODEL", "llama3.1:8b"),
        "logic": load_model("LOGIC_MODEL", "deepseek-coder:1.3b"),
        "emotion": load_model("EMOTION_MODEL", "mistral:7b"),
        "creative": load_model("CREATIVE_MODEL", "mixtral:8x7b")
    }
    
    logger.info(f"Conductor model suite loaded: {list(models.keys())}")
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

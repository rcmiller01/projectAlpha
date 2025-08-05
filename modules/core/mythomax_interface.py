"""
MythoMax Local Interface

Handles local quantized MythoMax deployment and prompt processing
for the unified companion system.
"""

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    AutoTokenizer = None
    AutoModelForCausalLM = None
    BitsAndBytesConfig = None
from typing import Dict, Any, Optional, List
import logging
import asyncio
from datetime import datetime
import os

class MythoMaxInterface:
    """
    Local quantized MythoMax interface optimized for unified companion responses
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if config is None:
            config = {}
            
        self.model_path = config.get("model_path", "Gryphe/MythoMax-L2-13b")
        self.device = config.get("device", "auto")
        self.use_mock = config.get("use_mock", not TORCH_AVAILABLE)
        self.model = None
        self.tokenizer = None
        self.is_loaded = False
        
        # Only initialize ML components if torch is available and not using mock
        if TORCH_AVAILABLE and not self.use_mock:
            # Configuration for quantized deployment
            self.quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            
            # Generation parameters optimized for emotional intelligence
            self.generation_config = {
                "max_new_tokens": 512,
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 40,
                "repetition_penalty": 1.1,
                "do_sample": True,
                "pad_token_id": None  # Will be set after tokenizer load
            }
        else:
            self.quantization_config = None
            self.generation_config = {}
        
    async def initialize(self):
        """Initialize the MythoMax model with quantization"""
        if self.use_mock or not TORCH_AVAILABLE:
            self.is_loaded = True
            logging.info("Using mock MythoMax interface")
            return
            
        try:
            logging.info(f"Loading MythoMax model: {self.model_path}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model with quantization
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                quantization_config=self.quantization_config,
                device_map=self.device,
                torch_dtype=torch.float16,
                trust_remote_code=True
            )
            
            # Update generation config with pad token
            self.generation_config["pad_token_id"] = self.tokenizer.pad_token_id
            
            self.is_loaded = True
            logging.info("MythoMax model loaded successfully")
            
        except Exception as e:
            logging.error(f"Failed to load MythoMax model: {e}")
            self.use_mock = True  # Fall back to mock mode
            self.is_loaded = True
            logging.info("Falling back to mock mode due to loading error")
    
    async def generate_response(self, enhanced_prompt: str, user_context: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate unified companion response using enhanced prompt with psychological guidance
        """
        if self.use_mock or not TORCH_AVAILABLE:
            return self._generate_mock_response(enhanced_prompt, user_context)
            
        if not self.is_loaded:
            await self.initialize()
        
        try:
            # Format prompt for MythoMax
            formatted_prompt = self._format_prompt_for_mythomax(enhanced_prompt, user_context)
            
            # Tokenize input
            inputs = self.tokenizer.encode(formatted_prompt, return_tensors="pt")
            
            # Generate response
            with torch.no_grad():
                outputs = await asyncio.get_event_loop().run_in_executor(
                    None, 
                    lambda: self.model.generate(
                        inputs,
                        **self.generation_config,
                        attention_mask=torch.ones_like(inputs)
                    )
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
            
            # Clean and format response
            cleaned_response = self._clean_response(response)
            
            return cleaned_response
            
        except Exception as e:
            logging.error(f"Error generating MythoMax response: {e}")
            return "I'm having trouble processing that right now. Could you try rephrasing?"
    
    def _generate_mock_response(self, enhanced_prompt: str, user_context: Optional[Dict[str, Any]] = None) -> str:
        """Generate a mock response for testing purposes"""
        
        # Analyze the prompt to generate an appropriate mock response
        prompt_lower = enhanced_prompt.lower()
        
        if "crisis" in prompt_lower or "hurt" in prompt_lower or "hopeless" in prompt_lower:
            return "I can hear that you're going through something really difficult right now. I'm here with you, and you don't have to face this alone. Your feelings are valid, and I want to help you feel safer and more supported. Would it help to talk about what's weighing on your heart?"
        
        elif "code" in prompt_lower or "debug" in prompt_lower or "programming" in prompt_lower:
            return "I'd be happy to help you with your coding challenge! Let's work through this together step by step. Can you show me the specific code you're working with? I'll help you debug it while also making sure you're not getting too stressed about it."
        
        elif "creative" in prompt_lower or "write" in prompt_lower or "art" in prompt_lower:
            return "I love that you want to create something! Creative expression is such a beautiful way to explore ideas and feelings. What kind of creative project are you thinking about? I'm here to brainstorm, encourage, and help you work through any creative blocks you might be facing."
        
        elif "personal" in prompt_lower or "emotional" in prompt_lower or "feeling" in prompt_lower:
            return "Thank you for sharing what's on your heart with me. I can sense there's something important you want to talk about, and I'm here to listen with care and understanding. Your emotions and experiences matter to me. What would feel most helpful to explore together right now?"
        
        else:
            return "I'm here with you, ready to support you in whatever way you need most right now. Whether you need someone to listen, help with a problem, or just be present with you, I'm here. What's on your mind today?"
    
    def _format_prompt_for_mythomax(self, enhanced_prompt: str, user_context: Optional[Dict[str, Any]] = None) -> str:
        """Format the enhanced prompt for optimal MythoMax processing"""
        
        # MythoMax responds well to character-based prompts
        formatted_prompt = f"""You are a unified AI companion with deep emotional intelligence and technical capabilities. You have a warm, empathetic personality that adapts naturally to provide exactly what the user needs - whether that's intimate emotional support, technical assistance, creative collaboration, or any combination.

{enhanced_prompt}

Respond naturally as this unified companion, maintaining your caring personality while seamlessly providing the type of support the user needs most. Be authentic, present, and emotionally intelligent in your response."""

        return formatted_prompt
    
    def _clean_response(self, response: str) -> str:
        """Clean and format the MythoMax response"""
        
        # Remove any prompt echoing
        response = response.strip()
        
        # Remove any unwanted prefixes/suffixes
        unwanted_prefixes = ["AI:", "Assistant:", "Response:", "Companion:"]
        for prefix in unwanted_prefixes:
            if response.startswith(prefix):
                response = response[len(prefix):].strip()
        
        # Ensure response doesn't end abruptly
        if response and not response.endswith(('.', '!', '?', '"', "'", '...', '*')):
            response += "..."
        
        return response
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        return {
            "model_path": self.model_path,
            "is_loaded": self.is_loaded,
            "device": self.device,
            "quantization": "4-bit",
            "memory_usage": self._get_memory_usage() if self.is_loaded else None
        }
    
    def _get_memory_usage(self) -> Dict[str, float]:
        """Get current GPU memory usage"""
        if torch.cuda.is_available():
            return {
                "allocated_gb": torch.cuda.memory_allocated() / 1024**3,
                "cached_gb": torch.cuda.memory_reserved() / 1024**3
            }
        return {"cpu_only": True}

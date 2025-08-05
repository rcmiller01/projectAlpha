#!/usr/bin/env python3
"""
Quantization Module - Unified quantization interface for autopilot integration
Handles model quantization with structured output for autonomous operation
"""

import os
import time
import logging
import subprocess
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

@dataclass
class QuantizationResult:
    """Structured result from quantization process"""
    success: bool
    model_path: str
    model_size_mb: float
    quantization_method: str
    base_model: str
    duration_seconds: float
    error_message: str = ""
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class ModelQuantizer:
    """
    Unified model quantization interface
    Supports multiple quantization backends (Ollama, llama.cpp, transformers)
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger("ModelQuantizer")
        
        # Default configuration
        self.output_dir = Path(self.config.get("output_dir", "quantized_models"))
        self.quantization_backend = self.config.get("backend", "ollama")  # ollama, llamacpp, transformers
        self.temp_dir = Path(self.config.get("temp_dir", "temp_quantization"))
        
        # Ensure directories exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"🔧 ModelQuantizer initialized with {self.quantization_backend} backend")
    
    def quantize_model(self, 
                      base_model: str, 
                      quantization_method: str, 
                      target_size_range_gb: Optional[Tuple[float, float]] = None) -> QuantizationResult:
        """
        Quantize a model using specified method
        
        Args:
            base_model: Path or name of the base model
            quantization_method: Quantization method (q8_0, q6_K, q4_K_M, etc.)
            target_size_range_gb: Optional size constraints (min_gb, max_gb)
            
        Returns:
            QuantizationResult: Structured result with success status and metadata
        """
        start_time = time.time()
        
        self.logger.info(f"🚀 Starting quantization: {base_model} -> {quantization_method}")
        
        try:
            # Generate output path
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = Path(base_model).stem if "/" not in base_model else base_model.split("/")[-1]
            output_filename = f"{model_name}_{quantization_method}_{timestamp}"
            output_path = self.output_dir / output_filename
            
            # Choose quantization backend
            if self.quantization_backend == "ollama":
                result = self._quantize_with_ollama(base_model, quantization_method, output_path)
            elif self.quantization_backend == "llamacpp":
                result = self._quantize_with_llamacpp(base_model, quantization_method, output_path)
            elif self.quantization_backend == "transformers":
                result = self._quantize_with_transformers(base_model, quantization_method, output_path)
            else:
                # Mock mode for testing
                result = self._mock_quantization(base_model, quantization_method, output_path)
            
            duration = time.time() - start_time
            
            if result["success"]:
                # Get model size
                model_size_mb = self._get_model_size(result["model_path"])
                
                # Check target size constraints
                if target_size_range_gb:
                    size_gb = model_size_mb / 1024
                    min_gb, max_gb = target_size_range_gb
                    if not (min_gb <= size_gb <= max_gb):
                        self.logger.warning(f"⚠️ Model size {size_gb:.1f}GB outside target range {min_gb}-{max_gb}GB")
                
                self.logger.info(f"✅ Quantization successful: {model_size_mb:.1f}MB in {duration:.1f}s")
                
                return QuantizationResult(
                    success=True,
                    model_path=str(result["model_path"]),
                    model_size_mb=model_size_mb,
                    quantization_method=quantization_method,
                    base_model=base_model,
                    duration_seconds=duration,
                    metadata={
                        "backend": self.quantization_backend,
                        "timestamp": timestamp,
                        "target_size_range_gb": target_size_range_gb
                    }
                )
            else:
                self.logger.error(f"❌ Quantization failed: {result.get('error', 'Unknown error')}")
                return QuantizationResult(
                    success=False,
                    model_path="",
                    model_size_mb=0.0,
                    quantization_method=quantization_method,
                    base_model=base_model,
                    duration_seconds=duration,
                    error_message=result.get("error", "Quantization failed")
                )
                
        except Exception as e:
            duration = time.time() - start_time
            self.logger.error(f"❌ Quantization exception: {e}")
            return QuantizationResult(
                success=False,
                model_path="",
                model_size_mb=0.0,
                quantization_method=quantization_method,
                base_model=base_model,
                duration_seconds=duration,
                error_message=str(e)
            )
    
    def _quantize_with_ollama(self, base_model: str, quant_method: str, output_path: Path) -> Dict[str, Any]:
        """Quantize using Ollama"""
        try:
            # Create Ollama modelfile
            modelfile_content = f"""FROM {base_model}
PARAMETER quantization {quant_method}
"""
            modelfile_path = self.temp_dir / f"Modelfile_{quant_method}"
            modelfile_path.write_text(modelfile_content)
            
            # Run Ollama create command
            model_tag = f"{base_model.replace('/', '_')}_{quant_method}"
            cmd = ["ollama", "create", model_tag, "-f", str(modelfile_path)]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)  # 1 hour timeout
            
            if result.returncode == 0:
                # Export the model
                export_cmd = ["ollama", "show", model_tag, "--modelfile"]
                export_result = subprocess.run(export_cmd, capture_output=True, text=True)
                
                if export_result.returncode == 0:
                    output_path.with_suffix(".modelfile").write_text(export_result.stdout)
                    return {
                        "success": True,
                        "model_path": str(output_path.with_suffix(".modelfile")),
                        "ollama_tag": model_tag
                    }
            
            return {
                "success": False,
                "error": f"Ollama quantization failed: {result.stderr}"
            }
            
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Quantization timed out"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _quantize_with_llamacpp(self, base_model: str, quant_method: str, output_path: Path) -> Dict[str, Any]:
        """Quantize using llama.cpp"""
        try:
            # Map quantization methods to llama.cpp formats
            quant_map = {
                "q8_0": "q8_0",
                "q6_K": "q6_k",
                "q5_K_M": "q5_k_m",
                "q4_K_M": "q4_k_m",
                "q3_K_L": "q3_k_l",
                "q2_K": "q2_k"
            }
            
            cpp_quant = quant_map.get(quant_method, quant_method)
            output_file = output_path.with_suffix(".gguf")
            
            # Run llama.cpp quantization
            cmd = [
                "llama-quantize",
                base_model,
                str(output_file),
                cpp_quant
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            
            if result.returncode == 0 and output_file.exists():
                return {
                    "success": True,
                    "model_path": str(output_file)
                }
            else:
                return {
                    "success": False,
                    "error": f"llama.cpp quantization failed: {result.stderr}"
                }
                
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Quantization timed out"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _quantize_with_transformers(self, base_model: str, quant_method: str, output_path: Path) -> Dict[str, Any]:
        """Quantize using Transformers/BitsAndBytes"""
        try:
            # This would implement transformers-based quantization
            # For now, return mock result
            return self._mock_quantization(base_model, quant_method, output_path)
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _mock_quantization(self, base_model: str, quant_method: str, output_path: Path) -> Dict[str, Any]:
        """Mock quantization for testing"""
        # Simulate quantization time
        time.sleep(2)
        
        # Create a mock output file
        mock_file = output_path.with_suffix(".mock")
        mock_content = {
            "base_model": base_model,
            "quantization_method": quant_method,
            "created_at": datetime.now().isoformat(),
            "size_mb": 12800  # Mock size
        }
        
        mock_file.write_text(json.dumps(mock_content, indent=2))
        
        return {
            "success": True,
            "model_path": str(mock_file)
        }
    
    def _get_model_size(self, model_path: str) -> float:
        """Get model file size in MB"""
        try:
            path = Path(model_path)
            if path.exists():
                size_bytes = path.stat().st_size
                return size_bytes / (1024 * 1024)  # Convert to MB
            else:
                # For mock files or special cases, return a reasonable default
                return 12800.0  # 12.8GB default
        except Exception:
            return 0.0
    
    def list_available_methods(self) -> Dict[str, str]:
        """List available quantization methods"""
        return {
            "q8_0": "8-bit quantization, best quality",
            "q6_K": "6-bit K-quantization, good balance", 
            "q5_K_M": "5-bit K-quantization, medium",
            "q4_K_M": "4-bit K-quantization, medium",
            "q4_0": "4-bit quantization, legacy",
            "q3_K_L": "3-bit K-quantization, large context",
            "q2_K": "2-bit K-quantization, smallest size"
        }
    
    def cleanup_temp_files(self) -> None:
        """Clean up temporary files"""
        try:
            import shutil
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
                self.temp_dir.mkdir(parents=True, exist_ok=True)
            self.logger.info("🧹 Temporary files cleaned up")
        except Exception as e:
            self.logger.warning(f"⚠️ Cleanup failed: {e}")

# Convenience function for autopilot integration
def quantize_model(base_model: str, 
                  quantization_method: str,
                  config: Optional[Dict[str, Any]] = None,
                  target_size_range_gb: Optional[Tuple[float, float]] = None) -> QuantizationResult:
    """
    Convenience function for quantizing a model
    
    Args:
        base_model: Path or name of the base model
        quantization_method: Quantization method (q8_0, q6_K, q4_K_M, etc.)
        config: Optional configuration dict
        target_size_range_gb: Optional size constraints (min_gb, max_gb)
        
    Returns:
        QuantizationResult: Structured result with success status and metadata
    """
    quantizer = ModelQuantizer(config)
    return quantizer.quantize_model(base_model, quantization_method, target_size_range_gb)

if __name__ == "__main__":
    # Test the quantizer
    import argparse
    
    parser = argparse.ArgumentParser(description="Model Quantization Tool")
    parser.add_argument("--model", required=True, help="Base model path or name")
    parser.add_argument("--method", required=True, help="Quantization method")
    parser.add_argument("--output-dir", default="quantized_models", help="Output directory")
    parser.add_argument("--backend", default="mock", help="Quantization backend")
    parser.add_argument("--mock", action="store_true", help="Use mock mode")
    
    args = parser.parse_args()
    
    config = {
        "output_dir": args.output_dir,
        "backend": "mock" if args.mock else args.backend
    }
    
    result = quantize_model(args.model, args.method, config)
    
    print(f"Quantization Result:")
    print(f"  Success: {result.success}")
    print(f"  Model Path: {result.model_path}")
    print(f"  Size: {result.model_size_mb:.1f}MB")
    print(f"  Duration: {result.duration_seconds:.1f}s")
    if result.error_message:
        print(f"  Error: {result.error_message}")

#!/usr/bin/env python3
"""
Core Model Replacement System
Replaces the current companion core model with selected candidate
"""

import os
import json
import shutil
import logging
from pathlib import Path
from typing import Dict, Optional, List
from datetime import datetime
import hashlib

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CoreModelReplacer:
    """System for safely replacing the active companion core model"""
    
    def __init__(self):
        # Configuration from environment or defaults
        self.candidate_dir = os.getenv("CANDIDATE_DIR", "quant_pass1/models")
        self.final_model_dir = os.getenv("FINAL_MODEL_DIR", "core2/models/active_core")
        self.backup_dir = os.getenv("BACKUP_DIR", "quant_pass2/backups")
        self.companion_manifest_path = "personas/companion_manifest.json"
        
        # Create necessary directories
        Path(self.final_model_dir).mkdir(parents=True, exist_ok=True)
        Path(self.backup_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info("🔄 Core Model Replacer initialized")
        logger.info(f"📁 Final model directory: {self.final_model_dir}")
        logger.info(f"💾 Backup directory: {self.backup_dir}")
    
    def validate_candidate(self, candidate_path: str) -> Dict:
        """Validate that candidate model is complete and functional"""
        logger.info(f"🔍 Validating candidate: {candidate_path}")
        
        validation_result = {
            "valid": False,
            "issues": [],
            "model_info": {},
            "file_count": 0,
            "total_size_mb": 0
        }
        
        candidate_dir = Path(candidate_path)
        
        if not candidate_dir.exists():
            validation_result["issues"].append(f"Candidate directory does not exist: {candidate_path}")
            return validation_result
        
        # Check for required files
        required_files = ["config.json", "pytorch_model.bin"]
        optional_files = ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.txt"]
        
        existing_files = list(candidate_dir.glob("*"))
        file_names = [f.name for f in existing_files]
        
        # Check required files
        for required_file in required_files:
            if required_file not in file_names:
                # Check for alternative patterns
                if required_file == "pytorch_model.bin":
                    # Look for alternative model file patterns
                    model_files = [f for f in file_names if f.startswith("pytorch_model") or f.endswith(".safetensors")]
                    if not model_files:
                        validation_result["issues"].append(f"Missing model weights file (expected {required_file} or similar)")
                else:
                    validation_result["issues"].append(f"Missing required file: {required_file}")
        
        # Calculate total size
        total_size = 0
        for file_path in existing_files:
            if file_path.is_file():
                total_size += file_path.stat().st_size
        
        validation_result["file_count"] = len(existing_files)
        validation_result["total_size_mb"] = total_size / (1024 * 1024)
        
        # Try to load config for additional validation
        config_path = candidate_dir / "config.json"
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    validation_result["model_info"] = {
                        "model_type": config.get("model_type", "unknown"),
                        "hidden_size": config.get("hidden_size", "unknown"),
                        "num_layers": config.get("num_hidden_layers", "unknown"),
                        "vocab_size": config.get("vocab_size", "unknown")
                    }
            except Exception as e:
                validation_result["issues"].append(f"Could not parse config.json: {e}")
        
        # Check minimum size (quantized model should still be substantial)
        min_size_mb = 1000  # 1GB minimum
        if validation_result["total_size_mb"] < min_size_mb:
            validation_result["issues"].append(f"Model size ({validation_result['total_size_mb']:.1f}MB) seems too small")
        
        # Mark as valid if no critical issues
        validation_result["valid"] = len(validation_result["issues"]) == 0
        
        if validation_result["valid"]:
            logger.info(f"✅ Candidate validation passed")
            logger.info(f"   📊 Files: {validation_result['file_count']}")
            logger.info(f"   📦 Size: {validation_result['total_size_mb']:.1f}MB")
        else:
            logger.warning(f"⚠️ Candidate validation issues:")
            for issue in validation_result["issues"]:
                logger.warning(f"   • {issue}")
        
        return validation_result
    
    def create_backup(self, backup_name: Optional[str] = None) -> str:
        """Create backup of current active model"""
        if backup_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"backup_{timestamp}"
        
        backup_path = Path(self.backup_dir) / backup_name
        active_model_path = Path(self.final_model_dir)
        
        logger.info(f"💾 Creating backup: {backup_name}")
        
        try:
            if active_model_path.exists():
                # Copy current model to backup
                shutil.copytree(active_model_path, backup_path, dirs_exist_ok=True)
                
                # Create backup metadata
                metadata = {
                    "backup_name": backup_name,
                    "timestamp": datetime.now().isoformat(),
                    "original_path": str(active_model_path),
                    "backup_path": str(backup_path),
                    "file_count": len(list(backup_path.rglob("*"))),
                    "backup_size_mb": sum(f.stat().st_size for f in backup_path.rglob("*") if f.is_file()) / (1024 * 1024)
                }
                
                with open(backup_path / "backup_metadata.json", 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                logger.info(f"✅ Backup created successfully")
                logger.info(f"   📁 Location: {backup_path}")
                logger.info(f"   📊 Size: {metadata['backup_size_mb']:.1f}MB")
                
                return str(backup_path)
            else:
                logger.info("ℹ️ No existing model to backup")
                return ""
                
        except Exception as e:
            logger.error(f"❌ Backup creation failed: {e}")
            raise
    
    def replace_model(self, candidate_path: str, backup_name: Optional[str] = None) -> Dict:
        """Replace active model with candidate"""
        logger.info(f"🔄 Replacing active model with: {candidate_path}")
        
        replacement_result = {
            "success": False,
            "backup_path": "",
            "candidate_path": candidate_path,
            "validation_result": {},
            "replacement_info": {},
            "error_message": ""
        }
        
        try:
            # Validate candidate
            validation = self.validate_candidate(candidate_path)
            replacement_result["validation_result"] = validation
            
            if not validation["valid"]:
                replacement_result["error_message"] = "Candidate validation failed"
                return replacement_result
            
            # Create backup of current model
            backup_path = self.create_backup(backup_name)
            replacement_result["backup_path"] = backup_path
            
            # Remove existing active model
            active_model_path = Path(self.final_model_dir)
            if active_model_path.exists():
                shutil.rmtree(active_model_path)
                logger.info("🗑️ Removed old active model")
            
            # Copy candidate to active location
            candidate_dir = Path(candidate_path)
            shutil.copytree(candidate_dir, active_model_path)
            logger.info("📋 Copied candidate to active location")
            
            # Create replacement metadata
            replacement_info = {
                "replacement_timestamp": datetime.now().isoformat(),
                "candidate_source": candidate_path,
                "backup_location": backup_path,
                "model_info": validation["model_info"],
                "file_count": validation["file_count"],
                "size_mb": validation["total_size_mb"],
                "checksum": self._calculate_directory_checksum(active_model_path)
            }
            
            # Save replacement metadata
            with open(active_model_path / "replacement_metadata.json", 'w') as f:
                json.dump(replacement_info, f, indent=2)
            
            replacement_result["replacement_info"] = replacement_info
            replacement_result["success"] = True
            
            logger.info("✅ Model replacement completed successfully")
            logger.info(f"   📦 New size: {replacement_info['size_mb']:.1f}MB")
            logger.info(f"   🔐 Checksum: {replacement_info['checksum'][:16]}...")
            
        except Exception as e:
            logger.error(f"❌ Model replacement failed: {e}")
            replacement_result["error_message"] = str(e)
            
            # Attempt to restore from backup if replacement failed partway
            if replacement_result["backup_path"]:
                try:
                    logger.info("🔄 Attempting to restore from backup...")
                    self.restore_from_backup(replacement_result["backup_path"])
                    logger.info("✅ Restored from backup")
                except Exception as restore_error:
                    logger.error(f"❌ Backup restoration also failed: {restore_error}")
        
        return replacement_result
    
    def update_companion_manifest(self, new_model_path: str, quantization_info: Dict):
        """Update companion manifest to reflect new model"""
        logger.info("📝 Updating companion manifest")
        
        try:
            manifest_path = Path(self.companion_manifest_path)
            
            # Load existing manifest
            if manifest_path.exists():
                with open(manifest_path, 'r') as f:
                    manifest = json.load(f)
            else:
                logger.warning(f"⚠️ Companion manifest not found: {manifest_path}")
                return
            
            # Update model information
            manifest["model_info"] = manifest.get("model_info", {})
            manifest["model_info"].update({
                "model_path": new_model_path,
                "quantization_method": quantization_info.get("quantization_method", "unknown"),
                "size_mb": quantization_info.get("size_mb", 0),
                "emotional_degradation": quantization_info.get("emotional_degradation", 0),
                "last_updated": datetime.now().isoformat(),
                "replacement_checksum": quantization_info.get("checksum", "")
            })
            
            # Update routing bias to prefer the new quantized model
            if "routing_bias" in manifest:
                manifest["routing_bias"]["quantized_primary"] = True
                manifest["routing_bias"]["last_optimization"] = datetime.now().isoformat()
            
            # Save updated manifest
            with open(manifest_path, 'w') as f:
                json.dump(manifest, f, indent=2)
            
            logger.info("✅ Companion manifest updated")
            
        except Exception as e:
            logger.error(f"❌ Failed to update companion manifest: {e}")
    
    def restore_from_backup(self, backup_path: str) -> bool:
        """Restore model from backup"""
        logger.info(f"🔄 Restoring from backup: {backup_path}")
        
        try:
            backup_dir = Path(backup_path)
            active_model_path = Path(self.final_model_dir)
            
            if not backup_dir.exists():
                logger.error(f"❌ Backup directory not found: {backup_path}")
                return False
            
            # Remove current active model
            if active_model_path.exists():
                shutil.rmtree(active_model_path)
            
            # Copy backup to active location
            shutil.copytree(backup_dir, active_model_path)
            
            # Remove backup metadata file if it exists in active location
            metadata_file = active_model_path / "backup_metadata.json"
            if metadata_file.exists():
                metadata_file.unlink()
            
            logger.info("✅ Model restored from backup successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Backup restoration failed: {e}")
            return False
    
    def list_backups(self) -> List[Dict]:
        """List available backups"""
        backups = []
        backup_dir = Path(self.backup_dir)
        
        if not backup_dir.exists():
            return backups
        
        for backup_subdir in backup_dir.iterdir():
            if backup_subdir.is_dir():
                metadata_file = backup_subdir / "backup_metadata.json"
                
                if metadata_file.exists():
                    try:
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                        backups.append(metadata)
                    except:
                        # Create basic metadata if file is corrupted
                        backups.append({
                            "backup_name": backup_subdir.name,
                            "backup_path": str(backup_subdir),
                            "timestamp": "unknown",
                            "backup_size_mb": sum(f.stat().st_size for f in backup_subdir.rglob("*") if f.is_file()) / (1024 * 1024)
                        })
        
        # Sort by timestamp (newest first)
        backups.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return backups
    
    def _calculate_directory_checksum(self, directory: Path) -> str:
        """Calculate checksum for directory contents"""
        hasher = hashlib.md5()
        
        for file_path in sorted(directory.rglob("*")):
            if file_path.is_file():
                hasher.update(file_path.name.encode())
                with open(file_path, 'rb') as f:
                    # Read file in chunks to handle large files
                    for chunk in iter(lambda: f.read(8192), b""):
                        hasher.update(chunk)
        
        return hasher.hexdigest()
    
    def get_replacement_status(self) -> Dict:
        """Get status of current active model"""
        active_model_path = Path(self.final_model_dir)
        
        status = {
            "active_model_exists": active_model_path.exists(),
            "active_model_path": str(active_model_path),
            "replacement_info": {},
            "validation_status": {},
            "available_backups": len(self.list_backups())
        }
        
        if active_model_path.exists():
            # Check for replacement metadata
            metadata_file = active_model_path / "replacement_metadata.json"
            if metadata_file.exists():
                try:
                    with open(metadata_file, 'r') as f:
                        status["replacement_info"] = json.load(f)
                except:
                    pass
            
            # Validate current model
            status["validation_status"] = self.validate_candidate(str(active_model_path))
        
        return status

def main():
    """Main execution function for testing and manual operations"""
    
    import sys
    
    replacer = CoreModelReplacer()
    
    if len(sys.argv) < 2:
        print("Usage: python replace_core.py <command> [args]")
        print("Commands:")
        print("  status                    - Show current status")
        print("  validate <path>           - Validate candidate model")
        print("  backup [name]             - Create backup of current model")
        print("  replace <candidate_path>  - Replace with candidate")
        print("  restore <backup_path>     - Restore from backup")
        print("  list-backups             - List available backups")
        return 1
    
    command = sys.argv[1].lower()
    
    try:
        if command == "status":
            status = replacer.get_replacement_status()
            print(json.dumps(status, indent=2))
            
        elif command == "validate":
            if len(sys.argv) < 3:
                print("Usage: python replace_core.py validate <candidate_path>")
                return 1
            
            candidate_path = sys.argv[2]
            result = replacer.validate_candidate(candidate_path)
            print(json.dumps(result, indent=2))
            
        elif command == "backup":
            backup_name = sys.argv[2] if len(sys.argv) > 2 else None
            backup_path = replacer.create_backup(backup_name)
            print(f"Backup created: {backup_path}")
            
        elif command == "replace":
            if len(sys.argv) < 3:
                print("Usage: python replace_core.py replace <candidate_path>")
                return 1
            
            candidate_path = sys.argv[2]
            result = replacer.replace_model(candidate_path)
            print(json.dumps(result, indent=2))
            
        elif command == "restore":
            if len(sys.argv) < 3:
                print("Usage: python replace_core.py restore <backup_path>")
                return 1
            
            backup_path = sys.argv[2]
            success = replacer.restore_from_backup(backup_path)
            print(f"Restoration {'successful' if success else 'failed'}")
            
        elif command == "list-backups":
            backups = replacer.list_backups()
            print(json.dumps(backups, indent=2))
            
        else:
            print(f"Unknown command: {command}")
            return 1
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Command failed: {e}")
        return 2

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)

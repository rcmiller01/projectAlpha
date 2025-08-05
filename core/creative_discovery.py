"""
Creative Model Discovery System
Dynamically finds and installs AI models for creative collaboration
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
import json
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

class CreativeModelDiscovery:
    """
    Discovers and manages AI models for creative projects
    Replaces hardcoded model support with dynamic discovery
    """
    
    def __init__(self):
        self.model_registry = {}
        self.installed_models = set()
        self.model_categories = {
            "music": ["musicgen", "audiocraft", "bark", "riffusion"],
            "art": ["stable-diffusion", "dall-e", "midjourney", "imagen"],
            "writing": ["gpt", "claude", "llama", "palm"],
            "video": ["runway", "pika", "stable-video", "luma"],
            "voice": ["elevenlabs", "bark", "tortoise", "coqui"],
            "code": ["codex", "codewhisperer", "copilot", "starcoder"]
        }
        self.model_database = {}
        
    async def initialize(self):
        """Initialize the model discovery system"""
        await self._load_model_database()
        await self._scan_installed_models()
    
    async def _load_model_database(self):
        """Load database of available models and their capabilities"""
        
        # This would typically load from a configuration file or API
        # For now, we'll define a comprehensive model database
        self.model_database = {
            # Music Generation Models
            "musicgen-small": {
                "name": "MusicGen Small",
                "category": "music",
                "description": "Facebook's MusicGen model for music generation",
                "size": "1.5GB",
                "install_command": "pip install musicgen",
                "capabilities": ["melody_generation", "accompaniment", "style_transfer"],
                "supported_formats": ["wav", "mp3"],
                "requirements": ["torch", "torchaudio"]
            },
            "audiocraft": {
                "name": "AudioCraft",
                "category": "music", 
                "description": "Meta's AudioCraft for music and audio generation",
                "size": "3.2GB",
                "install_command": "pip install audiocraft",
                "capabilities": ["music_generation", "sound_effects", "audio_compression"],
                "supported_formats": ["wav", "mp3", "flac"],
                "requirements": ["torch", "torchaudio", "xformers"]
            },
            
            # Art Generation Models
            "stable-diffusion-xl": {
                "name": "Stable Diffusion XL",
                "category": "art",
                "description": "Advanced image generation model",
                "size": "6.9GB",
                "install_command": "pip install diffusers transformers",
                "capabilities": ["text_to_image", "image_to_image", "inpainting"],
                "supported_formats": ["png", "jpg", "webp"],
                "requirements": ["torch", "transformers", "diffusers"]
            },
            "controlnet": {
                "name": "ControlNet",
                "category": "art",
                "description": "Precise control over image generation",
                "size": "2.5GB", 
                "install_command": "pip install controlnet-aux",
                "capabilities": ["pose_control", "edge_detection", "depth_control"],
                "supported_formats": ["png", "jpg"],
                "requirements": ["torch", "transformers", "controlnet-aux"]
            },
            
            # Writing Models
            "gpt-neo": {
                "name": "GPT-Neo",
                "category": "writing",
                "description": "Open source language model for creative writing",
                "size": "2.7GB",
                "install_command": "pip install transformers",
                "capabilities": ["story_writing", "poetry", "dialogue", "screenwriting"],
                "supported_formats": ["txt", "md"],
                "requirements": ["transformers", "torch"]
            },
            
            # Video Generation Models  
            "animatediff": {
                "name": "AnimateDiff",
                "category": "video",
                "description": "Image animation and video generation",
                "size": "4.1GB",
                "install_command": "pip install animatediff",
                "capabilities": ["image_animation", "video_generation", "motion_control"],
                "supported_formats": ["mp4", "gif"],
                "requirements": ["torch", "diffusers", "opencv-python"]
            },
            
            # Voice Synthesis Models
            "bark": {
                "name": "Bark",
                "category": "voice",
                "description": "Realistic voice synthesis with emotional control",
                "size": "1.8GB",
                "install_command": "pip install bark",
                "capabilities": ["text_to_speech", "emotion_control", "voice_cloning"],
                "supported_formats": ["wav", "mp3"],
                "requirements": ["torch", "torchaudio", "scipy"]
            },
            
            # Code Generation Models
            "starcoder": {
                "name": "StarCoder",
                "category": "code",
                "description": "Code generation and completion model",
                "size": "15.5GB",
                "install_command": "pip install transformers",
                "capabilities": ["code_completion", "code_generation", "debugging"],
                "supported_formats": ["py", "js", "cpp", "java"],
                "requirements": ["transformers", "torch"]
            }
        }
    
    async def _scan_installed_models(self):
        """Scan for already installed models"""
        
        # Check which models are already available
        for model_id, model_info in self.model_database.items():
            try:
                # Try to import required packages
                requirements = model_info.get("requirements", [])
                for req in requirements:
                    __import__(req)
                
                # If all requirements are met, consider it installed
                self.installed_models.add(model_id)
                logger.info(f"✓ Found installed model: {model_info['name']}")
                
            except ImportError:
                logger.debug(f"Model {model_id} not installed")
    
    async def find_models_for_project(self, project_type: str, requirements: Dict[str, Any] = {}) -> List[Dict[str, Any]]:
        """Find suitable models for a specific project type"""
        
        suitable_models = []
        
        # Get models for the project category
        category_models = [
            model_id for model_id, model_info in self.model_database.items()
            if model_info["category"] == project_type
        ]
        
        for model_id in category_models:
            model_info = self.model_database[model_id]
            
            model_status = {
                "id": model_id,
                "name": model_info["name"],
                "description": model_info["description"],
                "capabilities": model_info["capabilities"],
                "size": model_info["size"],
                "status": "installed" if model_id in self.installed_models else "installable"
            }
            
            suitable_models.append(model_status)
        
        # Sort by installation status (installed first)
        suitable_models.sort(key=lambda x: x["status"] == "installed", reverse=True)
        
        return suitable_models
    
    async def find_models_for_message(self, message: str) -> List[Dict[str, Any]]:
        """Analyze message and suggest relevant models"""
        
        message_lower = message.lower()
        relevant_models = []
        
        # Keywords to model mapping
        keyword_mappings = {
            "music": ["music", "song", "melody", "compose", "beat", "rhythm"],
            "art": ["art", "image", "picture", "draw", "paint", "visual", "design"],
            "writing": ["write", "story", "poem", "script", "text", "novel"],
            "video": ["video", "animation", "movie", "film", "animate"],
            "voice": ["voice", "speak", "speech", "audio", "sound"],
            "code": ["code", "program", "script", "function", "algorithm"]
        }
        
        # Find matching categories
        for category, keywords in keyword_mappings.items():
            if any(keyword in message_lower for keyword in keywords):
                models = await self.find_models_for_project(category)
                relevant_models.extend(models[:2])  # Top 2 models per category
        
        return relevant_models
    
    async def install_model(self, model_id: str) -> Dict[str, Any]:
        """Install a specific model"""
        
        if model_id not in self.model_database:
            raise ValueError(f"Unknown model: {model_id}")
        
        model_info = self.model_database[model_id]
        
        try:
            logger.info(f"Installing {model_info['name']}...")
            
            # Execute installation command
            install_cmd = model_info["install_command"]
            process = await asyncio.create_subprocess_shell(
                install_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                self.installed_models.add(model_id)
                logger.info(f"✓ Successfully installed {model_info['name']}")
                
                return {
                    "status": "success",
                    "model_id": model_id,
                    "message": f"{model_info['name']} installed successfully!"
                }
            else:
                error_msg = stderr.decode() if stderr else "Installation failed"
                logger.error(f"Failed to install {model_info['name']}: {error_msg}")
                
                return {
                    "status": "error",
                    "model_id": model_id,
                    "error": error_msg
                }
                
        except Exception as e:
            logger.error(f"Error installing {model_id}: {e}")
            return {
                "status": "error", 
                "model_id": model_id,
                "error": str(e)
            }
    
    async def get_all_available_models(self) -> List[Dict[str, Any]]:
        """Get comprehensive list of all available models"""
        
        all_models = []
        
        for model_id, model_info in self.model_database.items():
            model_data = {
                "id": model_id,
                "name": model_info["name"],
                "category": model_info["category"],
                "description": model_info["description"],
                "size": model_info["size"],
                "capabilities": model_info["capabilities"],
                "status": "installed" if model_id in self.installed_models else "available"
            }
            all_models.append(model_data)
        
        return all_models
    
    async def get_model_categories(self) -> Dict[str, List[str]]:
        """Get model categories and their descriptions"""
        
        category_descriptions = {
            "music": "Generate music, melodies, and audio content",
            "art": "Create images, artwork, and visual designs", 
            "writing": "Generate stories, poems, and written content",
            "video": "Create animations and video content",
            "voice": "Synthesize speech and voice content",
            "code": "Generate and assist with programming code"
        }
        
        return category_descriptions
    
    async def analyze_for_opportunities(self, message: str, response_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze conversation for creative opportunities"""
        
        opportunities = []
        
        # Check if user expressed interest in creating something
        creative_expressions = [
            "i want to make", "let's create", "can you help me build",
            "i'd like to try", "how about we", "let's work on"
        ]
        
        message_lower = message.lower()
        
        if any(expr in message_lower for expr in creative_expressions):
            # Find relevant models
            relevant_models = await self.find_models_for_message(message)
            
            if relevant_models:
                opportunities.append({
                    "type": "creative_collaboration",
                    "suggestion": f"I can help you create that! I have access to {len(relevant_models)} tools that could work for this project.",
                    "models": relevant_models[:3],  # Top 3 suggestions
                    "action": "start_project"
                })
        
        return opportunities
    
    async def get_project_next_steps(self, project: Dict[str, Any]) -> List[str]:
        """Get suggested next steps for a creative project"""
        
        project_type = project["type"]
        
        next_steps = {
            "music": [
                "Tell me what style or mood you're going for",
                "Do you have any reference tracks you like?",
                "Should we start with a melody or rhythm?"
            ],
            "art": [
                "Describe the image you want to create",
                "What style are you thinking? (realistic, cartoon, abstract)",
                "Any specific colors or themes in mind?"
            ],
            "writing": [
                "What genre are you interested in?",
                "Do you have characters or a plot in mind?",
                "How long should the piece be?"
            ],
            "video": [
                "What's the concept for your video?",
                "How long should it be?",
                "Any specific visual style you prefer?"
            ],
            "voice": [
                "What text would you like me to speak?",
                "What emotion or tone should I use?",
                "Male or female voice preference?"
            ],
            "code": [
                "What programming language?",
                "What should the code accomplish?",
                "Any specific requirements or constraints?"
            ]
        }
        
        return next_steps.get(project_type, [
            "Tell me more about what you'd like to create",
            "What's your vision for this project?",
            "How can I best help you with this?"
        ])

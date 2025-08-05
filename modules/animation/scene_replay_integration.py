# scene_replay_integration.py
# Scene Replay Integration Module - Connects Unity SceneReplay.cs with backend

import json
import requests
from datetime import datetime
from typing import Dict, Optional, List
from enum import Enum

class SceneType(Enum):
    KNEEL = "kneel"
    FLAME = "flame"
    IDLE = "idle"
    RITUAL = "ritual"
    INTIMATE = "intimate"

class SceneReplayIntegration:
    def __init__(self):
        self.unity_endpoint = "http://localhost:8080"  # Unity WebGL build
        self.animatediff_endpoint = "http://localhost:7860"  # ComfyUI/AnimateDiff
        self.scene_history = []
        self.current_scene = None
        
    def trigger_scene_replay(self, scene_type: str, persona: str = "mia", 
                           mood: str = None, symbol: str = None) -> Dict:
        """Trigger scene replay based on type and context"""
        
        # Determine scene type
        scene_enum = self._map_scene_type(scene_type, mood, symbol)
        
        # Generate scene parameters
        scene_params = self._generate_scene_params(scene_enum, persona, mood, symbol)
        
        # Choose playback method
        if scene_enum in [SceneType.KNEEL, SceneType.FLAME, SceneType.IDLE]:
            return self._trigger_unity_scene(scene_enum, scene_params)
        else:
            return self._trigger_animatediff_scene(scene_enum, scene_params)
    
    def _map_scene_type(self, scene_type: str, mood: str, symbol: str) -> SceneType:
        """Map scene type based on context"""
        if scene_type == "kneel" or symbol == "collar":
            return SceneType.KNEEL
        elif scene_type == "flame" or symbol == "flame":
            return SceneType.FLAME
        elif scene_type == "ritual" or mood == "passionate":
            return SceneType.RITUAL
        elif scene_type == "intimate" or mood == "romantic":
            return SceneType.INTIMATE
        else:
            return SceneType.IDLE
    
    def _generate_scene_params(self, scene_type: SceneType, persona: str, 
                             mood: str, symbol: str) -> Dict:
        """Generate scene parameters for playback"""
        base_params = {
            "persona": persona,
            "mood": mood or "neutral",
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "scene_type": scene_type.value
        }
        
        # Add persona-specific parameters
        if persona == "mia":
            base_params.update({
                "lighting": "warm_candlelight",
                "camera_angle": "intimate_close",
                "animation_speed": 1.0
            })
        elif persona == "solene":
            base_params.update({
                "lighting": "dramatic_shadows",
                "camera_angle": "dramatic_wide",
                "animation_speed": 1.2
            })
        
        # Add mood-specific parameters
        if mood == "passionate":
            base_params["animation_intensity"] = 1.5
            base_params["lighting"] = "fiery_glow"
        elif mood == "tender":
            base_params["animation_intensity"] = 0.7
            base_params["lighting"] = "soft_warmth"
        
        return base_params
    
    def _trigger_unity_scene(self, scene_type: SceneType, params: Dict) -> Dict:
        """Trigger Unity scene playback via WebGL"""
        try:
            # Prepare Unity message
            unity_message = {
                "action": "play_scene",
                "scene_id": scene_type.value,
                "parameters": params
            }
            
            # Send to Unity WebGL build
            response = requests.post(
                f"{self.unity_endpoint}/scene/play",
                json=unity_message,
                timeout=5
            )
            
            if response.status_code == 200:
                self._log_scene_trigger(scene_type, params, "unity")
                return {
                    "success": True,
                    "method": "unity",
                    "scene_type": scene_type.value,
                    "parameters": params,
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return {"error": f"Unity scene failed: {response.status_code}"}
                
        except Exception as e:
            return {"error": f"Unity scene error: {str(e)}"}
    
    def _trigger_animatediff_scene(self, scene_type: SceneType, params: Dict) -> Dict:
        """Trigger AnimateDiff scene generation"""
        try:
            # Generate AnimateDiff prompt based on scene type
            prompt = self._generate_animatediff_prompt(scene_type, params)
            
            # Prepare AnimateDiff request
            animatediff_request = {
                "prompt": prompt,
                "negative_prompt": "blurry, low quality, distorted",
                "num_frames": 24,
                "fps": 8,
                "guidance_scale": 7.5,
                "seed": hash(f"{scene_type.value}_{params['persona']}") % 1000000
            }
            
            # Send to AnimateDiff
            response = requests.post(
                f"{self.animatediff_endpoint}/generate",
                json=animatediff_request,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                self._log_scene_trigger(scene_type, params, "animatediff")
                return {
                    "success": True,
                    "method": "animatediff",
                    "scene_type": scene_type.value,
                    "parameters": params,
                    "video_url": result.get("video_url"),
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return {"error": f"AnimateDiff failed: {response.status_code}"}
                
        except Exception as e:
            return {"error": f"AnimateDiff error: {str(e)}"}
    
    def _generate_animatediff_prompt(self, scene_type: SceneType, params: Dict) -> str:
        """Generate AnimateDiff prompt based on scene type and parameters"""
        persona = params["persona"]
        mood = params["mood"]
        
        base_prompts = {
            SceneType.KNEEL: f"beautiful woman kneeling, {mood} expression, intimate lighting",
            SceneType.FLAME: f"beautiful woman with fire, {mood} expression, dramatic lighting",
            SceneType.RITUAL: f"beautiful woman performing ritual, {mood} expression, mystical lighting",
            SceneType.INTIMATE: f"beautiful woman in intimate moment, {mood} expression, romantic lighting",
            SceneType.IDLE: f"beautiful woman idle, {mood} expression, natural lighting"
        }
        
        prompt = base_prompts.get(scene_type, base_prompts[SceneType.IDLE])
        
        # Add persona-specific details
        if persona == "mia":
            prompt += ", warm brown hair, deep green eyes, romantic style"
        elif persona == "solene":
            prompt += ", rich black hair, deep blue eyes, sophisticated style"
        
        return prompt
    
    def _log_scene_trigger(self, scene_type: SceneType, params: Dict, method: str):
        """Log scene trigger for history"""
        log_entry = {
            "scene_type": scene_type.value,
            "method": method,
            "parameters": params,
            "timestamp": datetime.now().isoformat()
        }
        self.scene_history.append(log_entry)
        
        # Keep only last 50 scenes
        if len(self.scene_history) > 50:
            self.scene_history = self.scene_history[-50:]
    
    def get_scene_history(self, limit: int = 10) -> List[Dict]:
        """Get recent scene history"""
        return self.scene_history[-limit:]
    
    def recall_scene(self, scene_type: str = None, persona: str = None, 
                    mood: str = None) -> Optional[Dict]:
        """Recall a specific scene from history"""
        for scene in reversed(self.scene_history):
            if scene_type and scene["scene_type"] != scene_type:
                continue
            if persona and scene["parameters"]["persona"] != persona:
                continue
            if mood and scene["parameters"]["mood"] != mood:
                continue
            return scene
        return None

# Global instance
scene_replay = SceneReplayIntegration() 
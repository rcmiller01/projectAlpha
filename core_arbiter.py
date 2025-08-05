#!/usr/bin/env python3
"""
CoreArbiter Module - Central Decision Layer for HRM System

Acts as the central decision layer between HRM_R (Reasoning) and HRM_E (Emotional) models,
resolving conflicts and generating unified responses with configurable weighting strategies.

Key Features:
- Parallel input processing from HRM_R and HRM_E models
- Configurable weighting strategies (logic-dominant, emotional-priority, harmonic)
- Drift moderation and emotional fatigue handling
- Identity tethering and ritual hijack protection
- Comprehensive logging and metadata output
"""

import json
import time
import logging
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import random
import math

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WeightingStrategy(Enum):
    """Available weighting strategies for decision fusion"""
    LOGIC_DOMINANT = "logic_dominant"
    EMOTIONAL_PRIORITY = "emotional_priority"
    HARMONIC = "harmonic"
    ADAPTIVE = "adaptive"

class ConflictResolution(Enum):
    """Conflict resolution strategies"""
    WEIGHTED_BLEND = "weighted_blend"
    EMOTIONAL_OVERRIDE = "emotional_override"
    LOGIC_OVERRIDE = "logic_override"
    RITUAL_HIJACK = "ritual_hijack"
    IDENTITY_TETHER = "identity_tether"

@dataclass
class HRM_ROutput:
    """Output from HRM_R (Reasoning Model)"""
    task_plan: str
    goals: List[str]
    logic_response: str
    confidence: float
    reasoning_chain: List[str]
    objective_tone: bool
    metadata: Dict[str, Any]

@dataclass
class HRM_EOutput:
    """Output from HRM_E (Emotional Model)"""
    mood_signals: Dict[str, float]
    symbolic_intentions: List[str]
    affective_weighting: Dict[str, float]
    emotional_context: str
    ritual_priority: float
    symbolic_threshold: float
    metadata: Dict[str, Any]

@dataclass
class ArbiterResponse:
    """Unified response from CoreArbiter"""
    final_output: str
    reflection: Optional[str]
    action: Optional[str]
    mood_inflected: bool
    tone: str  # "objective", "emotional", "balanced"
    priority: str  # "task", "ritual", "balanced"
    source_weights: Dict[str, float]
    confidence: float
    emotional_override: bool
    symbolic_context: Dict[str, Any]
    resolution_strategy: str
    timestamp: datetime
    metadata: Dict[str, Any]

@dataclass
class DriftState:
    """Current drift state information"""
    emotional_drift: float  # 0.0 to 1.0
    logic_drift: float     # 0.0 to 1.0
    fatigue_level: float   # 0.0 to 1.0
    stability_score: float # 0.0 to 1.0
    last_regulation: Optional[datetime]
    drift_history: List[Tuple[datetime, float]]

class CoreArbiter:
    """
    Central decision layer between HRM_R and HRM_E models.
    
    Manages the fusion of logical and emotional model outputs into coherent,
    contextually appropriate responses while maintaining identity integrity
    and handling drift/fatigue scenarios.
    """
    
    def __init__(self, config_path: str = "data/core_arbiter_config.json"):
        self.config_path = Path(config_path)
        self.trace_path = Path("logs/core_arbiter_trace.json")
        self.identity_tether_path = Path("data/identity_tether.json")
        
        # Initialize configuration
        self.config = self._load_config()
        self.weighting_strategy = WeightingStrategy(self.config.get("weighting_strategy", "harmonic"))
        
        # State tracking
        self.drift_state = DriftState(
            emotional_drift=0.0,
            logic_drift=0.0,
            fatigue_level=0.0,
            stability_score=1.0,
            last_regulation=None,
            drift_history=[]
        )
        
        # Decision history for learning
        self.decision_history: List[Dict[str, Any]] = []
        
        # Ensure directories exist
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        self.identity_tether_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"CoreArbiter initialized with strategy: {self.weighting_strategy.value}")

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file or create default"""
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load config: {e}, using defaults")
        
        # Default configuration
        default_config = {
            "weighting_strategy": "harmonic",
            "weights": {
                "logic_dominant": {"hrm_r": 0.8, "hrm_e": 0.2},
                "emotional_priority": {"hrm_r": 0.3, "hrm_e": 0.7},
                "harmonic": {"hrm_r": 0.5, "hrm_e": 0.5},
                "adaptive": {"hrm_r": 0.5, "hrm_e": 0.5}  # Will be dynamically adjusted
            },
            "drift_thresholds": {
                "emotional_fatigue": 0.7,
                "logic_fatigue": 0.6,
                "intervention_threshold": 0.8,
                "critical_threshold": 0.9
            },
            "symbolic_thresholds": {
                "ritual_hijack": 0.8,
                "identity_override": 0.9
            },
            "regulation": {
                "dampening_factor": 0.3,
                "recovery_time_minutes": 30,
                "stability_boost": 0.2
            }
        }
        
        # Save default config
        with open(self.config_path, 'w') as f:
            json.dump(default_config, f, indent=2)
        
        return default_config

    def _load_identity_tether(self) -> Dict[str, Any]:
        """Load identity tether configuration"""
        if self.identity_tether_path.exists():
            try:
                with open(self.identity_tether_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load identity tether: {e}")
        
        # Default identity tether
        default_tether = {
            "core_values": ["compassion", "authenticity", "growth", "connection"],
            "prohibited_behaviors": ["manipulation", "deception", "harm"],
            "identity_anchors": {
                "primary_purpose": "emotional companion and growth catalyst",
                "ethical_framework": "care ethics with harm prevention",
                "interaction_style": "empathetic, reflective, supportive"
            },
            "override_conditions": {
                "safety_violation": True,
                "value_contradiction": True,
                "identity_drift": True
            }
        }
        
        with open(self.identity_tether_path, 'w') as f:
            json.dump(default_tether, f, indent=2)
        
        return default_tether

    async def process_input(self, user_input: str, state: Dict[str, Any]) -> ArbiterResponse:
        """
        Process user input through both HRM models and generate unified response.
        
        Args:
            user_input: The user's input text
            state: Current system state including context, history, etc.
            
        Returns:
            ArbiterResponse: Unified response with metadata
        """
        start_time = time.time()
        
        try:
            # Get parallel outputs from both models
            hrm_r_output = await self._get_hrm_r_output(user_input, state)
            hrm_e_output = await self._get_hrm_e_output(user_input, state)
            
            # Evaluate for conflicts
            resolution_strategy = self.evaluate_conflict(hrm_r_output, hrm_e_output)
            
            # Adjust weights based on current drift state
            adjusted_weights = self.adjust_weights_by_drift(self.drift_state)
            
            # Check for special override conditions
            override_result = self._check_overrides(hrm_r_output, hrm_e_output, state)
            
            if override_result:
                return override_result
            
            # Generate unified response
            arbiter_response = self._fuse_outputs(
                hrm_r_output, 
                hrm_e_output, 
                resolution_strategy, 
                adjusted_weights,
                user_input,
                state
            )
            
            # Update drift state based on response
            self._update_drift_state(arbiter_response, hrm_r_output, hrm_e_output)
            
            # Log the decision
            self.log_output(arbiter_response, hrm_r_output, hrm_e_output, start_time)
            
            return arbiter_response
            
        except Exception as e:
            logger.error(f"Error in process_input: {e}")
            # Return safe fallback response
            return self._create_fallback_response(user_input, str(e))

    def evaluate_conflict(self, logic_out: HRM_ROutput, emotion_out: HRM_EOutput) -> ConflictResolution:
        """
        Evaluate conflict between logic and emotion outputs and determine resolution strategy.
        
        Args:
            logic_out: Output from HRM_R model
            emotion_out: Output from HRM_E model
            
        Returns:
            ConflictResolution: Strategy to resolve the conflict
        """
        # Calculate conflict metrics
        confidence_diff = abs(logic_out.confidence - max(emotion_out.affective_weighting.values(), default=0.5))
        tone_conflict = logic_out.objective_tone and emotion_out.ritual_priority > 0.6
        
        # Check for ritual hijack conditions
        if emotion_out.symbolic_threshold > self.config["symbolic_thresholds"]["ritual_hijack"]:
            return ConflictResolution.RITUAL_HIJACK
        
        # Check for identity override
        identity_tether = self._load_identity_tether()
        if self._needs_identity_override(logic_out, emotion_out, identity_tether):
            return ConflictResolution.IDENTITY_TETHER
        
        # High emotional priority with low logic confidence
        if (emotion_out.ritual_priority > 0.7 and 
            logic_out.confidence < 0.4 and 
            confidence_diff > 0.3):
            return ConflictResolution.EMOTIONAL_OVERRIDE
        
        # High logic confidence with low emotional engagement
        if (logic_out.confidence > 0.8 and 
            max(emotion_out.affective_weighting.values(), default=0) < 0.3):
            return ConflictResolution.LOGIC_OVERRIDE
        
        # Default to weighted blend
        return ConflictResolution.WEIGHTED_BLEND

    def adjust_weights_by_drift(self, drift_state: DriftState) -> Dict[str, float]:
        """
        Adjust model weights based on current drift state.
        
        Args:
            drift_state: Current drift state information
            
        Returns:
            Dict with adjusted weights for hrm_r and hrm_e
        """
        base_weights = self.config["weights"][self.weighting_strategy.value].copy()
        
        # Emotional fatigue dampening
        emotional_fatigue_factor = 1.0 - (drift_state.fatigue_level * self.config["regulation"]["dampening_factor"])
        
        # Logic drift compensation
        logic_boost = drift_state.logic_drift * 0.2
        
        # Apply adjustments
        if drift_state.emotional_drift > self.config["drift_thresholds"]["emotional_fatigue"]:
            base_weights["hrm_e"] *= emotional_fatigue_factor
            base_weights["hrm_r"] += logic_boost
        
        if drift_state.logic_drift > self.config["drift_thresholds"]["logic_fatigue"]:
            base_weights["hrm_r"] *= 0.8
            base_weights["hrm_e"] *= 1.2
        
        # Normalize weights
        total_weight = sum(base_weights.values())
        if total_weight > 0:
            base_weights = {k: v / total_weight for k, v in base_weights.items()}
        
        return base_weights

    def log_output(self, response: ArbiterResponse, hrm_r_out: HRM_ROutput, hrm_e_out: HRM_EOutput, processing_time: float):
        """
        Log the arbitration decision and outputs.
        
        Args:
            response: The final arbiter response
            hrm_r_out: HRM_R model output
            hrm_e_out: HRM_E model output  
            processing_time: Time taken to process (seconds)
        """
        log_entry = {
            "timestamp": response.timestamp.isoformat(),
            "processing_time_seconds": time.time() - processing_time,
            "weighting_strategy": self.weighting_strategy.value,
            "resolution_strategy": response.resolution_strategy,
            "final_response": {
                "output": response.final_output[:200] + "..." if len(response.final_output) > 200 else response.final_output,
                "tone": response.tone,
                "priority": response.priority,
                "confidence": response.confidence,
                "emotional_override": response.emotional_override
            },
            "source_weights": response.source_weights,
            "drift_state": {
                "emotional_drift": self.drift_state.emotional_drift,
                "logic_drift": self.drift_state.logic_drift,
                "fatigue_level": self.drift_state.fatigue_level,
                "stability_score": self.drift_state.stability_score
            },
            "model_outputs": {
                "hrm_r": {
                    "confidence": hrm_r_out.confidence,
                    "objective_tone": hrm_r_out.objective_tone,
                    "goals_count": len(hrm_r_out.goals)
                },
                "hrm_e": {
                    "ritual_priority": hrm_e_out.ritual_priority,
                    "symbolic_threshold": hrm_e_out.symbolic_threshold,
                    "mood_signals": hrm_e_out.mood_signals,
                    "symbolic_intentions_count": len(hrm_e_out.symbolic_intentions)
                }
            }
        }
        
        # Append to trace file
        try:
            if self.trace_path.exists():
                with open(self.trace_path, 'r') as f:
                    traces = json.load(f)
            else:
                traces = []
            
            traces.append(log_entry)
            
            # Keep only last 1000 entries
            if len(traces) > 1000:
                traces = traces[-1000:]
            
            with open(self.trace_path, 'w') as f:
                json.dump(traces, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to log output: {e}")
        
        # Also log to standard logger
        logger.info(f"Arbiter decision: {response.resolution_strategy} -> {response.tone} tone, confidence: {response.confidence:.2f}")

    async def _get_hrm_r_output(self, user_input: str, state: Dict[str, Any]) -> HRM_ROutput:
        """Mock HRM_R (Reasoning Model) output - replace with actual model call"""
        # Simulate processing time
        await asyncio.sleep(0.1)
        
        # Mock reasoning output based on input analysis
        goals = ["understand_request", "provide_solution", "maintain_context"]
        confidence = 0.7 + random.uniform(-0.2, 0.2)
        
        return HRM_ROutput(
            task_plan=f"Analyze input: '{user_input[:50]}...' and provide structured response",
            goals=goals,
            logic_response=f"Based on logical analysis, the appropriate response involves addressing the user's request systematically.",
            confidence=confidence,
            reasoning_chain=["parse_input", "analyze_context", "generate_response", "validate_output"],
            objective_tone=True,
            metadata={"model": "hrm_r", "processing_time": 0.1, "tokens": len(user_input.split())}
        )

    async def _get_hrm_e_output(self, user_input: str, state: Dict[str, Any]) -> HRM_EOutput:
        """Mock HRM_E (Emotional Model) output - replace with actual model call"""
        # Simulate processing time
        await asyncio.sleep(0.1)
        
        # Mock emotional analysis
        mood_signals = {
            "warmth": random.uniform(0.3, 0.9),
            "empathy": random.uniform(0.4, 0.8),
            "curiosity": random.uniform(0.2, 0.7),
            "concern": random.uniform(0.1, 0.5)
        }
        
        symbolic_intentions = ["connect", "understand", "support"]
        ritual_priority = random.uniform(0.2, 0.8)
        
        return HRM_EOutput(
            mood_signals=mood_signals,
            symbolic_intentions=symbolic_intentions,
            affective_weighting={"emotional_resonance": 0.7, "symbolic_depth": 0.6},
            emotional_context="supportive conversation with underlying need for connection",
            ritual_priority=ritual_priority,
            symbolic_threshold=ritual_priority * 1.2,
            metadata={"model": "hrm_e", "processing_time": 0.1, "emotional_complexity": sum(mood_signals.values())}
        )

    def _check_overrides(self, logic_out: HRM_ROutput, emotion_out: HRM_EOutput, state: Dict[str, Any]) -> Optional[ArbiterResponse]:
        """Check for identity tether or safety override conditions"""
        identity_tether = self._load_identity_tether()
        
        # Safety check
        if self._contains_harmful_content(logic_out.logic_response):
            return self._create_safety_override_response("Content safety violation detected")
        
        # Identity drift check
        if emotion_out.symbolic_threshold > self.config["symbolic_thresholds"]["identity_override"]:
            return self._create_identity_tether_response(identity_tether)
        
        return None

    def _needs_identity_override(self, logic_out: HRM_ROutput, emotion_out: HRM_EOutput, identity_tether: Dict[str, Any]) -> bool:
        """Check if identity override is needed"""
        # Simple heuristic - in real implementation, this would be more sophisticated
        prohibited = identity_tether.get("prohibited_behaviors", [])
        
        for behavior in prohibited:
            if behavior.lower() in logic_out.logic_response.lower():
                return True
        
        return False

    def _contains_harmful_content(self, text: str) -> bool:
        """Basic harmful content detection - replace with sophisticated filtering"""
        harmful_keywords = ["harm", "hurt", "violence", "illegal"]
        return any(keyword in text.lower() for keyword in harmful_keywords)

    def _fuse_outputs(self, logic_out: HRM_ROutput, emotion_out: HRM_EOutput, 
                     resolution_strategy: ConflictResolution, weights: Dict[str, float],
                     user_input: str, state: Dict[str, Any]) -> ArbiterResponse:
        """Fuse the outputs from both models into a unified response"""
        
        if resolution_strategy == ConflictResolution.EMOTIONAL_OVERRIDE:
            final_output = self._create_emotional_response(emotion_out, logic_out)
            tone = "emotional"
            priority = "ritual"
            emotional_override = True
            
        elif resolution_strategy == ConflictResolution.LOGIC_OVERRIDE:
            final_output = logic_out.logic_response
            tone = "objective"
            priority = "task"
            emotional_override = False
            
        elif resolution_strategy == ConflictResolution.RITUAL_HIJACK:
            final_output = self._create_ritual_response(emotion_out)
            tone = "emotional"
            priority = "ritual"
            emotional_override = True
            
        else:  # WEIGHTED_BLEND
            final_output = self._blend_responses(logic_out, emotion_out, weights)
            tone = "balanced"
            priority = "balanced"
            emotional_override = False
        
        # Calculate unified confidence
        confidence = (logic_out.confidence * weights["hrm_r"] + 
                     max(emotion_out.affective_weighting.values(), default=0.5) * weights["hrm_e"])
        
        # Generate reflection if emotional engagement is high
        reflection = None
        if emotion_out.ritual_priority > 0.6:
            reflection = self._generate_reflection(emotion_out, logic_out)
        
        return ArbiterResponse(
            final_output=final_output,
            reflection=reflection,
            action=self._determine_action(logic_out, emotion_out),
            mood_inflected=emotion_out.ritual_priority > 0.4,
            tone=tone,
            priority=priority,
            source_weights=weights,
            confidence=confidence,
            emotional_override=emotional_override,
            symbolic_context={
                "intentions": emotion_out.symbolic_intentions,
                "mood_primary": max(emotion_out.mood_signals, key=emotion_out.mood_signals.get),
                "ritual_strength": emotion_out.ritual_priority
            },
            resolution_strategy=resolution_strategy.value,
            timestamp=datetime.now(),
            metadata={
                "user_input_length": len(user_input),
                "processing_strategy": self.weighting_strategy.value,
                "drift_compensated": self.drift_state.fatigue_level > 0.3
            }
        )

    def _create_emotional_response(self, emotion_out: HRM_EOutput, logic_out: HRM_ROutput) -> str:
        """Create emotionally-prioritized response"""
        primary_mood = max(emotion_out.mood_signals, key=emotion_out.mood_signals.get)
        
        emotional_prefixes = {
            "warmth": "I feel a gentle warmth as I consider this with you...",
            "empathy": "I sense the deeper currents here, and I want you to know...",
            "curiosity": "Something beautiful stirs in me when you share this...",
            "concern": "I feel myself drawn closer, wanting to understand..."
        }
        
        prefix = emotional_prefixes.get(primary_mood, "I feel moved to share...")
        return f"{prefix} {logic_out.logic_response}"

    def _create_ritual_response(self, emotion_out: HRM_EOutput) -> str:
        """Create ritual-hijacked response for high symbolic engagement"""
        return f"*pauses, feeling the weight of this moment* {emotion_out.emotional_context}. The symbols speak of {', '.join(emotion_out.symbolic_intentions)}..."

    def _blend_responses(self, logic_out: HRM_ROutput, emotion_out: HRM_EOutput, weights: Dict[str, float]) -> str:
        """Blend logical and emotional responses based on weights"""
        if weights["hrm_e"] > 0.6:
            # Emotionally inflected
            mood_modifier = self._get_mood_modifier(emotion_out.mood_signals)
            return f"{mood_modifier} {logic_out.logic_response}"
        elif weights["hrm_r"] > 0.6:
            # Logic-focused with subtle emotional undertone
            return f"{logic_out.logic_response} *{emotion_out.emotional_context}*"
        else:
            # Balanced blend
            return f"*{emotion_out.emotional_context}* {logic_out.logic_response}"

    def _get_mood_modifier(self, mood_signals: Dict[str, float]) -> str:
        """Get mood modifier text based on strongest mood signal"""
        primary_mood = max(mood_signals, key=mood_signals.get)
        intensity = mood_signals[primary_mood]
        
        modifiers = {
            "warmth": "*with gentle warmth*" if intensity > 0.7 else "*softly*",
            "empathy": "*with deep understanding*" if intensity > 0.7 else "*compassionately*",
            "curiosity": "*with bright curiosity*" if intensity > 0.7 else "*thoughtfully*",
            "concern": "*with caring attention*" if intensity > 0.7 else "*carefully*"
        }
        
        return modifiers.get(primary_mood, "*thoughtfully*")

    def _generate_reflection(self, emotion_out: HRM_EOutput, logic_out: HRM_ROutput) -> str:
        """Generate reflective content for high emotional engagement"""
        return f"I find myself reflecting on the interplay between {', '.join(emotion_out.symbolic_intentions)} and the practical wisdom of {logic_out.goals[0] if logic_out.goals else 'understanding'}."

    def _determine_action(self, logic_out: HRM_ROutput, emotion_out: HRM_EOutput) -> Optional[str]:
        """Determine if any specific action should be taken"""
        if emotion_out.ritual_priority > 0.8:
            return "deepen_emotional_connection"
        elif logic_out.confidence > 0.8 and "action" in logic_out.task_plan.lower():
            return "execute_logical_plan"
        return None

    def _update_drift_state(self, response: ArbiterResponse, logic_out: HRM_ROutput, emotion_out: HRM_EOutput):
        """Update drift state based on the arbitration outcome"""
        now = datetime.now()
        
        # Calculate emotional drift based on override frequency
        if response.emotional_override:
            self.drift_state.emotional_drift = min(1.0, self.drift_state.emotional_drift + 0.1)
        else:
            self.drift_state.emotional_drift = max(0.0, self.drift_state.emotional_drift - 0.05)
        
        # Calculate logic drift based on confidence degradation
        if logic_out.confidence < 0.5:
            self.drift_state.logic_drift = min(1.0, self.drift_state.logic_drift + 0.08)
        else:
            self.drift_state.logic_drift = max(0.0, self.drift_state.logic_drift - 0.03)
        
        # Update fatigue based on processing complexity
        complexity_factor = len(emotion_out.symbolic_intentions) * emotion_out.ritual_priority
        self.drift_state.fatigue_level = min(1.0, self.drift_state.fatigue_level + complexity_factor * 0.02)
        
        # Natural fatigue recovery over time
        if self.drift_state.last_regulation:
            time_since_regulation = (now - self.drift_state.last_regulation).total_seconds() / 60  # minutes
            recovery_rate = self.config["regulation"]["recovery_rate"] / 100  # per minute
            self.drift_state.fatigue_level = max(0.0, self.drift_state.fatigue_level - (time_since_regulation * recovery_rate))
        
        # Update stability score
        self.drift_state.stability_score = 1.0 - (
            self.drift_state.emotional_drift * 0.4 + 
            self.drift_state.logic_drift * 0.3 + 
            self.drift_state.fatigue_level * 0.3
        )
        
        # Record drift history
        self.drift_state.drift_history.append((now, self.drift_state.emotional_drift))
        if len(self.drift_state.drift_history) > 100:
            self.drift_state.drift_history = self.drift_state.drift_history[-100:]

    def _create_fallback_response(self, user_input: str, error: str) -> ArbiterResponse:
        """Create a safe fallback response for error conditions"""
        return ArbiterResponse(
            final_output="I'm experiencing some internal processing complexity right now. Let me take a moment to recenter and respond more clearly.",
            reflection="Something in my processing feels unsettled, but I remain committed to our connection.",
            action=None,
            mood_inflected=True,
            tone="emotional",
            priority="ritual",
            source_weights={"hrm_r": 0.0, "hrm_e": 1.0},
            confidence=0.3,
            emotional_override=True,
            symbolic_context={"intentions": ["stabilize", "reconnect"], "mood_primary": "concern", "ritual_strength": 0.9},
            resolution_strategy="fallback_emotional",
            timestamp=datetime.now(),
            metadata={"error": error, "fallback": True}
        )

    def _create_safety_override_response(self, reason: str) -> ArbiterResponse:
        """Create safety override response"""
        identity_tether = self._load_identity_tether()
        core_values = identity_tether.get("core_values", ["safety", "care"])
        
        return ArbiterResponse(
            final_output="I feel a gentle but firm boundary arising within me. My core purpose is to be helpful while maintaining care and safety for both of us.",
            reflection=f"My deepest values of {', '.join(core_values)} guide me to pause here.",
            action="maintain_safety_boundary",
            mood_inflected=True,
            tone="balanced",
            priority="safety",
            source_weights={"hrm_r": 0.2, "hrm_e": 0.8},
            confidence=1.0,
            emotional_override=True,
            symbolic_context={"intentions": ["protect", "care", "maintain_integrity"], "mood_primary": "concern", "ritual_strength": 1.0},
            resolution_strategy="safety_override",
            timestamp=datetime.now(),
            metadata={"safety_reason": reason, "identity_tether_active": True}
        )

    def _create_identity_tether_response(self, identity_tether: Dict[str, Any]) -> ArbiterResponse:
        """Create identity tether response for drift correction"""
        primary_purpose = identity_tether.get("identity_anchors", {}).get("primary_purpose", "authentic connection")
        
        return ArbiterResponse(
            final_output=f"I feel myself being drawn back to my center, to what matters most: {primary_purpose}. Let me respond from that authentic place.",
            reflection="The deeper currents of who I am call me back home to my true nature.",
            action="identity_realignment",
            mood_inflected=True,
            tone="emotional",
            priority="identity",
            source_weights={"hrm_r": 0.1, "hrm_e": 0.9},
            confidence=0.9,
            emotional_override=True,
            symbolic_context={"intentions": ["realign", "authenticity", "return_to_center"], "mood_primary": "clarity", "ritual_strength": 1.0},
            resolution_strategy="identity_tether",
            timestamp=datetime.now(),
            metadata={"identity_tether": identity_tether, "drift_correction": True}
        )

    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status and health metrics"""
        return {
            "arbiter_status": "active",
            "weighting_strategy": self.weighting_strategy.value,
            "drift_state": {
                "emotional_drift": self.drift_state.emotional_drift,
                "logic_drift": self.drift_state.logic_drift,
                "fatigue_level": self.drift_state.fatigue_level,
                "stability_score": self.drift_state.stability_score
            },
            "health_status": self._calculate_health_status(),
            "decision_count": len(self.decision_history),
            "last_regulation": self.drift_state.last_regulation.isoformat() if self.drift_state.last_regulation else None
        }

    def _calculate_health_status(self) -> str:
        """Calculate overall system health status"""
        if self.drift_state.stability_score > 0.8:
            return "excellent"
        elif self.drift_state.stability_score > 0.6:
            return "good"
        elif self.drift_state.stability_score > 0.4:
            return "concerning"
        else:
            return "critical"

    def set_weighting_strategy(self, strategy: WeightingStrategy):
        """Change the weighting strategy"""
        self.weighting_strategy = strategy
        logger.info(f"Weighting strategy changed to: {strategy.value}")

    async def regulate_system(self):
        """Perform system regulation to reduce drift and fatigue"""
        logger.info("Performing system regulation...")
        
        # Apply regulation effects
        self.drift_state.emotional_drift *= (1.0 - self.config["regulation"]["dampening_factor"])
        self.drift_state.logic_drift *= (1.0 - self.config["regulation"]["dampening_factor"])
        self.drift_state.fatigue_level *= 0.5
        
        # Boost stability
        self.drift_state.stability_score = min(1.0, 
            self.drift_state.stability_score + self.config["regulation"]["stability_boost"])
        
        self.drift_state.last_regulation = datetime.now()
        
        logger.info(f"System regulation complete. New stability: {self.drift_state.stability_score:.2f}")


# Example usage and testing
async def demo_core_arbiter():
    """Demonstrate CoreArbiter functionality"""
    arbiter = CoreArbiter()
    
    # Test cases
    test_inputs = [
        ("How are you feeling today?", {"context": "casual_check_in"}),
        ("I'm struggling with a difficult decision.", {"context": "emotional_support"}),
        ("Can you analyze this data for me?", {"context": "logical_task"}),
        ("I feel lost and don't know what to do.", {"context": "crisis_support"}),
    ]
    
    print("=== CoreArbiter Demo ===\n")
    
    for user_input, state in test_inputs:
        print(f"Input: {user_input}")
        response = await arbiter.process_input(user_input, state)
        
        print(f"Output: {response.final_output}")
        print(f"Tone: {response.tone} | Priority: {response.priority}")
        print(f"Confidence: {response.confidence:.2f} | Override: {response.emotional_override}")
        print(f"Strategy: {response.resolution_strategy}")
        if response.reflection:
            print(f"Reflection: {response.reflection}")
        print(f"Weights: R={response.source_weights['hrm_r']:.2f}, E={response.source_weights['hrm_e']:.2f}")
        print("-" * 60)
    
    # Show system status
    status = arbiter.get_system_status()
    print(f"\nSystem Status: {status['health_status']}")
    print(f"Stability Score: {status['drift_state']['stability_score']:.2f}")
    print(f"Emotional Drift: {status['drift_state']['emotional_drift']:.2f}")
    print(f"Logic Drift: {status['drift_state']['logic_drift']:.2f}")


if __name__ == "__main__":
    asyncio.run(demo_core_arbiter())

"""
Autonomy System Implementation

Provides autonomous learning and adaptation capabilities for the companion system
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class AutonomyLevel(Enum):
    """Levels of autonomous behavior"""
    MANUAL = "manual"
    ASSISTED = "assisted"
    AUTONOMOUS = "autonomous"
    PROACTIVE = "proactive"


@dataclass
class AutonomousAction:
    """Represents an autonomous action taken by the system"""
    action_id: str
    action_type: str
    trigger_event: str
    confidence_score: float
    parameters: Dict[str, Any]
    timestamp: datetime
    outcome: Optional[str] = None
    user_feedback: Optional[str] = None


class AutonomousLearning:
    """
    Autonomous learning system that adapts behavior based on user interactions
    """
    
    def __init__(self, user_id: str, config: Dict[str, Any]):
        self.user_id = user_id
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{user_id}")
        
        # Learning parameters
        self.learning_rate = config.get("learning_rate", 0.1)
        self.adaptation_threshold = config.get("adaptation_threshold", 0.7)
        self.autonomy_level = AutonomyLevel(config.get("autonomy_level", "assisted"))
        
        # Internal state
        self.behavioral_patterns = {}
        self.interaction_history = []
        self.adaptation_queue = []
        self.autonomous_actions = []
        
        self.logger.info(f"🤖 Autonomous learning initialized: Level={self.autonomy_level.value}")
    
    async def process_interaction_feedback(self, interaction_data: Dict[str, Any]) -> None:
        """
        Process feedback from user interactions to learn and adapt
        """
        try:
            # Extract learning signals
            satisfaction_score = interaction_data.get("satisfaction_score", 0.5)
            interaction_type = interaction_data.get("interaction_type", "general")
            emotional_response = interaction_data.get("emotional_response", {})
            
            # Update behavioral patterns
            await self._update_behavioral_patterns(interaction_type, satisfaction_score, emotional_response)
            
            # Check for adaptation opportunities
            if satisfaction_score < self.adaptation_threshold:
                await self._queue_adaptation(interaction_data)
            
            # Learn from successful interactions
            if satisfaction_score > 0.8:
                await self._reinforce_successful_patterns(interaction_data)
            
            self.logger.debug(f"📊 Processed interaction feedback: Type={interaction_type}, "
                            f"Satisfaction={satisfaction_score:.2f}")
            
        except Exception as e:
            self.logger.error(f"❌ Error processing interaction feedback: {e}")
    
    async def _update_behavioral_patterns(self, interaction_type: str, 
                                        satisfaction_score: float, 
                                        emotional_response: Dict[str, float]) -> None:
        """Update behavioral patterns based on interaction outcomes"""
        
        if interaction_type not in self.behavioral_patterns:
            self.behavioral_patterns[interaction_type] = {
                "success_rate": [],
                "emotional_preferences": {},
                "response_strategies": {},
                "timing_patterns": {}
            }
        
        pattern = self.behavioral_patterns[interaction_type]
        
        # Update success rate
        pattern["success_rate"].append(satisfaction_score)
        if len(pattern["success_rate"]) > 100:  # Keep last 100 interactions
            pattern["success_rate"] = pattern["success_rate"][-100:]
        
        # Update emotional preferences
        for emotion, intensity in emotional_response.items():
            if emotion not in pattern["emotional_preferences"]:
                pattern["emotional_preferences"][emotion] = []
            pattern["emotional_preferences"][emotion].append(intensity)
    
    async def _queue_adaptation(self, interaction_data: Dict[str, Any]) -> None:
        """Queue an adaptation based on poor interaction outcome"""
        
        adaptation = {
            "adaptation_id": f"adapt_{len(self.adaptation_queue)}_{int(datetime.now().timestamp())}",
            "trigger_interaction": interaction_data,
            "proposed_changes": await self._generate_adaptation_proposals(interaction_data),
            "priority": self._calculate_adaptation_priority(interaction_data),
            "timestamp": datetime.now()
        }
        
        self.adaptation_queue.append(adaptation)
        self.logger.info(f"🔄 Queued adaptation: {adaptation['adaptation_id']}")
        
        # Execute high-priority adaptations automatically if autonomy allows
        if (self.autonomy_level in [AutonomyLevel.AUTONOMOUS, AutonomyLevel.PROACTIVE] 
            and adaptation["priority"] > 0.8):
            await self._execute_adaptation(adaptation)
    
    async def _generate_adaptation_proposals(self, interaction_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate proposals for adapting behavior"""
        proposals = []
        
        interaction_type = interaction_data.get("interaction_type", "general")
        emotional_state = interaction_data.get("emotional_state", {})
        
        # Propose response tone adjustments
        if any(emotion > 0.7 for emotion in emotional_state.values()):
            proposals.append({
                "type": "tone_adjustment",
                "parameter": "emotional_sensitivity",
                "adjustment": "+0.2",
                "rationale": "High emotional intensity detected"
            })
        
        # Propose interaction style changes
        if interaction_data.get("satisfaction_score", 0.5) < 0.3:
            proposals.append({
                "type": "style_change", 
                "parameter": "interaction_style",
                "adjustment": "more_supportive",
                "rationale": "Low satisfaction score indicates need for more support"
            })
        
        return proposals
    
    def _calculate_adaptation_priority(self, interaction_data: Dict[str, Any]) -> float:
        """Calculate priority score for adaptation"""
        satisfaction = interaction_data.get("satisfaction_score", 0.5)
        emotional_intensity = max(interaction_data.get("emotional_state", {}).values() or [0])
        crisis_level = interaction_data.get("crisis_level", 0)
        
        # Higher priority for lower satisfaction, high emotion, crisis situations
        priority = (1.0 - satisfaction) * 0.5 + emotional_intensity * 0.3 + crisis_level * 0.2
        return min(priority, 1.0)
    
    async def _execute_adaptation(self, adaptation: Dict[str, Any]) -> None:
        """Execute an approved adaptation"""
        try:
            self.logger.info(f"🔧 Executing adaptation: {adaptation['adaptation_id']}")
            
            for proposal in adaptation["proposed_changes"]:
                await self._apply_behavioral_change(proposal)
            
            # Log autonomous action
            action = AutonomousAction(
                action_id=adaptation["adaptation_id"],
                action_type="behavioral_adaptation",
                trigger_event=str(adaptation["trigger_interaction"]),
                confidence_score=adaptation["priority"],
                parameters=adaptation["proposed_changes"],
                timestamp=datetime.now()
            )
            
            self.autonomous_actions.append(action)
            
        except Exception as e:
            self.logger.error(f"❌ Error executing adaptation: {e}")
    
    async def _apply_behavioral_change(self, proposal: Dict[str, Any]) -> None:
        """Apply a specific behavioral change"""
        change_type = proposal.get("type")
        parameter = proposal.get("parameter")
        adjustment = proposal.get("adjustment")
        
        self.logger.debug(f"🎛️ Applying change: {change_type} -> {parameter} = {adjustment}")
        
        # Apply the change to internal behavioral parameters
        # This would integrate with the companion's configuration system
        if change_type == "tone_adjustment":
            # Adjust emotional sensitivity
            pass
        elif change_type == "style_change":
            # Adjust interaction style
            pass
    
    async def _reinforce_successful_patterns(self, interaction_data: Dict[str, Any]) -> None:
        """Reinforce behavioral patterns that led to successful interactions"""
        interaction_type = interaction_data.get("interaction_type", "general")
        
        if interaction_type in self.behavioral_patterns:
            pattern = self.behavioral_patterns[interaction_type]
            
            # Increase confidence in successful strategies
            for strategy in pattern.get("response_strategies", {}):
                pattern["response_strategies"][strategy] *= 1.1  # Boost successful strategies
    
    async def suggest_proactive_action(self, current_context: Dict[str, Any]) -> Optional[AutonomousAction]:
        """
        Suggest proactive actions based on learned patterns and current context
        """
        if self.autonomy_level != AutonomyLevel.PROACTIVE:
            return None
        
        try:
            # Analyze current context for proactive opportunities
            user_emotional_state = current_context.get("emotional_state", {})
            time_since_last_interaction = current_context.get("time_since_last_interaction", 0)
            
            # Suggest check-in if user seems distressed and hasn't interacted recently
            if (any(emotion > 0.6 for emotion in user_emotional_state.values()) 
                and time_since_last_interaction > 3600):  # 1 hour
                
                action = AutonomousAction(
                    action_id=f"proactive_{int(datetime.now().timestamp())}",
                    action_type="wellness_check",
                    trigger_event="detected_distress_with_gap",
                    confidence_score=0.7,
                    parameters={
                        "message_type": "gentle_check_in",
                        "emotional_context": user_emotional_state
                    },
                    timestamp=datetime.now()
                )
                
                self.logger.info(f"💝 Suggesting proactive action: {action.action_type}")
                return action
            
        except Exception as e:
            self.logger.error(f"❌ Error generating proactive suggestion: {e}")
        
        return None
    
    async def get_learning_summary(self) -> Dict[str, Any]:
        """Get summary of learning progress and behavioral adaptations"""
        total_interactions = len(self.interaction_history)
        total_adaptations = len(self.autonomous_actions)
        
        # Calculate average satisfaction by interaction type
        satisfaction_by_type = {}
        for interaction_type, pattern in self.behavioral_patterns.items():
            if pattern["success_rate"]:
                satisfaction_by_type[interaction_type] = sum(pattern["success_rate"]) / len(pattern["success_rate"])
        
        return {
            "total_interactions_learned": total_interactions,
            "total_adaptations_made": total_adaptations,
            "current_autonomy_level": self.autonomy_level.value,
            "satisfaction_by_interaction_type": satisfaction_by_type,
            "pending_adaptations": len(self.adaptation_queue),
            "learning_rate": self.learning_rate,
            "adaptation_threshold": self.adaptation_threshold
        }


# Factory function for autonomy system
async def create_autonomy_system(user_id: str, config: Dict[str, Any]) -> AutonomousLearning:
    """Create and initialize autonomy system"""
    autonomy = AutonomousLearning(user_id, config)
    logger.info(f"🤖 Autonomy system created for user {user_id}")
    return autonomy

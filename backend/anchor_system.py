#!/usr/bin/env python3
"""
Anchor System - Safety and Approval Mechanism

This module implements the Anchor system that prevents uncontrolled autopilot
actions by requiring confirmation before external or core memory changes.
"""

import logging
import time
from typing import Dict, Any, List, Optional
from enum import Enum

logger = logging.getLogger(__name__)

class ActionType(Enum):
    """Types of actions that require anchor approval"""
    MEMORY_WRITE = "memory_write"
    MEMORY_DELETE = "memory_delete"
    EXTERNAL_API = "external_api"
    SYSTEM_CONFIG = "system_config"
    EMOTIONAL_STATE = "emotional_state"

class AnchorResponse(Enum):
    """Anchor response types"""
    APPROVED = "approved"
    DENIED = "denied"
    PENDING = "pending"
    TIMEOUT = "timeout"

class AnchorSystem:
    """
    Anchor system for controlling autopilot actions.
    
    Provides approval mechanism for potentially dangerous or
    impactful actions before they are executed.
    """
    
    def __init__(self, timeout_seconds: int = 30):
        """
        Initialize the Anchor system.
        
        Args:
            timeout_seconds (int): Timeout for pending approvals
        """
        self.timeout_seconds = timeout_seconds
        self.pending_actions = {}
        self.approval_history = []
        
    def confirm(self, autopilot_action: Dict[str, Any]) -> AnchorResponse:
        """
        Confirm whether an autopilot action should be allowed.
        
        Args:
            autopilot_action (Dict): Action details including type, target, data
            
        Returns:
            AnchorResponse: The approval decision
        """
        action_id = self._generate_action_id()
        action_type = ActionType(autopilot_action.get("type", "unknown"))
        
        logger.info(f"Anchor evaluating action {action_id}: {action_type.value}")
        
        # Evaluate action safety
        safety_score = self._evaluate_safety(autopilot_action)
        
        if safety_score > 0.8:  # High safety threshold
            response = AnchorResponse.APPROVED
            logger.info(f"Action {action_id} approved (safety: {safety_score:.2f})")
        elif safety_score > 0.5:  # Medium safety - requires review
            response = AnchorResponse.PENDING
            self.pending_actions[action_id] = {
                "action": autopilot_action,
                "timestamp": time.time(),
                "safety_score": safety_score
            }
            logger.warning(f"Action {action_id} pending review (safety: {safety_score:.2f})")
        else:  # Low safety - denied
            response = AnchorResponse.DENIED
            logger.error(f"Action {action_id} denied (safety: {safety_score:.2f})")
        
        # Record in history
        self.approval_history.append({
            "action_id": action_id,
            "action_type": action_type.value,
            "response": response.value,
            "safety_score": safety_score,
            "timestamp": time.time()
        })
        
        return response
    
    def _evaluate_safety(self, action: Dict[str, Any]) -> float:
        """
        Evaluate the safety score of an action.
        
        Args:
            action (Dict): Action to evaluate
            
        Returns:
            float: Safety score between 0 and 1
        """
        action_type = action.get("type", "unknown")
        target = action.get("target", "")
        
        # Base safety scores by action type
        base_scores = {
            "memory_write": 0.7,
            "memory_delete": 0.3,  # More dangerous
            "external_api": 0.5,
            "system_config": 0.2,  # Very dangerous
            "emotional_state": 0.8
        }
        
        safety_score = base_scores.get(action_type, 0.5)
        
        # Adjust based on target sensitivity
        if "identity" in target.lower():
            safety_score *= 0.5  # Identity changes are risky
        elif "core" in target.lower():
            safety_score *= 0.6  # Core system changes
        elif "temp" in target.lower() or "cache" in target.lower():
            safety_score *= 1.2  # Temporary changes are safer
            
        return min(1.0, safety_score)
    
    def _generate_action_id(self) -> str:
        """Generate a unique action ID"""
        return f"anchor_{int(time.time() * 1000)}"
    
    def review_pending_actions(self) -> List[Dict[str, Any]]:
        """
        Review and clean up pending actions.
        
        Returns:
            List[Dict]: List of pending actions that haven't timed out
        """
        current_time = time.time()
        expired_actions = []
        
        for action_id, action_data in self.pending_actions.items():
            if current_time - action_data["timestamp"] > self.timeout_seconds:
                expired_actions.append(action_id)
                logger.warning(f"Action {action_id} timed out")
        
        # Remove expired actions
        for action_id in expired_actions:
            del self.pending_actions[action_id]
        
        return list(self.pending_actions.values())
    
    def approve_pending_action(self, action_id: str) -> bool:
        """
        Manually approve a pending action.
        
        Args:
            action_id (str): ID of the action to approve
            
        Returns:
            bool: True if action was found and approved
        """
        if action_id in self.pending_actions:
            del self.pending_actions[action_id]
            logger.info(f"Manually approved action {action_id}")
            return True
        return False
    
    def get_approval_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get recent approval history.
        
        Args:
            limit (int): Maximum number of records to return
            
        Returns:
            List[Dict]: Recent approval history
        """
        return self.approval_history[-limit:]

# Global anchor instance
anchor = AnchorSystem()

def require_anchor_approval(action_type: str, target: str = "", data: Optional[Dict] = None) -> bool:
    """
    Decorator function to require anchor approval for actions.
    
    Args:
        action_type (str): Type of action being performed
        target (str): Target of the action
        data (Dict): Additional action data
        
    Returns:
        bool: True if action is approved, False otherwise
    """
    action = {
        "type": action_type,
        "target": target,
        "data": data or {}
    }
    
    response = anchor.confirm(action)
    return response == AnchorResponse.APPROVED

if __name__ == "__main__":
    # Test the anchor system
    test_actions = [
        {"type": "memory_write", "target": "temp_cache", "data": {"key": "test"}},
        {"type": "memory_delete", "target": "core_identity", "data": {"key": "personality"}},
        {"type": "external_api", "target": "weather_service", "data": {"endpoint": "/current"}},
    ]
    
    for action in test_actions:
        response = anchor.confirm(action)
        print(f"Action {action['type']} -> {response.value}")

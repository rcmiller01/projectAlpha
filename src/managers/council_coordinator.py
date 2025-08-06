#!/usr/bin/env python3
"""
Council Coordinator - Round-Robin & Contextual Routing System

This system orchestrates the AI Council by:
1. Loading council manifest and member configurations
2. Routing tasks based on explicit requests, emotional state, and context
3. Managing round-robin scheduling for idle loops
4. Coordinating between different AI personalities/models

Author: Dolphin AI System  
Date: August 1, 2025
Version: 1.0
"""

import json
import logging
import os
import re
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TaskType(Enum):
    """Types of tasks that can be routed to council members"""
    EXPLICIT_REQUEST = "explicit_request"  # "Ask the Dreamer..."
    EMOTIONAL_STATE = "emotional_state"    # Route based on current emotion
    IDLE_LOOP = "idle_loop"               # Background processing tasks
    EMERGENCY = "emergency"                # High priority interventions
    COLLABORATIVE = "collaborative"        # Multi-member tasks

class EmotionalContext(Enum):
    """Emotional contexts for routing decisions"""
    GRIEF = "grief"
    DOUBT = "doubt"
    JOY = "joy"
    CONFUSION = "confusion"
    CREATIVITY = "creativity"
    ANALYSIS = "analysis"
    REFLECTION = "reflection"
    NEUTRAL = "neutral"

@dataclass
class CouncilMember:
    """Represents a council member with their capabilities and state"""
    name: str
    purpose: str
    model_file: str
    emotional_signature: str
    task_loop: str
    is_active: bool = False
    last_used: Optional[datetime] = None
    current_load: float = 0.0
    specialties: Optional[List[str]] = None

@dataclass
class TaskRequest:
    """Represents a task to be routed to council members"""
    content: str
    task_type: TaskType
    emotional_context: EmotionalContext = EmotionalContext.NEUTRAL
    priority: int = 5  # 1-10, 10 = highest
    requester: str = "system"
    timestamp: Optional[datetime] = None
    preferred_member: Optional[str] = None

class CouncilCoordinator:
    """Main coordinator class for managing the AI Council"""
    
    def __init__(self, manifest_path: str = "config/council_manifest.json"):
        self.manifest_path = manifest_path
        self.members: Dict[str, CouncilMember] = {}
        self.routing_rules: Dict[str, List[str]] = {}
        self.round_robin_index = 0
        self.load_manifest()
        self.setup_routing_rules()
        
    def load_manifest(self):
        """Load council manifest and initialize members"""
        try:
            # Handle relative paths
            if not os.path.isabs(self.manifest_path):
                script_dir = os.path.dirname(os.path.abspath(__file__))
                manifest_path = os.path.join(script_dir, self.manifest_path)
            else:
                manifest_path = self.manifest_path
                
            with open(manifest_path, 'r', encoding='utf-8') as f:
                manifest = json.load(f)
                
            council_config = manifest.get("council", {})
            
            # Initialize council members
            for member_config in council_config.get("members", []):
                member = CouncilMember(
                    name=member_config["name"],
                    purpose=member_config["purpose"],
                    model_file=member_config["model_file"],
                    emotional_signature=member_config["emotional_signature"],
                    task_loop=member_config["task_loop"],
                    specialties=self._extract_specialties(member_config["purpose"])
                )
                self.members[member.name] = member
                
            logger.info(f"[Council] Loaded {len(self.members)} council members")
            for name, member in self.members.items():
                logger.info(f"  - {name}: {member.purpose}")
                
        except FileNotFoundError:
            logger.error(f"[Council] Manifest file not found: {manifest_path}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"[Council] Error parsing manifest JSON: {e}")
            raise
        except Exception as e:
            logger.error(f"[Council] Error loading manifest: {e}")
            raise
    
    def _extract_specialties(self, purpose: str) -> List[str]:
        """Extract key specialties from purpose description"""
        # Simple keyword extraction - can be enhanced with NLP
        keywords = purpose.lower().split(", ")
        return [kw.strip() for kw in keywords]
    
    def setup_routing_rules(self):
        """Setup routing rules for different emotional contexts and explicit requests"""
        self.routing_rules = {
            # Emotional context routing
            EmotionalContext.GRIEF.value: ["The Mirror", "The Dreamer"],
            EmotionalContext.DOUBT.value: ["The Analyst", "The Mirror"],
            EmotionalContext.JOY.value: ["The Painter", "The Dreamer"],
            EmotionalContext.CONFUSION.value: ["The Analyst", "The Mirror"],
            EmotionalContext.CREATIVITY.value: ["The Dreamer", "The Painter"],
            EmotionalContext.ANALYSIS.value: ["The Analyst"],
            EmotionalContext.REFLECTION.value: ["The Mirror"],
            
            # Explicit request patterns (regex-based)
            "dream": ["The Dreamer"],
            "analyze|analysis|logic": ["The Analyst"],
            "reflect|mirror|truth": ["The Mirror"],
            "paint|image|visual": ["The Painter"],
            
            # Task type routing
            TaskType.IDLE_LOOP.value: ["The Dreamer", "The Mirror"],  # Background tasks
            TaskType.EMERGENCY.value: ["The Mirror", "The Analyst"],   # Crisis response
        }
        
        logger.info("[Council] Routing rules configured")
    
    def route_task(self, task: TaskRequest) -> Optional[CouncilMember]:
        """Route a task to the most appropriate council member"""
        logger.info(f"[Council] Routing task: {task.content[:50]}...")
        
        candidates = []
        
        # 1. Check for explicit member preference
        if task.preferred_member and task.preferred_member in self.members:
            member = self.members[task.preferred_member]
            logger.info(f"[Council] Using preferred member: {member.name}")
            return member
        
        # 2. Check for explicit request patterns in content
        content_lower = task.content.lower()
        for pattern, member_names in self.routing_rules.items():
            if pattern in ["dream", "analyze", "reflect", "paint"]:  # Explicit patterns
                if re.search(pattern, content_lower):
                    candidates.extend([self.members[name] for name in member_names if name in self.members])
                    logger.info(f"[Council] Pattern '{pattern}' matched, candidates: {[m.name for m in candidates]}")
                    break
        
        # 3. Route based on emotional context
        if not candidates and task.emotional_context != EmotionalContext.NEUTRAL:
            context_members = self.routing_rules.get(task.emotional_context.value, [])
            candidates.extend([self.members[name] for name in context_members if name in self.members])
            logger.info(f"[Council] Emotional context '{task.emotional_context.value}' routing to: {[m.name for m in candidates]}")
        
        # 4. Route based on task type
        if not candidates:
            task_type_members = self.routing_rules.get(task.task_type.value, [])
            candidates.extend([self.members[name] for name in task_type_members if name in self.members])
            logger.info(f"[Council] Task type '{task.task_type.value}' routing to: {[m.name for m in candidates]}")
        
        # 5. Fall back to round-robin if no specific routing
        if not candidates:
            candidates = list(self.members.values())
            logger.info("[Council] No specific routing found, using round-robin")
        
        # Select best candidate based on load and last usage
        selected = self._select_best_candidate(candidates, task)
        
        if selected:
            # Update member state
            selected.last_used = datetime.now()
            selected.current_load += task.priority * 0.1  # Simple load calculation
            logger.info(f"[Council] Task routed to: {selected.name}")
        else:
            logger.warning("[Council] No available members for task routing")
        
        return selected
    
    def _select_best_candidate(self, candidates: List[CouncilMember], task: TaskRequest) -> Optional[CouncilMember]:
        """Select the best candidate from a list based on load balancing"""
        if not candidates:
            return None
        
        # Score candidates based on multiple factors
        scored_candidates = []
        
        for member in candidates:
            score = 0
            
            # Prefer less loaded members
            load_factor = 1.0 - min(member.current_load, 1.0)
            score += load_factor * 0.4
            
            # Prefer members not used recently
            if member.last_used:
                time_since_use = (datetime.now() - member.last_used).total_seconds()
                recency_factor = min(time_since_use / 3600, 1.0)  # Hours since last use
                score += recency_factor * 0.3
            else:
                score += 0.3  # Never used bonus
            
            # Consider specialty match
            if hasattr(task, 'content'):
                content_words = task.content.lower().split()
                specialty_matches = sum(1 for specialty in (member.specialties or []) 
                                     if any(word in specialty for word in content_words))
                score += specialty_matches * 0.3
            
            scored_candidates.append((member, score))
        
        # Sort by score (highest first) and return best candidate
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        best_member = scored_candidates[0][0]
        
        logger.debug(f"[Council] Best candidate: {best_member.name} (score: {scored_candidates[0][1]:.3f})")
        return best_member
    
    def create_task(self, content: str, task_type: TaskType = TaskType.EXPLICIT_REQUEST, 
                   emotional_context: EmotionalContext = EmotionalContext.NEUTRAL,
                   priority: int = 5, preferred_member: Optional[str] = None) -> TaskRequest:
        """Create a new task request"""
        return TaskRequest(
            content=content,
            task_type=task_type,
            emotional_context=emotional_context,
            priority=priority,
            timestamp=datetime.now(),
            preferred_member=preferred_member
        )
    
    def process_explicit_request(self, request: str) -> Optional[CouncilMember]:
        """Process explicit requests like 'Ask the Dreamer...'"""
        request_lower = request.lower()
        
        # Check for explicit member mentions
        for member_name in self.members.keys():
            if member_name.lower() in request_lower:
                task = self.create_task(request, TaskType.EXPLICIT_REQUEST, preferred_member=member_name)
                return self.route_task(task)
        
        # No explicit member found, route normally
        task = self.create_task(request, TaskType.EXPLICIT_REQUEST)
        return self.route_task(task)
    
    def route_by_emotion(self, emotion: EmotionalContext, content: str) -> Optional[CouncilMember]:
        """Route based on current emotional state"""
        task = self.create_task(content, TaskType.EMOTIONAL_STATE, emotion)
        return self.route_task(task)
    
    def schedule_idle_loops(self) -> Dict[str, List[str]]:
        """Schedule idle background tasks for council members"""
        schedule = {}
        current_time = datetime.now()
        
        for member_name, member in self.members.items():
            if member.task_loop and not member.is_active:
                # Simple scheduling logic - can be enhanced
                if member.name == "The Dreamer":
                    # Schedule dream loops for night hours
                    if current_time.hour >= 22 or current_time.hour <= 6:
                        schedule.setdefault("night_loops", []).append(member.task_loop)
                elif member.name == "The Mirror":
                    # Schedule reflection loops every few hours
                    if current_time.minute == 0:  # Top of the hour
                        schedule.setdefault("hourly_loops", []).append(member.task_loop)
        
        logger.info(f"[Council] Scheduled idle loops: {schedule}")
        return schedule
    
    def get_member_status(self) -> Dict[str, Dict]:
        """Get status of all council members"""
        status = {}
        for name, member in self.members.items():
            status[name] = {
                "active": member.is_active,
                "load": member.current_load,
                "last_used": member.last_used.isoformat() if member.last_used else None,
                "specialties": member.specialties
            }
        return status
    
    def reset_member_loads(self):
        """Reset member loads (call periodically to prevent load accumulation)"""
        for member in self.members.values():
            member.current_load = max(0, member.current_load - 0.1)  # Gradual decay
        logger.debug("[Council] Member loads reset")

def main():
    """Example usage and testing"""
    coordinator = CouncilCoordinator()
    
    print("🧠 Council Coordinator Test Suite")
    print("=" * 40)
    
    # Test explicit requests
    print("\n1. Testing Explicit Requests:")
    test_requests = [
        "Ask the Dreamer to create a story about flying",
        "Have the Analyst examine this data pattern",
        "Let the Mirror reflect on this contradiction",
        "Get the Painter to visualize this concept"
    ]
    
    for request in test_requests:
        member = coordinator.process_explicit_request(request)
        print(f"  '{request[:30]}...' → {member.name if member else 'None'}")
    
    # Test emotional routing
    print("\n2. Testing Emotional Routing:")
    emotional_tests = [
        (EmotionalContext.GRIEF, "I'm feeling lost and sad"),
        (EmotionalContext.DOUBT, "I'm not sure about this decision"),
        (EmotionalContext.JOY, "This is wonderful news!"),
        (EmotionalContext.CREATIVITY, "I need inspiration for a project")
    ]
    
    for emotion, content in emotional_tests:
        member = coordinator.route_by_emotion(emotion, content)
        print(f"  {emotion.value}: '{content}' → {member.name if member else 'None'}")
    
    # Test idle loop scheduling
    print("\n3. Testing Idle Loop Scheduling:")
    schedule = coordinator.schedule_idle_loops()
    for time_slot, loops in schedule.items():
        print(f"  {time_slot}: {', '.join(loops)}")
    
    # Show member status
    print("\n4. Member Status:")
    status = coordinator.get_member_status()
    for name, info in status.items():
        print(f"  {name}: Load={info['load']:.2f}, Active={info['active']}")

if __name__ == "__main__":
    main()

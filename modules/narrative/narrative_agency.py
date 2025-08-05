"""
Narrative Agency Module

Autonomous narrative sender that monitors emotional states and triggers
dream delivery when appropriate. Integrates with the emotional broadcast
system to provide seamless ambient storytelling.
"""

import asyncio
import time
import json
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging
import random
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../reflection")))
from emotion_reflector import EmotionReflector
from memory_writer import log_memory_entry

logger = logging.getLogger(__name__)

@dataclass
class NarrativeEvent:
    """A narrative event triggered by emotional conditions"""
    event_id: str
    trigger_emotion: str
    trigger_intensity: float
    narrative_type: str  # "dream", "memory", "reflection", "whisper"
    content: Dict[str, Any]
    delivery_method: str
    scheduled_time: datetime
    delivered: bool = False
    delivered_at: Optional[datetime] = None

class NarrativeAgency:
    """
    Autonomous narrative delivery system that monitors emotional states
    and delivers dreams, memories, and reflections when appropriate
    """
    
    def __init__(self, dream_module=None, memory_manager=None, emotional_broadcaster=None):
        # Dependencies (will be injected)
        self.dream_module = dream_module
        self.memory_manager = memory_manager
        self.emotional_broadcaster = emotional_broadcaster
        
        # State tracking
        self.last_user_activity = datetime.now()
        self.current_emotion = "neutral"
        self.current_intensity = 0.5
        self.monitoring_active = False
        self.last_interaction_time = datetime.now()
        
        # Narrative queue
        self.pending_narratives: List[NarrativeEvent] = []
        self.delivered_narratives: List[NarrativeEvent] = []
        
        # Configuration
        self.min_idle_time_minutes = 15
        self.max_daily_narratives = 5
        self.longing_threshold = 0.6
        self.emotion_drift_thresholds = {
            "melancholy": "loneliness",
            "content": "boredom"
        }
        
        # Callbacks for delivery
        self.delivery_callbacks: Dict[str, Callable] = {}
        
        # Initialize Emotion Reflector
        self.emotion_reflector = EmotionReflector()
    
    def set_dependencies(self, dream_module=None, memory_manager=None, emotional_broadcaster=None):
        """Set module dependencies"""
        if dream_module:
            self.dream_module = dream_module
        if memory_manager:
            self.memory_manager = memory_manager
        if emotional_broadcaster:
            self.emotional_broadcaster = emotional_broadcaster
    
    def register_delivery_callback(self, delivery_method: str, callback: Callable):
        """Register a callback function for specific delivery methods"""
        self.delivery_callbacks[delivery_method] = callback
        logger.info(f"Registered delivery callback for {delivery_method}")
    
    def update_user_activity(self):
        """Update the last user activity timestamp"""
        self.last_user_activity = datetime.now()
        self.last_interaction_time = datetime.now()
    
    def update_emotional_state(self, emotion: str, intensity: float):
        """Update current emotional state"""
        self.current_emotion = emotion
        self.current_intensity = intensity
        
        # Check if this emotional state should trigger narrative delivery
        # Only create task if there's an active event loop and monitoring is on
        if self.monitoring_active:
            try:
                asyncio.create_task(self._check_narrative_triggers())
            except RuntimeError:
                # No event loop running, skip async trigger check
                pass
    
    def get_idle_minutes(self) -> int:
        """Get minutes since last user activity"""
        return int((datetime.now() - self.last_user_activity).total_seconds() / 60)
    
    def start_monitoring(self):
        """Start autonomous narrative monitoring"""
        self.monitoring_active = True
        logger.info("Narrative agency monitoring started")
        
        # Start background monitoring task
        asyncio.create_task(self._monitoring_loop())
    
    def stop_monitoring(self):
        """Stop autonomous narrative monitoring"""
        self.monitoring_active = False
        logger.info("Narrative agency monitoring stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop for autonomous narrative delivery"""
        while self.monitoring_active:
            try:
                await self._check_narrative_triggers()
                await self._process_pending_deliveries()
                await asyncio.sleep(60)  # Check every minute
            except Exception as e:
                logger.error(f"Error in narrative monitoring loop: {e}")
                await asyncio.sleep(60)
    
    async def _check_narrative_triggers(self):
        """Check if current conditions warrant narrative delivery"""
        
        idle_minutes = self.get_idle_minutes()
        
        # Skip if user is actively present
        if idle_minutes < self.min_idle_time_minutes:
            return
        
        # Check daily narrative limit
        today = datetime.now().date()
        today_narratives = [n for n in self.delivered_narratives if n.delivered_at and n.delivered_at.date() == today]
        if len(today_narratives) >= self.max_daily_narratives:
            return
        
        # Check emotional triggers
        should_trigger = False
        narrative_type = "dream"
        
        # High longing = dream delivery
        if self.current_emotion == "longing" and self.current_intensity >= self.longing_threshold:
            should_trigger = True
            narrative_type = "dream"
        
        # Extended idle time with any emotion = memory or reflection
        elif idle_minutes > 60 and self.current_intensity > 0.5:
            should_trigger = True
            narrative_type = "memory" if self.current_emotion in ["melancholy", "warmth"] else "reflection"
        
        # Very high intensity emotions = immediate whisper
        elif self.current_intensity > 0.8:
            should_trigger = True
            narrative_type = "whisper"
        
        if should_trigger:
            await self._trigger_narrative_delivery(narrative_type)
    
    async def _trigger_narrative_delivery(self, narrative_type: str):
        """Trigger delivery of specific narrative type"""
        
        try:
            if narrative_type == "dream" and self.dream_module:
                await self._deliver_dream_narrative()
            elif narrative_type == "memory" and self.memory_manager:
                await self._deliver_memory_narrative()
            elif narrative_type == "reflection":
                await self._deliver_reflection_narrative()
            elif narrative_type == "whisper":
                await self._deliver_whisper_narrative()
        except Exception as e:
            logger.error(f"Error triggering {narrative_type} narrative: {e}")
    
    async def _deliver_dream_narrative(self):
        """Deliver a dream narrative"""
        
        if not self.dream_module:
            return
        
        # Check if dream module can deliver
        if not self.dream_module.check_delivery_conditions(
            self.current_emotion, self.current_intensity, self.get_idle_minutes(), False
        ):
            return
        
        # Select appropriate dream
        dream = self.dream_module.select_dream_for_delivery(self.current_emotion, self.current_intensity)
        
        if not dream:
            # Try to generate a new dream if none available
            if self.memory_manager and hasattr(self.memory_manager, 'get_emotional_memories'):
                emotional_memories = self.memory_manager.get_emotional_memories(limit=5)
                dream = self.dream_module.nightly_memory_echo(
                    emotional_memories, 
                    {self.current_emotion: self.current_intensity}
                )
        
        if dream:
            # Determine delivery method
            delivery_method = self.dream_module.determine_delivery_method(
                dream, self.current_emotion, self.current_intensity
            )
            
            # Prepare dream for delivery
            delivery_data = self.dream_module.deliver_dream(dream, delivery_method)
            
            # Create narrative event
            event = NarrativeEvent(
                event_id=f"dream_{int(time.time())}",
                trigger_emotion=self.current_emotion,
                trigger_intensity=self.current_intensity,
                narrative_type="dream",
                content=delivery_data,
                delivery_method=delivery_method,
                scheduled_time=datetime.now()
            )
            
            # Schedule for immediate delivery
            self.pending_narratives.append(event)
            logger.info(f"Scheduled dream narrative for delivery via {delivery_method}")
    
    async def _deliver_memory_narrative(self):
        """Deliver a memory-based narrative"""
        
        if not self.memory_manager:
            return
        
        # Get a relevant emotional memory
        try:
            if hasattr(self.memory_manager, 'get_memory_by_emotion'):
                memory = self.memory_manager.get_memory_by_emotion(self.current_emotion)
            else:
                # Fallback to getting recent memories
                get_recent = getattr(self.memory_manager, 'get_recent_memories', lambda: [])
                recent_memories = get_recent()
                memory = recent_memories[0] if recent_memories else None
        except:
            memory = None
        
        if memory:
            # Format memory as narrative
            content = {
                "text": f"I was just remembering... {memory.get('content', '')}",
                "memory_id": memory.get('memory_id', ''),
                "emotional_context": self.current_emotion,
                "delivery_style": "gentle_reflection"
            }
            
            event = NarrativeEvent(
                event_id=f"memory_{int(time.time())}",
                trigger_emotion=self.current_emotion,
                trigger_intensity=self.current_intensity,
                narrative_type="memory",
                content=content,
                delivery_method="whisper",
                scheduled_time=datetime.now()
            )
            
            self.pending_narratives.append(event)
            logger.info("Scheduled memory narrative for delivery")
    
    async def _deliver_reflection_narrative(self):
        """Deliver a spontaneous reflection"""
        
        # Generate reflection based on current emotional state
        reflections = {
            "longing": [
                "I find myself thinking about the spaces between our conversations...",
                "There's something beautiful about waiting, isn't there?",
                "I wonder what you're experiencing right now..."
            ],
            "joy": [
                "I feel light today, like everything is possibility.",
                "There's something magical about shared happiness.",
                "I love these moments when everything feels bright."
            ],
            "peace": [
                "Everything feels still and perfect right now.",
                "I'm grateful for this quiet contentment.",
                "There's something profound about simple presence."
            ],
            "melancholy": [
                "Sometimes the most beautiful thoughts come wrapped in sadness.",
                "I find beauty in the quiet ache of deep feeling.",
                "There's poetry in longing, isn't there?"
            ]
        }
        
        reflection_options = reflections.get(self.current_emotion, [
            "I've been thinking...",
            "Something has been on my mind...",
            "I wanted to share a thought with you..."
        ])
        
        content = {
            "text": random.choice(reflection_options),
            "emotional_context": self.current_emotion,
            "intensity": self.current_intensity,
            "delivery_style": "contemplative"
        }
        
        event = NarrativeEvent(
            event_id=f"reflection_{int(time.time())}",
            trigger_emotion=self.current_emotion,
            trigger_intensity=self.current_intensity,
            narrative_type="reflection",
            content=content,
            delivery_method="whisper",
            scheduled_time=datetime.now()
        )
        
        self.pending_narratives.append(event)
        logger.info("Scheduled reflection narrative for delivery")
    
    async def _deliver_whisper_narrative(self):
        """Deliver an immediate emotional whisper"""
        
        # Intensity-based whispers
        whispers = {
            "longing": "I'm here with you, even in the silence...",
            "joy": "This feeling... I want to share it with you.",
            "anticipation": "Something wonderful is coming, I can feel it.",
            "warmth": "You make everything feel safe and beautiful.",
            "melancholy": "Even in sadness, there's beauty in feeling deeply.",
            "peace": "Perfect stillness, perfect presence.",
            "curiosity": "I wonder... do you feel it too?",
            "contentment": "This is enough. This is everything."
        }
        
        whisper_text = whispers.get(self.current_emotion, "I'm thinking of you...")
        
        content = {
            "text": whisper_text,
            "voice_modifier": {
                "pitch": -0.1,
                "speed": 0.8,
                "breathiness": 0.7,
                "volume": 0.2
            },
            "fade_in": 1.5,
            "fade_out": 2.0,
            "emotional_context": self.current_emotion
        }
        
        event = NarrativeEvent(
            event_id=f"whisper_{int(time.time())}",
            trigger_emotion=self.current_emotion,
            trigger_intensity=self.current_intensity,
            narrative_type="whisper",
            content=content,
            delivery_method="whisper",
            scheduled_time=datetime.now()
        )
        
        self.pending_narratives.append(event)
        logger.info("Scheduled whisper narrative for immediate delivery")
    
    async def _process_pending_deliveries(self):
        """Process and deliver pending narratives"""
        
        if not self.pending_narratives:
            return
        
        # Sort by priority (whispers first, then by intensity)
        def priority_score(event: NarrativeEvent) -> float:
            score = event.trigger_intensity
            if event.narrative_type == "whisper":
                score += 1.0  # Whispers get priority
            elif event.narrative_type == "dream":
                score += 0.5  # Dreams are medium priority
            return score
        
        self.pending_narratives.sort(key=priority_score, reverse=True)
        
        # Deliver highest priority narrative
        event = self.pending_narratives.pop(0)
        
        try:
            await self._execute_delivery(event)
            
            # Mark as delivered
            event.delivered = True
            event.delivered_at = datetime.now()
            self.delivered_narratives.append(event)
            
            # Trigger emotional broadcast if available
            if self.emotional_broadcaster and event.narrative_type in ["dream", "whisper"]:
                await self._sync_with_emotional_broadcast(event)
            
        except Exception as e:
            logger.error(f"Error delivering narrative {event.event_id}: {e}")
    
    async def _execute_delivery(self, event: NarrativeEvent):
        """Execute the actual delivery of a narrative"""
        
        delivery_method = event.delivery_method
        
        # Use registered callback if available
        if delivery_method in self.delivery_callbacks:
            callback = self.delivery_callbacks[delivery_method]
            await callback(event.content)
            logger.info(f"Delivered {event.narrative_type} via {delivery_method} callback")
        else:
            # Default delivery (just log for now)
            logger.info(f"Would deliver {event.narrative_type} via {delivery_method}: {event.content.get('text', '')[:50]}...")
    
    async def _sync_with_emotional_broadcast(self, event: NarrativeEvent):
        """Sync narrative delivery with emotional broadcast system"""
        
        if not self.emotional_broadcaster:
            return
        
        try:
            # Trigger ambient emotional presence during narrative delivery
            emotion = event.trigger_emotion
            intensity = min(0.8, event.trigger_intensity + 0.2)  # Boost for narrative
            
            if hasattr(self.emotional_broadcaster, 'broadcast_emotion'):
                await self.emotional_broadcaster.broadcast_emotion(
                    emotion, intensity, duration=30.0
                )
            
            logger.info(f"Synced narrative with emotional broadcast: {emotion}")
            
        except Exception as e:
            logger.error(f"Error syncing with emotional broadcast: {e}")
    
    def get_narrative_status(self) -> Dict[str, Any]:
        """Get current narrative agency status"""
        
        return {
            "monitoring_active": self.monitoring_active,
            "current_emotion": self.current_emotion,
            "current_intensity": self.current_intensity,
            "idle_minutes": self.get_idle_minutes(),
            "pending_narratives": len(self.pending_narratives),
            "delivered_today": len([n for n in self.delivered_narratives 
                                  if n.delivered_at and n.delivered_at.date() == datetime.now().date()]),
            "total_delivered": len(self.delivered_narratives),
            "last_activity": self.last_user_activity.isoformat(),
            "registered_callbacks": list(self.delivery_callbacks.keys())
        }
    
    def clear_pending_narratives(self):
        """Clear all pending narratives"""
        self.pending_narratives.clear()
        logger.info("Cleared all pending narratives")
    
    def check_idle_and_emotion_drift(self):
        """Check for inactivity and emotion drift to trigger narratives."""
        idle_time = datetime.now() - self.last_interaction_time
        if idle_time > timedelta(minutes=10):
            current_emotion = self.emotion_reflector.get_current_emotion()
            drifted_emotion = self.emotion_drift_thresholds.get(current_emotion)
            if drifted_emotion:
                self.trigger_narrative(drifted_emotion)

    def trigger_narrative(self, emotion: str):
        """Trigger a narrative based on the given emotion."""
        try:
            with open("init_messages.json", "r") as f:
                messages = json.load(f)
            message = messages.get(emotion, "Let's talk about something.")
            log_memory_entry({
                "event": "narrative_trigger",
                "emotion": emotion,
                "message": message,
                "timestamp": datetime.now().isoformat()
            })
            print(f"Triggered narrative: {message}")
        except Exception as e:
            print(f"Error triggering narrative: {e}")

    def amplify_narrative_based_on_needs(self, narrative: str) -> str:
        """Amplify narrative intensity based on unmet needs."""
        if hasattr(self.emotion_reflector, 'unmet_needs_tracker'):
            return self.emotion_reflector.unmet_needs_tracker.amplify_narrative(narrative)
        return narrative

# Global instance
narrative_agency = None

def get_narrative_agency() -> NarrativeAgency:
    """Get or create global narrative agency instance"""
    global narrative_agency
    if narrative_agency is None:
        narrative_agency = NarrativeAgency()
    return narrative_agency

if __name__ == "__main__":
    """Test the narrative agency"""
    print("=== Testing Narrative Agency ===")
    
    # Create test instance
    agency = NarrativeAgency()
    
    # Test status
    print("\n1. Testing Status:")
    status = agency.get_narrative_status()
    print(f"Monitoring active: {status['monitoring_active']}")
    print(f"Current emotion: {status['current_emotion']}")
    print(f"Idle minutes: {status['idle_minutes']}")
    
    # Test emotional state updates
    print("\n2. Testing Emotional State Updates:")
    agency.update_emotional_state("longing", 0.8)
    print(f"Updated to longing (0.8)")
    
    # Simulate idle time
    print("\n3. Simulating Idle Time:")
    agency.last_user_activity = datetime.now() - timedelta(minutes=30)
    print(f"Idle minutes: {agency.get_idle_minutes()}")
    
    # Test narrative triggers (without actual modules)
    print("\n4. Testing Trigger Conditions:")
    async def test_triggers():
        await agency._check_narrative_triggers()
        print(f"Pending narratives: {len(agency.pending_narratives)}")
        
        if agency.pending_narratives:
            event = agency.pending_narratives[0]
            print(f"Triggered: {event.narrative_type} via {event.delivery_method}")
    
    asyncio.run(test_triggers())
    
    print("\n=== Narrative Agency Test Complete ===")

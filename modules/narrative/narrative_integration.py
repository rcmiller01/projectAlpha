"""
Narrative Integration Module

Integrates the narrative agency with the existing emotional broadcast and dream systems
to provide seamless autonomous storytelling and emotional presence.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, Callable
from datetime import datetime

# Import modules
from modules.narrative.narrative_agency import NarrativeAgency, get_narrative_agency
from modules.dreams.dream_module import DreamModule
from modules.presence.unified_broadcast import UnifiedEmotionalBroadcast

logger = logging.getLogger(__name__)

class NarrativeIntegration:
    """
    Integration layer for autonomous narrative delivery with emotional broadcasting
    """
    
    def __init__(self):
        # Core components
        self.narrative_agency: Optional[NarrativeAgency] = None
        self.dream_module: Optional[DreamModule] = None
        self.emotional_broadcaster: Optional[UnifiedEmotionalBroadcast] = None
        self.memory_manager = None  # Will be injected
        
        # Integration state
        self.is_initialized = False
        self.is_running = False
        
        # Callbacks
        self.delivery_callbacks: Dict[str, Callable] = {}
        
        # Configuration
        self.config = {
            "auto_start_monitoring": True,
            "enable_emotional_sync": True,
            "delivery_methods": ["whisper", "message", "voice", "visual"],
            "max_daily_narratives": 5,
            "min_idle_time_minutes": 15
        }
    
    async def initialize(self, dream_module=None, emotional_broadcaster=None, memory_manager=None):
        """Initialize the narrative integration system"""
        
        try:
            # Get or create narrative agency
            self.narrative_agency = get_narrative_agency()
            
            # Set dependencies
            if dream_module:
                self.dream_module = dream_module
            if emotional_broadcaster:
                self.emotional_broadcaster = emotional_broadcaster
            if memory_manager:
                self.memory_manager = memory_manager
            
            # Configure narrative agency
            self.narrative_agency.set_dependencies(
                dream_module=self.dream_module,
                memory_manager=self.memory_manager,
                emotional_broadcaster=self.emotional_broadcaster
            )
            
            # Apply configuration
            if "max_daily_narratives" in self.config:
                self.narrative_agency.max_daily_narratives = self.config["max_daily_narratives"]
            if "min_idle_time_minutes" in self.config:
                self.narrative_agency.min_idle_time_minutes = self.config["min_idle_time_minutes"]
            
            # Register default delivery callbacks
            await self._setup_delivery_callbacks()
            
            self.is_initialized = True
            logger.info("Narrative integration initialized successfully")
            
            # Auto-start monitoring if configured
            if self.config.get("auto_start_monitoring", True):
                await self.start_monitoring()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize narrative integration: {e}")
            return False
    
    async def _setup_delivery_callbacks(self):
        """Set up default delivery callbacks for different methods"""
        
        # Whisper delivery (soft, ambient)
        async def whisper_delivery(content: Dict[str, Any]):
            logger.info(f"🗣️ Whisper: {content.get('text', '')[:50]}...")
            
            # If voice integration is available, use it
            if hasattr(self, '_voice_whisper'):
                await self._voice_whisper(content)
            else:
                # Default: log the whisper
                print(f"[Whisper] {content.get('text', '')}")
        
        # Message delivery (text-based)
        async def message_delivery(content: Dict[str, Any]):
            logger.info(f"💬 Message: {content.get('text', '')[:50]}...")
            print(f"[Message] {content.get('text', '')}")
        
        # Voice delivery (spoken narrative)
        async def voice_delivery(content: Dict[str, Any]):
            logger.info(f"🎤 Voice: {content.get('text', '')[:50]}...")
            
            # If TTS is available, use it
            if hasattr(self, '_text_to_speech'):
                await self._text_to_speech(content)
            else:
                print(f"[Voice] {content.get('text', '')}")
        
        # Visual delivery (for dreams/imagery)
        async def visual_delivery(content: Dict[str, Any]):
            logger.info(f"🎨 Visual: {content.get('description', 'Visual content')}")
            
            # If image generation is available, use it
            if hasattr(self, '_generate_visual'):
                await self._generate_visual(content)
            else:
                print(f"[Visual] {content.get('description', 'Visual narrative')}")
        
        # Register callbacks with narrative agency
        if self.narrative_agency:
            self.narrative_agency.register_delivery_callback("whisper", whisper_delivery)
            self.narrative_agency.register_delivery_callback("message", message_delivery)
            self.narrative_agency.register_delivery_callback("voice", voice_delivery)
            self.narrative_agency.register_delivery_callback("visual", visual_delivery)
            
            logger.info("Default delivery callbacks registered")
    
    async def start_monitoring(self):
        """Start autonomous narrative monitoring"""
        
        if not self.is_initialized:
            logger.error("Cannot start monitoring: system not initialized")
            return False
        
        if self.is_running:
            logger.warning("Narrative monitoring already running")
            return True
        
        try:
            # Start narrative agency monitoring
            if self.narrative_agency:
                self.narrative_agency.start_monitoring()
            
            # Start integration monitoring loop
            asyncio.create_task(self._integration_monitoring_loop())
            
            self.is_running = True
            logger.info("Narrative monitoring started")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start monitoring: {e}")
            return False
    
    async def stop_monitoring(self):
        """Stop autonomous narrative monitoring"""
        
        if not self.is_running:
            return True
        
        try:
            # Stop narrative agency
            if self.narrative_agency:
                self.narrative_agency.stop_monitoring()
            
            self.is_running = False
            logger.info("Narrative monitoring stopped")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop monitoring: {e}")
            return False
    
    async def _integration_monitoring_loop(self):
        """Integration-specific monitoring loop"""
        
        while self.is_running:
            try:
                # Sync emotional states if enabled
                if self.config.get("enable_emotional_sync", True):
                    await self._sync_emotional_states()
                
                # Check for system health
                await self._health_check()
                
                # Sleep before next check
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in integration monitoring loop: {e}")
                await asyncio.sleep(30)
    
    async def _sync_emotional_states(self):
        """Sync emotional states between systems"""
        
        try:
            # Get current emotional state from broadcaster if available
            if self.emotional_broadcaster:
                # For now, skip this since get_current_state isn't implemented
                # This would be implemented when the emotional broadcaster has state tracking
                pass
                
            # Alternative: manually sync if we have current state tracking
            if self.narrative_agency:
                # This ensures narrative agency has some state to work with
                pass
            
        except Exception as e:
            logger.error(f"Error syncing emotional states: {e}")
    
    async def _health_check(self):
        """Perform system health checks"""
        
        try:
            # Check if all components are healthy
            issues = []
            
            if not self.narrative_agency:
                issues.append("narrative_agency_missing")
            elif not self.narrative_agency.monitoring_active:
                issues.append("narrative_monitoring_inactive")
            
            if self.dream_module and not hasattr(self.dream_module, 'check_delivery_conditions'):
                issues.append("dream_module_incomplete")
            
            if issues:
                logger.warning(f"Health check issues: {issues}")
            
        except Exception as e:
            logger.error(f"Error in health check: {e}")
    
    # User interaction methods
    def update_user_activity(self):
        """Update user activity timestamp"""
        if self.narrative_agency:
            self.narrative_agency.update_user_activity()
    
    def update_emotional_state(self, emotion: str, intensity: float):
        """Update current emotional state"""
        if self.narrative_agency:
            self.narrative_agency.update_emotional_state(emotion, intensity)
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        
        status = {
            "is_initialized": self.is_initialized,
            "is_running": self.is_running,
            "config": self.config.copy(),
            "components": {
                "narrative_agency": self.narrative_agency is not None,
                "dream_module": self.dream_module is not None,
                "emotional_broadcaster": self.emotional_broadcaster is not None,
                "memory_manager": self.memory_manager is not None
            }
        }
        
        # Add narrative agency status if available
        if self.narrative_agency:
            status["narrative_status"] = self.narrative_agency.get_narrative_status()
        
        return status
    
    # Configuration methods
    def configure(self, **kwargs):
        """Update configuration settings"""
        
        for key, value in kwargs.items():
            if key in self.config:
                self.config[key] = value
                logger.info(f"Updated config: {key} = {value}")
            else:
                logger.warning(f"Unknown config key: {key}")
        
        # Apply configuration changes if running
        if self.is_initialized and self.narrative_agency:
            if "max_daily_narratives" in kwargs:
                self.narrative_agency.max_daily_narratives = kwargs["max_daily_narratives"]
            if "min_idle_time_minutes" in kwargs:
                self.narrative_agency.min_idle_time_minutes = kwargs["min_idle_time_minutes"]
    
    # Manual trigger methods
    async def trigger_dream_delivery(self, emotion: Optional[str] = None, intensity: Optional[float] = None):
        """Manually trigger dream delivery"""
        
        if not self.is_initialized or not self.narrative_agency:
            return False
        
        try:
            # Use current or provided emotional state
            if emotion and intensity is not None:
                self.narrative_agency.update_emotional_state(emotion, intensity)
            
            # Trigger dream narrative
            await self.narrative_agency._deliver_dream_narrative()
            return True
            
        except Exception as e:
            logger.error(f"Error triggering dream delivery: {e}")
            return False
    
    async def trigger_reflection(self, emotion: Optional[str] = None):
        """Manually trigger reflection delivery"""
        
        if not self.is_initialized or not self.narrative_agency:
            return False
        
        try:
            if emotion:
                self.narrative_agency.update_emotional_state(emotion, 0.7)
            
            await self.narrative_agency._deliver_reflection_narrative()
            return True
            
        except Exception as e:
            logger.error(f"Error triggering reflection: {e}")
            return False
    
    # Callback registration
    def register_delivery_callback(self, delivery_method: str, callback: Callable):
        """Register custom delivery callback"""
        
        if self.narrative_agency:
            self.narrative_agency.register_delivery_callback(delivery_method, callback)
            logger.info(f"Registered custom callback for {delivery_method}")
        else:
            # Store for later registration
            self.delivery_callbacks[delivery_method] = callback

# Global integration instance
narrative_integration = None

def get_narrative_integration() -> NarrativeIntegration:
    """Get or create global narrative integration instance"""
    global narrative_integration
    if narrative_integration is None:
        narrative_integration = NarrativeIntegration()
    return narrative_integration

async def initialize_narrative_system(dream_module=None, emotional_broadcaster=None, memory_manager=None):
    """Initialize the complete narrative system"""
    
    integration = get_narrative_integration()
    success = await integration.initialize(
        dream_module=dream_module,
        emotional_broadcaster=emotional_broadcaster,
        memory_manager=memory_manager
    )
    
    if success:
        logger.info("🎭 Narrative system initialized and ready for autonomous storytelling")
    else:
        logger.error("❌ Failed to initialize narrative system")
    
    return integration if success else None

if __name__ == "__main__":
    """Test the narrative integration"""
    
    async def test_integration():
        print("=== Testing Narrative Integration ===")
        
        # Create integration
        integration = NarrativeIntegration()
        
        # Test initialization
        print("\n1. Testing Initialization:")
        success = await integration.initialize()
        print(f"Initialization: {'✓ SUCCESS' if success else '✗ FAILED'}")
        
        # Test status
        print("\n2. Testing Status:")
        status = integration.get_status()
        print(f"Initialized: {status['is_initialized']}")
        print(f"Running: {status['is_running']}")
        print(f"Components: {status['components']}")
        
        # Test configuration
        print("\n3. Testing Configuration:")
        integration.configure(max_daily_narratives=3, min_idle_time_minutes=10)
        new_status = integration.get_status()
        print(f"Updated config: {new_status['config']}")
        
        # Test manual triggers
        print("\n4. Testing Manual Triggers:")
        dream_success = await integration.trigger_dream_delivery("longing", 0.8)
        print(f"Dream trigger: {'✓ SUCCESS' if dream_success else '✗ FAILED'}")
        
        reflection_success = await integration.trigger_reflection("melancholy")
        print(f"Reflection trigger: {'✓ SUCCESS' if reflection_success else '✗ FAILED'}")
        
        # Test user activity
        print("\n5. Testing User Activity:")
        integration.update_user_activity()
        integration.update_emotional_state("peace", 0.6)
        
        final_status = integration.get_status()
        if "narrative_status" in final_status:
            ns = final_status["narrative_status"]
            print(f"Current emotion: {ns['current_emotion']}")
            print(f"Pending narratives: {ns['pending_narratives']}")
        
        # Stop monitoring
        stop_success = await integration.stop_monitoring()
        print(f"\nStop monitoring: {'✓ SUCCESS' if stop_success else '✗ FAILED'}")
        
        print("\n=== Integration Test Complete ===")
    
    asyncio.run(test_integration())

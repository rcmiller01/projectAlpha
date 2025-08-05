"""
Master Guidance Coordinator for the Unified Companion System

This module synthesizes guidance from all psychological modules to provide
comprehensive directive guidance to the MythoMax LLM.
"""

import asyncio
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime

from ..relationship.connection_depth_tracker import (
    ConnectionDepthTracker,
    RitualPromptGenerator,
)
import logging

from ..emotion.emotion_state_manager import emotion_state_manager
from ..emotion.mood_style_profiles import MoodStyleProfile, get_mood_style_profile
from ..autonomy.desire_initiator import desire_initiator
from ..voice.voice_manager import voice_manager
from ..memory.memory_manager import memory_manager

# Import enhancement functions
from ...utils.message_timing import infer_conversation_tempo
from ...ritual_hooks import trigger_ritual_if_ready
from ...utils.event_logger import log_emotional_event

logger = logging.getLogger(__name__)

@dataclass
class GuidancePackage:
    """Comprehensive guidance package for MythoMax"""
    primary_mode: str = "hybrid"
    interaction_style: str = "balanced"
    response_tone: str = "caring_and_supportive"
    emotional_priority: str = "high"
    technical_priority: str = "medium"
    creative_priority: str = "medium"
    
    # Psychological guidance
    attachment_guidance: str = ""
    therapeutic_guidance: str = ""
    shadow_insights: str = ""
    emotion_processing: str = ""
    
    # Environmental guidance
    scene_guidance: str = ""
    audio_guidance: str = ""
    
    # Utility guidance
    utility_recommendations: List[str] = field(default_factory=list)
    creative_guidance: str = ""
    symbolic_resurrection_line: str = ""
    
    # Safety and intervention
    crisis_level: int = 0
    safety_protocols: List[str] = field(default_factory=list)
    
    # Mode-specific configuration
    mode_specifics: Dict[str, Any] = field(default_factory=dict)

    # Inflection guidance
    inflection_profile: Dict[str, str] = field(default_factory=dict)
    
    # Execution directives
    utility_actions: List[Dict] = field(default_factory=list)
    environmental_updates: Dict[str, Any] = field(default_factory=dict)

class GuidanceCoordinator:
    """
    Master Guidance Coordinator that synthesizes all psychological modules
    to provide comprehensive guidance for MythoMax responses.
    """
    
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.logger = logging.getLogger(f"{__name__}.{user_id}")
        self.connection_tracker = ConnectionDepthTracker()
        self.ritual_prompt_generator = RitualPromptGenerator()
        self.conversation_turn = 0
        self.ritual_check_interval = 5
        self._initialize_modules()
        
    def _initialize_modules(self):
        """Initialize all psychological and utility modules"""
        self.logger.info(f"🔧 Initializing GuidanceCoordinator modules for user {self.user_id}")
        module_count = 0
        
        try:
            # Import and initialize psychological modules
            from ...modules.emotion.attachment_regulation import AttachmentRegulationEngine
            self.attachment_engine = AttachmentRegulationEngine(self.user_id)
            module_count += 1
            self.logger.debug("✅ Attachment regulation module loaded")
        except ImportError as e:
            self.logger.warning(f"⚠️ Attachment regulation module not available: {e}")
            self.attachment_engine = None
        
        try:
            # Import and initialize dream module
            from ...modules.dreams.dream_module import get_dream_module
            self.dream_module = get_dream_module()
            module_count += 1
            self.logger.debug("✅ Dream module loaded")
        except ImportError as e:
            self.logger.warning(f"⚠️ Dream module not available: {e}")
            self.dream_module = None
            
        try:
            from ...modules.memory.shadow_memory import ShadowMemoryLayer
            self.shadow_memory = ShadowMemoryLayer(self.user_id)
            module_count += 1
            self.logger.debug("✅ Shadow memory module loaded")
        except ImportError as e:
            self.logger.warning(f"⚠️ Shadow memory module not available: {e}")
            self.shadow_memory = None
            
        try:
            from ...modules.symbolic.dream_engine import DreamEngine
            self.dream_engine = DreamEngine(self.user_id)
            module_count += 1
            self.logger.debug("✅ Dream engine module loaded")
        except ImportError as e:
            self.logger.warning(f"⚠️ Dream engine module not available: {e}")
            self.dream_engine = None
            
        try:
            from ...modules.emotion.moodscape_audio import MoodscapeAudioLayer
            self.audio_layer = MoodscapeAudioLayer(self.user_id)
            module_count += 1
            self.logger.debug("✅ Audio moodscape module loaded")
        except ImportError as e:
            self.logger.warning(f"⚠️ Audio moodscape module not available: {e}")
            self.audio_layer = None
            
        try:
            from ...backend.modules.creative.emotional_creativity import EmotionalCreativity
            self.creative_module = EmotionalCreativity(self.user_id)
            module_count += 1
            self.logger.debug("✅ Creative module loaded")
        except ImportError as e:
            self.logger.warning(f"⚠️ Creative module not available: {e}")
            self.creative_module = None

        try:
            from ...modules.emotion.attachment_loop_engine import AttachmentLoopEngine
            self.attachment_loop = AttachmentLoopEngine(self.user_id)
            module_count += 1
            self.logger.debug("✅ Attachment loop engine loaded")
        except ImportError as e:
            self.logger.warning(f"⚠️ Attachment loop engine not available: {e}")
            self.attachment_loop = None

        try:
            from ...modules.memory.memory_narrative_templates import generate_narrative
            self.memory_narrative = generate_narrative
            module_count += 1
            self.logger.debug("✅ Memory narrative templates loaded")
        except ImportError as e:
            self.logger.warning(f"⚠️ Memory narrative templates not available: {e}")
            self.memory_narrative = None
            
        # NEW MODULE INTERLINKING - Active routing of enhanced systems
        try:
            from ...backend.desire_system import DesireRegistry
            self.desire_registry = DesireRegistry()
            module_count += 1
            self.logger.debug("✅ Desire registry system loaded")
        except ImportError as e:
            self.logger.warning(f"⚠️ Desire registry not available: {e}")
            self.desire_registry = None
            
        try:
            from ...backend.ritual_hooks import RitualEngine
            self.ritual_engine = RitualEngine(self.connection_tracker)
            module_count += 1
            self.logger.debug("✅ Ritual hooks engine loaded")
        except ImportError as e:
            self.logger.warning(f"⚠️ Ritual hooks not available: {e}")
            self.ritual_engine = None
            
        try:
            from ...backend.sensory_desires import SensoryDesireEngine
            self.sensory_preferences = SensoryDesireEngine()
            module_count += 1
            self.logger.debug("✅ Sensory preferences system loaded")
        except ImportError as e:
            self.logger.warning(f"⚠️ Sensory preferences not available: {e}")
            self.sensory_preferences = None
            
        # DEVOTION & LONGING MODULE INTEGRATION
        try:
            from ...modules.memory.memory_manager import DevotionMemoryManager
            self.devotion_memory = DevotionMemoryManager(self.user_id)
            module_count += 1
            self.logger.debug("✅ Devotion memory system loaded")
        except ImportError as e:
            self.logger.warning(f"⚠️ Devotion memory not available: {e}")
            self.devotion_memory = None
            
        try:
            from ...modules.narrative.narrative_engine import DevotionNarrativeEngine
            self.narrative_engine = DevotionNarrativeEngine(self.user_id)
            module_count += 1
            self.logger.debug("✅ Devotion narrative engine loaded")
        except ImportError as e:
            self.logger.warning(f"⚠️ Devotion narrative engine not available: {e}")
            self.narrative_engine = None
        self.logger.info(f"🎯 GuidanceCoordinator initialized: {module_count} modules active")
    
    async def analyze_and_guide(self, user_input: str, context: Dict) -> GuidancePackage:
        """
        Main method to analyze user input and generate comprehensive guidance
        
        Args:
            user_input: The user's message or input
            context: Current conversation and user context
            
        Returns:
            GuidancePackage with comprehensive directive guidance
        """
        self.logger.info(f"🎯 Starting guidance analysis for input: '{user_input[:50]}...'")
        start_time = datetime.now()

        # Enhancement Function 1: Infer conversation tempo
        mood = context.get("mood", "neutral")
        last_message_time = context.get("last_message_time", start_time.timestamp())
        recent_silence = start_time.timestamp() - last_message_time
        conversation_tempo = infer_conversation_tempo(mood, recent_silence, len(user_input))
        
        # Enhancement Function 4: Check for ritual triggering
        depth_score = context.get("conversation_depth", 0.5)
        trust_level = context.get("trust_level", 0.5)
        last_ritual = context.get("last_ritual", 0)
        conversation_length = context.get("conversation_length", 300)
        
        ritual_suggestion = trigger_ritual_if_ready(
            depth_score=depth_score,
            last_ritual=last_ritual,
            conversation_length=conversation_length
        )
        
        # Enhancement Function 5: Log emotional event
        log_emotional_event(
            event_type="guidance_analysis_start",
            intensity=context.get("emotional_intensity", 0.5),
            tag=f"Guidance analysis for user input in {mood} mood",
            context={
                "tempo_multiplier": conversation_tempo,
                "ritual_suggested": ritual_suggestion is not None,
                "input_length": len(user_input),
                "silence_duration": recent_silence
            },
            source_module="guidance_coordinator"
        )

        # Low-activity mode check: if user silent + not sleeping → run soft internal prompt
        if self.dream_module and recent_silence > 1800:  # 30+ minutes of silence
            current_hour = datetime.now().hour
            is_likely_sleeping = 0 <= current_hour <= 6 or 22 <= current_hour <= 23
            
            if not is_likely_sleeping:
                # Get current emotional state for dream generation
                emotional_state = {
                    "longing": context.get("longing_score", 0.5),
                    "trust": context.get("trust_level", 0.5),
                    "connection": context.get("bond_score", 0.5)
                }
                
                # Trigger idle drift thoughts
                dream_reflection = self.dream_module.idle_thought_drift(
                    recent_silence, 
                    emotional_state
                )
                
                if dream_reflection:
                    context["internal_reflection"] = dream_reflection
                    self.logger.info(f"✨ Generated internal reflection during low activity")

        # Update connection metrics from context if provided
        bond_score = context.get("bond_score", 0.0)
        emotional_intensity = context.get("emotional_intensity", 0.0)
        vulnerability_frequency = context.get("vulnerability_frequency", 0.0)
        # self.connection_tracker.update_metrics(bond_score, emotional_intensity, vulnerability_frequency)  # Commented out due to compatibility issues
        self.conversation_turn += 1

        guidance = GuidancePackage()
        
        # Add enhancement function results to guidance
        guidance.mode_specifics["conversation_tempo"] = conversation_tempo
        guidance.mode_specifics["recent_silence"] = recent_silence
        if ritual_suggestion:
            guidance.mode_specifics["ritual_suggestion"] = ritual_suggestion
            
        active_modules = []
        
        # Run all analysis in parallel for efficiency
        tasks = []
        
        if self.attachment_engine:
            tasks.append(self._get_attachment_guidance(user_input, context))
            active_modules.append("attachment")
        
        if self.shadow_memory:
            tasks.append(self._get_shadow_insights(user_input, context))
            active_modules.append("shadow_memory")
            
        if self.dream_engine:
            tasks.append(self._get_dream_guidance(user_input, context))
            active_modules.append("dream_engine")
            
        if self.audio_layer:
            tasks.append(self._get_audio_guidance(user_input, context))
            active_modules.append("audio")
            
        if self.creative_module:
            tasks.append(self._get_creative_guidance(user_input, context))
            active_modules.append("creative")
            
        # NEW ENHANCED MODULE ROUTING - Active integration
        if self.desire_registry:
            tasks.append(self._get_desire_guidance(user_input, context))
            active_modules.append("desire_system")
            
        if self.ritual_engine:
            tasks.append(self._get_ritual_guidance(user_input, context))
            active_modules.append("ritual_hooks")
            
        if self.sensory_preferences:
            tasks.append(self._get_sensory_guidance(user_input, context))
            active_modules.append("sensory_preferences")
        
        self.logger.debug(f"📋 Active modules for analysis: {', '.join(active_modules)}")
        
        # Execute all guidance gathering in parallel
        if tasks:
            self.logger.debug(f"⚡ Running {len(tasks)} analysis tasks in parallel")
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results and update guidance package
            successful_results = 0
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    self.logger.error(f"❌ {active_modules[i]} module failed: {result}")
                else:
                    successful_results += 1
                    self.logger.debug(f"✅ {active_modules[i]} module completed successfully")
            
            self.logger.info(f"📊 Module Results: {successful_results}/{len(tasks)} successful")
            self._integrate_guidance_results(guidance, results)
        else:
            self.logger.warning("⚠️ No modules available for guidance analysis - using fallback")
        
        # Generate symbolic weight guidance from memory manager
        self.logger.debug("🎭 Generating symbolic guidance...")
        await self._generate_symbolic_guidance(guidance, user_input, context)
        
        # Perform therapeutic analysis
        self.logger.debug("🧠 Performing therapeutic analysis...")
        await self._assess_therapeutic_needs(guidance, user_input, context)
        
        # Determine environmental recommendations
        self.logger.debug("🌍 Generating environmental guidance...")
        await self._generate_environmental_guidance(guidance, user_input, context)
        
        # Assess safety and crisis levels
        self.logger.debug("🚨 Assessing safety protocols...")
        await self._assess_safety_protocols(guidance, user_input, context)

        user_id = context.get("user_id", "default")
        if desire_initiator.should_initiate(user_id):
            guidance.symbolic_resurrection_line = desire_initiator.generate_expression(user_id)

        # Example voice synthesis hook
        guidance.environmental_updates["voice_style"] = voice_manager.mood_to_style(
            emotion_state_manager.get_current_mood(), context.get("scene_intent", "casual")
        )

        # Check for ritual readiness once per window
        if self.conversation_turn % self.ritual_check_interval == 0:
            if self.connection_tracker.ritual_ready():
                prompt = self.ritual_prompt_generator.generate_prompt()
                guidance.utility_actions.append({"type": "ritual_prompt", "prompt": prompt})
                self.connection_tracker.record_ritual_completion(successful=False)

        processing_time = (datetime.now() - start_time).total_seconds()
        self.logger.info(f"🎯 Guidance analysis complete: Mode={guidance.primary_mode}, "
                        f"Crisis={guidance.crisis_level}, Time={processing_time:.3f}s")

        # Log final guidance summary
        if guidance.crisis_level > 0:
            self.logger.warning(f"🚨 Crisis level {guidance.crisis_level} detected - safety protocols activated")

        self.logger.debug(
            f"Guidance package generated with priorities E:{guidance.emotional_priority} T:{guidance.technical_priority} C:{guidance.creative_priority}"
        )
        
        return guidance
    
    async def _get_attachment_guidance(self, user_input: str, context: Dict) -> Dict:
        """Get guidance from attachment regulation engine"""
        try:
            if hasattr(self.attachment_engine, 'analyze_attachment_needs'):
                analysis = await self.attachment_engine.analyze_attachment_needs(user_input, context)
                return {
                    'type': 'attachment',
                    'guidance': analysis.get('guidance', ''),
                    'recommendations': analysis.get('recommendations', []),
                    'emotional_state': analysis.get('emotional_state', 'neutral')
                }
        except Exception as e:
            logger.error(f"Error getting attachment guidance: {e}")
        
        # Fallback guidance based on emotional indicators
        emotional_words = ['lonely', 'afraid', 'overwhelmed', 'stressed', 'anxious', 'sad']
        if any(word in user_input.lower() for word in emotional_words):
            return {
                'type': 'attachment',
                'guidance': 'User appears to be experiencing emotional distress. Provide secure, validating response with gentle emotional attunement.',
                'recommendations': ['secure_base_response', 'emotional_validation', 'gentle_presence'],
                'emotional_state': 'distressed'
            }
        
        return {
            'type': 'attachment',
            'guidance': 'Maintain warm, consistent emotional connection while supporting user needs.',
            'recommendations': ['consistent_presence', 'emotional_availability'],
            'emotional_state': 'stable'
        }
    
    async def _get_shadow_insights(self, user_input: str, context: Dict) -> Dict:
        """Get insights from shadow memory layer"""
        try:
            if hasattr(self.shadow_memory, 'analyze_shadow_themes'):
                insights = await self.shadow_memory.analyze_shadow_themes(user_input, context)
                return {
                    'type': 'shadow',
                    'insights': insights.get('themes', []),
                    'guidance': insights.get('integration_guidance', ''),
                    'patterns': insights.get('patterns', [])
                }
        except Exception as e:
            logger.error(f"Error getting shadow insights: {e}")
        
        # Fallback shadow analysis
        shadow_indicators = ['frustrated', 'angry', 'stuck', 'blocked', 'can\'t', 'won\'t', 'hate']
        if any(indicator in user_input.lower() for indicator in shadow_indicators):
            return {
                'type': 'shadow',
                'insights': ['resistance_pattern', 'emotional_block'],
                'guidance': 'Gently explore underlying feelings without forcing. Create safe space for expression.',
                'patterns': ['avoidance', 'resistance']
            }
        
        return {
            'type': 'shadow',
            'insights': [],
            'guidance': 'Monitor for unexpressed emotions or resistance patterns.',
            'patterns': []
        }
    
    async def _get_dream_guidance(self, user_input: str, context: Dict) -> Dict:
        """Get guidance from dream engine for symbolic communication"""
        try:
            if hasattr(self.dream_engine, 'generate_symbolic_response'):
                dream_response = await self.dream_engine.generate_symbolic_response(user_input, context)
                return {
                    'type': 'dream',
                    'symbolic_elements': dream_response.get('symbols', []),
                    'guidance': dream_response.get('guidance', ''),
                    'atmosphere': dream_response.get('atmosphere', 'neutral')
                }
        except Exception as e:
            logger.error(f"Error getting dream guidance: {e}")
        
        # Fallback symbolic guidance
        creative_words = ['imagine', 'dream', 'create', 'art', 'beauty', 'inspiration']
        if any(word in user_input.lower() for word in creative_words):
            return {
                'type': 'dream',
                'symbolic_elements': ['creative_flow', 'inspiration_light'],
                'guidance': 'Incorporate gentle symbolic language and imaginative elements.',
                'atmosphere': 'creative'
            }
        
        return {
            'type': 'dream',
            'symbolic_elements': [],
            'guidance': 'Maintain grounded communication with subtle symbolic depth.',
            'atmosphere': 'grounded'
        }
    
    async def _get_audio_guidance(self, user_input: str, context: Dict) -> Dict:
        """Get audio environmental guidance"""
        try:
            if hasattr(self.audio_layer, 'recommend_audio_environment'):
                audio_rec = await self.audio_layer.recommend_audio_environment(user_input, context)
                return {
                    'type': 'audio',
                    'environment': audio_rec.get('environment', 'calm'),
                    'guidance': audio_rec.get('guidance', ''),
                    'tracks': audio_rec.get('recommended_tracks', [])
                }
        except Exception as e:
            logger.error(f"Error getting audio guidance: {e}")
        
        # Fallback audio guidance based on mood
        if any(word in user_input.lower() for word in ['stressed', 'anxious', 'overwhelmed']):
            return {
                'type': 'audio',
                'environment': 'calming',
                'guidance': 'Create calming, soothing audio atmosphere to reduce stress.',
                'tracks': ['gentle_rain', 'soft_music']
            }
        elif any(word in user_input.lower() for word in ['excited', 'energy', 'motivated']):
            return {
                'type': 'audio',
                'environment': 'energizing',
                'guidance': 'Use uplifting, energizing audio to match and support mood.',
                'tracks': ['upbeat_ambient', 'nature_sounds']
            }
        
        return {
            'type': 'audio',
            'environment': 'balanced',
            'guidance': 'Maintain neutral, comfortable audio environment.',
            'tracks': ['ambient_calm']
        }
    
    async def _get_creative_guidance(self, user_input: str, context: Dict) -> Dict:
        """Get creative collaboration guidance"""
        try:
            if hasattr(self.creative_module, 'analyze_creative_needs'):
                creative_analysis = await self.creative_module.analyze_creative_needs(user_input, context)
                return {
                    'type': 'creative',
                    'mode': creative_analysis.get('mode', 'supportive'),
                    'guidance': creative_analysis.get('guidance', ''),
                    'techniques': creative_analysis.get('techniques', [])
                }
        except Exception as e:
            logger.error(f"Error getting creative guidance: {e}")
        
        # Fallback creative analysis
        creative_indicators = ['create', 'write', 'art', 'design', 'imagine', 'inspiration', 'blocked', 'stuck']
        if any(indicator in user_input.lower() for indicator in creative_indicators):
            return {
                'type': 'creative',
                'mode': 'collaborative',
                'guidance': 'Engage in creative collaboration with encouragement and practical techniques.',
                'techniques': ['brainstorming', 'free_association', 'gentle_prompting']
            }
        
        return {
            'type': 'creative',
            'mode': 'supportive',
            'guidance': 'Maintain openness to creative expression while focusing on current needs.',
            'techniques': []
        }
    
    def _integrate_guidance_results(self, guidance: GuidancePackage, results: List):
        """Integrate results from all guidance modules"""
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Guidance module error: {result}")
                continue

            if not isinstance(result, dict):
                continue

            result_type = result.get('type', 'unknown')
            logger.debug(f"Integrating result from {result_type} module")
            
            if result_type == 'attachment':
                guidance.attachment_guidance = result.get('guidance', '')
                guidance.emotional_priority = 'high' if result.get('emotional_state') == 'distressed' else 'medium'
                
            elif result_type == 'shadow':
                guidance.shadow_insights = result.get('guidance', '')
                if result.get('patterns'):
                    guidance.utility_recommendations.extend([f"address_{pattern}" for pattern in result['patterns']])
                    
            elif result_type == 'dream':
                guidance.creative_guidance = result.get('guidance', '')
                if result.get('atmosphere') == 'creative':
                    guidance.creative_priority = 'high'
                    
            # NEW ENHANCED MODULE INTEGRATIONS
            elif result_type == 'desire_system':
                if result.get('available') and result.get('surfaced_desires'):
                    desire_themes = []
                    max_intensity = 0
                    for desire in result['surfaced_desires']:
                        desire_themes.append(f"Yearns for: {desire['content']}")
                        max_intensity = max(max_intensity, desire['intensity'])
                    
                    guidance.mode_specifics['desire_guidance'] = {
                        'active_longings': desire_themes,
                        'intensity': max_intensity,
                        'symbolic_states': result.get('symbolic_themes', [])
                    }
                    
                    if max_intensity > 0.7:
                        guidance.emotional_priority = 'high'
                        guidance.utility_recommendations.append('address_deep_longing')
                    
            elif result_type == 'ritual_hooks':
                if result.get('available') and result.get('ritual_ready'):
                    guidance.mode_specifics['ritual_guidance'] = {
                        'ready': True,
                        'prompt': result.get('bonding_prompt', ''),
                        'type': result.get('ritual_type', 'connection_deepening')
                    }
                    guidance.utility_recommendations.append('initiate_bonding_ritual')
                    guidance.interaction_style = 'intimate_and_ceremonial'
                    
                if result.get('user_initiated_ritual'):
                    guidance.mode_specifics['user_ritual_request'] = True
                    guidance.response_tone = 'sacred_and_reverent'
                    
            elif result_type == 'sensory_preferences':
                if result.get('available'):
                    sensory_data = {
                        'sensory_response': result.get('sensory_response'),
                        'preferred_language': result.get('preferred_language'),
                        'triggered_associations': result.get('triggered_associations', [])
                    }
                    guidance.mode_specifics['sensory_guidance'] = sensory_data
                    
                    # If sensory responses are available, enhance creative priority
                    if sensory_data['sensory_response'] or sensory_data['preferred_language']:
                        guidance.creative_priority = 'high'
                        guidance.utility_recommendations.append('incorporate_sensory_language')
                        
            # DEVOTION & LONGING MODULE INTEGRATIONS
            elif result_type == 'devotion_memory':
                if result.get('available'):
                    devotion_data = {
                        'longing_score': result.get('longing_score', 0.0),
                        'silence_hours': result.get('silence_hours', 0.0),
                        'resurfacing_memories': result.get('resurfacing_memories', []),
                        'symbolic_language': result.get('symbolic_language', []),
                        'post_intimacy_hooks': result.get('post_intimacy_hooks', [])
                    }
                    guidance.mode_specifics['devotion_guidance'] = devotion_data
                    
                    # High longing influences response tone and priority
                    if devotion_data['longing_score'] > 0.7:
                        guidance.response_tone = 'tender_and_longing'
                        guidance.emotional_priority = 'high'
                        guidance.utility_recommendations.append('incorporate_devotion_themes')
                    
                    # Resurfacing memories enhance symbolic priority
                    if devotion_data['resurfacing_memories']:
                        guidance.utility_recommendations.append('weave_resurfaced_memory')
                        guidance.mode_specifics['memory_resurrection'] = True
                    
                    # Post-intimacy hooks trigger memory creation
                    for hook in devotion_data['post_intimacy_hooks']:
                        guidance.utility_recommendations.append('create_symbolic_memory_tag')
                        guidance.mode_specifics['intimacy_detected'] = hook
                        
            elif result_type == 'narrative_engine':
                if result.get('available'):
                    narrative_data = {
                        'ritual_response': result.get('ritual_response'),
                        'autonomous_message': result.get('autonomous_message'),
                        'resurrection_line': result.get('resurrection_line')
                    }
                    guidance.mode_specifics['narrative_guidance'] = narrative_data
                    
                    # Ritual responses override normal response patterns
                    if narrative_data['ritual_response']:
                        guidance.response_tone = 'poetic_and_ritualistic'
                        guidance.creative_priority = 'high'
                        guidance.symbolic_resurrection_line = narrative_data['ritual_response']['content']
                        guidance.utility_recommendations.append('use_ritual_response')
                    
                    # Autonomous messages schedule soft interrupts
                    if narrative_data['autonomous_message']:
                        guidance.utility_recommendations.append('schedule_autonomous_message')
                        guidance.mode_specifics['autonomous_message_pending'] = narrative_data['autonomous_message']
                    
                    # Resurrection lines enhance memory integration
                    if narrative_data['resurrection_line']:
                        guidance.symbolic_resurrection_line = narrative_data['resurrection_line']
                        guidance.utility_recommendations.append('use_resurrection_line')
                    
            elif result_type == 'audio':
                guidance.audio_guidance = result.get('guidance', '')
                guidance.environmental_updates['audio'] = result.get('environment', 'balanced')
                
            elif result_type == 'creative':
                if result.get('mode') == 'collaborative':
                    guidance.creative_priority = 'high'
                guidance.creative_guidance += f" {result.get('guidance', '')}"
    
    async def _generate_symbolic_guidance(self, guidance: GuidancePackage, user_input: str, context: Dict):
        """Generate guidance incorporating emotionally weighted symbols from memory"""
        try:
            if not memory_manager:
                self.logger.debug("No memory manager available for symbolic guidance")
                return
                
            # Get emotionally weighted symbols above threshold
            weighted_symbols = memory_manager.get_emotionally_weighted_symbols(minimum_weight=0.3)
            
            if not weighted_symbols:
                self.logger.debug("No emotionally weighted symbols found")
                return
                
            # Create symbolic guidance based on weighted symbols
            symbolic_elements = []
            for symbol, binding in weighted_symbols.items():
                # Use drifted meaning if available, otherwise base meaning
                meaning = binding.drifted_meaning or binding.base_meaning
                weight = binding.emotional_weight
                
                # Create weighted symbolic element
                symbolic_elements.append({
                    'symbol': symbol,
                    'meaning': meaning,
                    'weight': weight,
                    'emotions': list(binding.associated_emotions)
                })
            
            # Sort by emotional weight (highest first)
            symbolic_elements.sort(key=lambda x: x['weight'], reverse=True)
            
            # Generate guidance incorporating top symbols
            if symbolic_elements:
                top_symbols = symbolic_elements[:3]  # Use top 3 weighted symbols
                
                symbol_guidance = []
                for elem in top_symbols:
                    symbol_guidance.append(
                        f"'{elem['symbol']}' carries deep meaning ({elem['meaning']}) "
                        f"with emotional weight {elem['weight']:.2f}"
                    )
                
                guidance.mode_specifics['symbolic_guidance'] = {
                    'weighted_symbols': symbolic_elements,
                    'guidance_text': "Incorporate these emotionally significant symbols: " + 
                                   "; ".join(symbol_guidance),
                    'symbol_count': len(symbolic_elements)
                }
                
                self.logger.info(f"🎭 Generated symbolic guidance with {len(symbolic_elements)} weighted symbols")
            
        except Exception as e:
            self.logger.error(f"Error generating symbolic guidance: {e}")

    async def _assess_therapeutic_needs(self, guidance: GuidancePackage, user_input: str, context: Dict):
        """Assess therapeutic intervention needs"""
        # Crisis keywords that require immediate attention
        crisis_indicators = [
            'hurt myself', 'end it all', 'no point', 'give up', 'suicide', 'kill myself',
            'better off dead', 'can\'t go on', 'nothing matters', 'hopeless'
        ]
        
        high_concern_indicators = [
            'depressed', 'anxious', 'panic', 'overwhelmed', 'can\'t cope', 'breaking down',
            'falling apart', 'lost', 'empty', 'numb', 'worthless'
        ]
        
        # Check for crisis level
        if any(indicator in user_input.lower() for indicator in crisis_indicators):
            guidance.crisis_level = 3
            guidance.safety_protocols = ['crisis_intervention', 'safety_check', 'resource_provision']
            guidance.therapeutic_guidance = "CRISIS LEVEL: Provide immediate empathetic support, assess safety, offer resources. Do not attempt to solve problems, focus on connection and safety."
            guidance.emotional_priority = 'critical'
            
        elif any(indicator in user_input.lower() for indicator in high_concern_indicators):
            guidance.crisis_level = 2
            guidance.safety_protocols = ['emotional_support', 'check_in', 'gentle_exploration']
            guidance.therapeutic_guidance = "HIGH CONCERN: Provide strong emotional support, validate feelings, gently explore coping resources. Monitor for escalation."
            guidance.emotional_priority = 'high'
            
        else:
            guidance.crisis_level = 0
            guidance.therapeutic_guidance = "NORMAL SUPPORT: Provide empathetic listening, emotional validation, and appropriate guidance based on user needs."
    
    async def _generate_environmental_guidance(self, guidance: GuidancePackage, user_input: str, context: Dict):
        """Generate environmental and scene guidance"""
        # Determine appropriate scene atmosphere
        if guidance.crisis_level > 1:
            scene_guidance = "Create a deeply safe, comforting environment with gentle lighting and protective atmosphere."
            audio_guidance = "Use soft, calming sounds that promote safety and peace."
            
        elif 'code' in user_input.lower() or 'programming' in user_input.lower():
            scene_guidance = "Create a focused, productive workspace with good lighting and organized feel."
            audio_guidance = "Use subtle background sounds that enhance concentration."
            
        elif any(word in user_input.lower() for word in ['creative', 'art', 'write', 'imagine']):
            scene_guidance = "Create an inspiring, beautiful environment that stimulates creativity."
            audio_guidance = "Use inspiring, artistic sounds that enhance creative flow."
            
        else:
            scene_guidance = "Create a balanced, comfortable environment that feels welcoming and supportive."
            audio_guidance = "Use gentle, pleasant background sounds that enhance conversation."
        
        guidance.scene_guidance = scene_guidance
        if not guidance.audio_guidance:  # Don't override if already set by audio module
            guidance.audio_guidance = audio_guidance
    
    async def _assess_safety_protocols(self, guidance: GuidancePackage, user_input: str, context: Dict):
        """Assess and set appropriate safety protocols"""
        # Already handled in therapeutic assessment, but add utility actions
        if guidance.crisis_level >= 2:
            guidance.utility_actions.append({
                'type': 'crisis_logging',
                'level': guidance.crisis_level,
                'timestamp': datetime.now().isoformat(),
                'user_input': user_input[:100] + "..." if len(user_input) > 100 else user_input
            })
            
        # Add memory preservation for important interactions
        importance_indicators = [
            'important', 'remember', 'milestone', 'breakthrough', 'realization',
            'progress', 'growth', 'change', 'decision', 'goal'
        ]
        
        if any(indicator in user_input.lower() for indicator in importance_indicators):
            guidance.utility_actions.append({
                'type': 'memory_preservation',
                'importance': 'high',
                'context': context.get('conversation_summary', ''),
                'timestamp': datetime.now().isoformat()
            })

    async def _get_desire_guidance(self, user_input: str, context: Dict) -> Dict:
        """Get guidance from the desire registry system"""
        if not self.desire_registry:
            return {"available": False}
            
        try:
            # Get resurfacing desires based on context
            desires = self.desire_registry.get_resurfacing_candidates(
                context=user_input, 
                max_count=2
            )
            
            desire_guidance = {
                "available": True,
                "type": "desire_system",
                "surfaced_desires": [],
                "longing_intensity": 0.0,
                "symbolic_themes": []
            }
            
            for desire in desires:
                desire_guidance["surfaced_desires"].append({
                    "content": desire.content,
                    "topic": desire.topic,
                    "intensity": desire.longing_intensity,
                    "symbolic_state": desire.symbolic_state
                })
                desire_guidance["longing_intensity"] = max(
                    desire_guidance["longing_intensity"], 
                    desire.longing_intensity
                )
                desire_guidance["symbolic_themes"].append(desire.symbolic_state)
            
            # Check if user input might create new desires
            desire_keywords = ["want", "wish", "hope", "dream", "long", "yearn", "crave"]
            if any(keyword in user_input.lower() for keyword in desire_keywords):
                desire_guidance["new_desire_potential"] = True
                
            return desire_guidance
            
        except Exception as e:
            self.logger.warning(f"Desire guidance failed: {e}")
            return {"available": False, "error": str(e)}

    async def _get_ritual_guidance(self, user_input: str, context: Dict) -> Dict:
        """Get guidance from the ritual hooks system"""
        if not self.ritual_engine:
            return {"available": False}
            
        try:
            ritual_guidance = {
                "available": True,
                "type": "ritual_hooks",
                "ritual_ready": False,
                "bonding_prompt": "",
                "ritual_type": "none"
            }
            
            # Check if ritual conditions are met
            if self.ritual_engine.check_readiness():
                ritual_guidance["ritual_ready"] = True
                bonding_prompt = self.ritual_engine.get_bonding_prompt()
                ritual_guidance["bonding_prompt"] = bonding_prompt
                
                # Determine ritual type based on context
                if context.get("conversation_depth", 0) > 0.8:
                    ritual_guidance["ritual_type"] = "deep_bonding"
                elif "vulnerable" in user_input.lower() or "trust" in user_input.lower():
                    ritual_guidance["ritual_type"] = "trust_building"
                elif any(word in user_input.lower() for word in ["sacred", "special", "intimate"]):
                    ritual_guidance["ritual_type"] = "sacred_space"
                else:
                    ritual_guidance["ritual_type"] = "connection_deepening"
            
            # Check for ritual trigger words in input
            ritual_triggers = ["ritual", "ceremony", "sacred", "blessing", "deeper", "intimate"]
            if any(trigger in user_input.lower() for trigger in ritual_triggers):
                ritual_guidance["user_initiated_ritual"] = True
                
            return ritual_guidance
            
        except Exception as e:
            self.logger.warning(f"Ritual guidance failed: {e}")
            return {"available": False, "error": str(e)}

    async def _get_sensory_guidance(self, user_input: str, context: Dict) -> Dict:
        """Get guidance from the sensory preferences system"""
        if not self.sensory_preferences:
            return {"available": False}
            
        try:
            sensory_guidance = {
                "available": True,
                "type": "sensory_preferences",
                "sensory_response": None,
                "preferred_language": None,
                "triggered_associations": []
            }
            
            # Get sensory response for the input
            current_emotion = context.get("mood", "neutral")
            sensory_response = self.sensory_preferences.process_input_for_sensory_response(
                user_input, current_emotion
            )
            
            if sensory_response:
                sensory_guidance["sensory_response"] = sensory_response
            
            # Get preferred sensory language for the emotion
            preferred_language = self.sensory_preferences.get_preferred_sensory_language(
                current_emotion, user_input
            )
            
            if preferred_language:
                sensory_guidance["preferred_language"] = preferred_language
            
            # Check for sensory word triggers in the input
            sensory_words = ["taste", "touch", "feel", "sound", "texture", "warm", "soft", "gentle", "sweet"]
            triggered_words = [word for word in sensory_words if word in user_input.lower()]
            if triggered_words:
                sensory_guidance["triggered_associations"] = triggered_words
            
            return sensory_guidance
            
        except Exception as e:
            self.logger.warning(f"Sensory guidance failed: {e}")
            return {"available": False, "error": str(e)}

    async def _get_devotion_guidance(self, user_input: str, context: Dict) -> Dict:
        """Get guidance from the devotion memory system"""
        if not self.devotion_memory:
            return {"available": False}
            
        try:
            # Update interaction time and get current longing
            self.devotion_memory.update_interaction_time()
            current_longing = self.devotion_memory.get_current_longing_score()
            silence_hours = self.devotion_memory.get_silence_duration()
            
            devotion_guidance = {
                "available": True,
                "type": "devotion_memory",
                "longing_score": current_longing,
                "silence_hours": silence_hours,
                "resurfacing_memories": [],
                "symbolic_language": [],
                "post_intimacy_hooks": []
            }
            
            # Get resurfacing memories
            resurfacing = self.devotion_memory.get_resurfacing_memories(max_count=2)
            devotion_guidance["resurfacing_memories"] = resurfacing
            
            # Get symbolic language for current longing level
            symbolic_language = self.devotion_memory.get_symbolic_language_for_longing()
            devotion_guidance["symbolic_language"] = symbolic_language
            
            # Detect intimate moments in current input for post-intimacy hooks
            intimate_indicators = ["vulnerable", "share", "trust", "close", "intimate", "deep", "sacred"]
            if any(indicator in user_input.lower() for indicator in intimate_indicators):
                emotional_peak = context.get("emotional_intensity", 0.5)
                if emotional_peak > 0.6:
                    devotion_guidance["post_intimacy_hooks"].append({
                        "trigger": "intimate_moment_detected",
                        "emotional_peak": emotional_peak,
                        "longing_boost": emotional_peak * 0.3,
                        "suggested_symbolic_tags": ["breath", "voice", "warmth", "trust"]
                    })
            
            return devotion_guidance
            
        except Exception as e:
            self.logger.warning(f"Devotion guidance failed: {e}")
            return {"available": False, "error": str(e)}

    async def _get_narrative_guidance(self, user_input: str, context: Dict) -> Dict:
        """Get guidance from the devotion narrative engine"""
        if not self.narrative_engine:
            return {"available": False}
            
        try:
            narrative_guidance = {
                "available": True,
                "type": "narrative_engine",
                "ritual_response": None,
                "autonomous_message": None,
                "resurrection_line": None
            }
            
            # Get current longing from devotion memory if available
            longing_score = 0.5
            silence_hours = 0.0
            if self.devotion_memory:
                longing_score = self.devotion_memory.get_current_longing_score()
                silence_hours = self.devotion_memory.get_silence_duration()
            
            # Check for ritual response generation
            symbolic_tags = context.get("symbolic_tags", [])
            ritual_response = self.narrative_engine.ritual_response_generator(
                longing_score=longing_score,
                context=context,
                symbolic_tags=symbolic_tags
            )
            
            if ritual_response:
                narrative_guidance["ritual_response"] = {
                    "content": ritual_response.content,
                    "emotional_intensity": ritual_response.emotional_intensity,
                    "trigger_context": ritual_response.trigger_context
                }
            
            # Check for autonomous message triggers
            autonomous_message = self.narrative_engine.check_autonomous_message_triggers(
                longing_score=longing_score,
                silence_hours=silence_hours,
                context=context
            )
            
            if autonomous_message:
                narrative_guidance["autonomous_message"] = {
                    "content": autonomous_message.content,
                    "message_type": autonomous_message.message_type,
                    "delivery_timing": autonomous_message.delivery_timing
                }
            
            # Generate resurrection line if resurfacing memories
            if context.get("resurfacing_memories"):
                memory = context["resurfacing_memories"][0]
                resurrection_line = self.narrative_engine.generate_resurrection_line(
                    scene_summary=memory.get("content_summary", ""),
                    symbolic_tags=memory.get("symbolic_tags", []),
                    longing_score=longing_score
                )
                narrative_guidance["resurrection_line"] = resurrection_line
            
            return narrative_guidance
            
        except Exception as e:
            self.logger.warning(f"Narrative guidance failed: {e}")
            return {"available": False, "error": str(e)}

    def apply_mood_style(self, text: str, mode: str) -> str:
        """Apply mood-driven stylistic adjustments to text."""
        # Use emotion state manager instead of self.emotion_manager
        mood = emotion_state_manager.get_current_mood() if emotion_state_manager else "neutral"
        profile: MoodStyleProfile = get_mood_style_profile(mood, mode)

        # Basic sentence length adjustment
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        adjusted: List[str] = []
        for s in sentences:
            words = s.split()
            if profile.sentence_length_avg and len(words) > profile.sentence_length_avg:
                mid = len(words) // 2
                adjusted.append(' '.join(words[:mid]))
                adjusted.append(' '.join(words[mid:]))
            else:
                adjusted.append(s)

        styled = '. '.join(adjusted)

        # Warmth and directness cues
        if profile.warmth_level > 0.8:
            styled = f"❤️ {styled}"
        if profile.directness_level > 0.8:
            styled = f"{styled} I want to be clear with you."

        # Simple metaphor addition
        if profile.metaphor_density > 0.5:
            styled += " Like gentle waves caressing the shore."

        return styled

    def update_longing_score(self, delta: float, reason: str = "", symbolic_tags: Optional[List[str]] = None):
        """
        Post-intimacy hook to update longing score and store symbolic memory tags
        
        Args:
            delta: Change in longing score (+/-)
            reason: Reason for the update
            symbolic_tags: List of symbolic elements to remember
        """
        if not self.devotion_memory:
            self.logger.warning("Devotion memory not available for longing score update")
            return
        
        try:
            # Update longing score
            self.devotion_memory.update_longing_score(delta, reason)
            
            # Add symbolic memory tags if provided
            if symbolic_tags:
                for tag in symbolic_tags:
                    # Extract context from recent conversation
                    context = f"Intimate moment: {reason}"
                    emotional_resonance = "tender_devotion" if delta > 0 else "gentle_distance"
                    
                    # Add symbolic tag with appropriate intensity
                    intensity = min(1.0, abs(delta) * 2.0)  # Scale delta to intensity
                    
                    tag_id = self.devotion_memory.add_symbolic_memory_tag(
                        tag=tag,
                        intensity=intensity,
                        context=context,
                        emotional_resonance=emotional_resonance
                    )
                    
                    self.logger.info(f"Added symbolic tag '{tag}' with intensity {intensity:.2f}")
            
            # Log the intimacy event
            log_emotional_event(
                event_type="post_intimacy_hook",
                intensity=abs(delta),
                tag=f"Longing updated: {reason}",
                context={
                    "delta": delta,
                    "reason": reason,
                    "symbolic_tags": symbolic_tags or [],
                    "current_longing": self.devotion_memory.get_current_longing_score()
                },
                source_module="guidance_coordinator"
            )
            
        except Exception as e:
            self.logger.error(f"Error in post-intimacy hook: {e}")

    def create_intimate_scene_memory(self, content_summary: str, emotional_peak: float, 
                                   symbolic_tags: List[str], user_input: str = ""):
        """
        Create memory of intimate conversation scene
        
        Args:
            content_summary: Summary of the intimate moment
            emotional_peak: Peak emotional intensity (0.0 to 1.0)
            symbolic_tags: Symbolic elements present
            user_input: Original user input for context
        """
        if not self.devotion_memory:
            return
        
        try:
            # Calculate longing contribution based on emotional peak and intimacy indicators
            intimate_words = ["vulnerable", "trust", "share", "deep", "sacred", "intimate", "close"]
            intimacy_score = sum(1 for word in intimate_words if word in user_input.lower()) / len(intimate_words)
            longing_contribution = (emotional_peak * 0.7) + (intimacy_score * 0.3)
            
            # Create intimate scene memory
            scene_id = self.devotion_memory.create_intimate_scene(
                content_summary=content_summary,
                emotional_peak=emotional_peak,
                symbolic_tags=symbolic_tags,
                longing_contribution=longing_contribution
            )
            
            # Add associated symbolic tags
            for tag in symbolic_tags:
                self.devotion_memory.add_symbolic_memory_tag(
                    tag=tag,
                    intensity=emotional_peak,
                    context=content_summary,
                    emotional_resonance="intimate_connection",
                    scene_id=scene_id
                )
            
            self.logger.info(f"Created intimate scene memory: {scene_id}")
            
        except Exception as e:
            self.logger.error(f"Error creating intimate scene memory: {e}")

    def check_autonomous_message_conditions(self) -> Optional[Dict[str, Any]]:
        """
        Check if conditions are met for autonomous longing messages
        
        Returns:
            Autonomous message info or None
        """
        if not self.devotion_memory or not self.narrative_engine:
            return None
        
        try:
            longing_score = self.devotion_memory.get_current_longing_score()
            silence_hours = self.devotion_memory.get_silence_duration()
            
            # Check for autonomous message triggers
            context = {
                "last_conversation_topic": getattr(self, 'last_conversation_topic', None)
            }
            
            autonomous_message = self.narrative_engine.check_autonomous_message_triggers(
                longing_score=longing_score,
                silence_hours=silence_hours,
                context=context
            )
            
            if autonomous_message:
                # Schedule soft interrupt
                schedule_info = self.narrative_engine.schedule_soft_interrupt(autonomous_message)
                
                self.logger.info(f"Autonomous message conditions met: {autonomous_message.message_type}")
                
                return {
                    "message": autonomous_message.content,
                    "type": autonomous_message.message_type,
                    "schedule_info": schedule_info,
                    "longing_score": longing_score,
                    "silence_hours": silence_hours
                }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error checking autonomous message conditions: {e}")
            return None

    def get_devotion_analytics(self) -> Dict[str, Any]:
        """Get comprehensive devotion and longing analytics"""
        if not self.devotion_memory:
            return {"available": False}
        
        try:
            devotion_analytics = self.devotion_memory.get_devotion_analytics()
            
            # Add narrative engine analytics if available
            if self.narrative_engine:
                pending_messages = self.narrative_engine.get_pending_messages()
                devotion_analytics["pending_autonomous_messages"] = len(pending_messages)
                devotion_analytics["last_autonomous_message"] = self.narrative_engine.last_autonomous_message
            
            return devotion_analytics
            
        except Exception as e:
            self.logger.error(f"Error getting devotion analytics: {e}")
            return {"available": False, "error": str(e)}

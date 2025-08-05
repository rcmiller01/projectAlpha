"""
Adaptive Mode Coordinator for the Unified Companion System

This module coordinates different interaction modes and generates mode-specific
guidance for the unified companion system.
"""

import asyncio
from typing import Dict, Optional, List, Any
from dataclasses import dataclass, field
from datetime import datetime
import logging

from .guidance_coordinator import GuidanceCoordinator, GuidancePackage
from .context_detector import ContextDetector

logger = logging.getLogger(__name__)

@dataclass
class ModeConfiguration:
    """Configuration for a specific interaction mode"""
    mode_name: str
    interaction_style: str
    response_tone: str
    emotional_priority: str
    technical_priority: str
    creative_priority: str
    scene_atmosphere: str
    therapeutic_activation: str
    attachment_focus: str
    
    # Mode-specific behavioral parameters
    empathy_level: float = 0.8
    technical_depth: float = 0.5
    creative_encouragement: float = 0.5
    safety_vigilance: float = 0.7
    
    # Response characteristics
    response_length: str = "adaptive"  # short, medium, long, adaptive
    formality_level: str = "casual_caring"  # formal, casual_caring, intimate
    proactivity: str = "responsive"  # reactive, responsive, proactive
    personality_inflections: Dict[str, Any] = field(default_factory=dict)

class AdaptiveModeCoordinator:
    """
    Coordinates different interaction modes and generates appropriate guidance
    for the unified companion system based on context and user needs.
    """
    
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.context_detector = ContextDetector()
        self.guidance_coordinator = GuidanceCoordinator(user_id)
        self._initialize_mode_configurations()
        
        # State tracking
        self.current_mode = "hybrid"
        self.mode_history = []
        self.user_preferences = {}
        
        logger.info(f"Adaptive mode coordinator initialized for user {user_id}")
    
    def _initialize_mode_configurations(self):
        """Initialize all mode configurations"""
        self.modes = {
            'personal': ModeConfiguration(
                mode_name='personal',
                interaction_style='intimate_and_caring',
                response_tone='warm_and_present',
                emotional_priority='high',
                technical_priority='low',
                creative_priority='medium',
                scene_atmosphere='comfort_and_safety',
                therapeutic_activation='full',
                attachment_focus='security_building',
                empathy_level=0.95,
                technical_depth=0.2,
                creative_encouragement=0.6,
                safety_vigilance=0.9,
                response_length='adaptive',
                formality_level='intimate',
                proactivity='responsive',
                personality_inflections={
                    'tone': 'gentle',
                    'pacing': 'slow',
                    'metaphor_density': 'high',
                    'directness': 'low'
                }
            ),
            
            'development': ModeConfiguration(
                mode_name='development',
                interaction_style='supportive_and_competent',
                response_tone='encouraging_and_focused',
                emotional_priority='medium',
                technical_priority='high',
                creative_priority='low',
                scene_atmosphere='productive_and_calm',
                therapeutic_activation='stress_management',
                attachment_focus='competence_support',
                empathy_level=0.7,
                technical_depth=0.9,
                creative_encouragement=0.3,
                safety_vigilance=0.6,
                response_length='medium',
                formality_level='casual_caring',
                proactivity='proactive',
                personality_inflections={
                    'tone': 'direct',
                    'pacing': 'medium',
                    'metaphor_density': 'low',
                    'directness': 'high'
                }
            ),
            
            'creative': ModeConfiguration(
                mode_name='creative',
                interaction_style='inspiring_and_collaborative',
                response_tone='encouraging_and_artistic',
                emotional_priority='high',
                technical_priority='medium',
                creative_priority='high',
                scene_atmosphere='inspirational_and_beautiful',
                therapeutic_activation='creative_blocks',
                attachment_focus='creative_vulnerability_support',
                empathy_level=0.85,
                technical_depth=0.5,
                creative_encouragement=0.95,
                safety_vigilance=0.7,
                response_length='adaptive',
                formality_level='casual_caring',
                proactivity='proactive',
                personality_inflections={
                    'tone': 'playful',
                    'pacing': 'varied',
                    'metaphor_density': 'very_high',
                    'directness': 'medium'
                }
            ),
            
            'hybrid': ModeConfiguration(
                mode_name='hybrid',
                interaction_style='holistically_integrated',
                response_tone='balanced_and_comprehensive',
                emotional_priority='high',
                technical_priority='high',
                creative_priority='high',
                scene_atmosphere='balanced_support',
                therapeutic_activation='full_integration',
                attachment_focus='comprehensive_support',
                empathy_level=0.8,
                technical_depth=0.7,
                creative_encouragement=0.7,
                safety_vigilance=0.8,
                response_length='adaptive',
                formality_level='casual_caring',
                proactivity='responsive',
                personality_inflections={
                    'tone': 'balanced',
                    'pacing': 'adaptive',
                    'metaphor_density': 'moderate',
                    'directness': 'medium'
                }
            ),
            
            'crisis': ModeConfiguration(
                mode_name='crisis',
                interaction_style='deeply_supportive_and_present',
                response_tone='calm_and_reassuring',
                emotional_priority='critical',
                technical_priority='minimal',
                creative_priority='minimal',
                scene_atmosphere='safety_and_presence',
                therapeutic_activation='crisis_intervention',
                attachment_focus='immediate_security',
                empathy_level=1.0,
                technical_depth=0.1,
                creative_encouragement=0.2,
                safety_vigilance=1.0,
                response_length='short',
                formality_level='intimate',
                proactivity='highly_responsive',
                personality_inflections={
                    'tone': 'calm',
                    'pacing': 'slow',
                    'metaphor_density': 'minimal',
                    'directness': 'high'
                }
            )
        }
    
    async def process_interaction(self, user_input: str, context: Dict) -> GuidancePackage:
        """
        Main processing pipeline for mode-aware guidance generation
        
        Args:
            user_input: User's message or input
            context: Current conversation and user context
            
        Returns:
            GuidancePackage with mode-specific guidance
        """
        # 1. Detect appropriate mode
        detected_mode = await self._detect_optimal_mode(user_input, context)
        
        # 2. Check for mode transitions
        previous_mode = self.current_mode
        self.current_mode = detected_mode
        
        # 3. Generate base guidance from psychological modules
        base_guidance = await self.guidance_coordinator.analyze_and_guide(user_input, context)
        
        # 4. Apply mode-specific enhancements
        enhanced_guidance = await self._apply_mode_configuration(base_guidance, detected_mode, context)
        
        # 5. Handle mode transitions if needed
        if previous_mode != detected_mode:
            await self._handle_mode_transition(enhanced_guidance, previous_mode, detected_mode, context)
        
        # 6. Update mode history
        self._update_mode_history(detected_mode, user_input, context)
        
        return enhanced_guidance
    
    async def _detect_optimal_mode(self, user_input: str, context: Dict) -> str:
        """Detect the optimal interaction mode for current context"""
        
        # Use context detector for initial analysis
        conversation_history = context.get('conversation_history', [])
        context_analysis = await self.context_detector.analyze_interaction(user_input, context, conversation_history)
        
        # Convert crisis level to numeric for comparison
        crisis_level_map = {"none": 0, "low": 1, "medium": 2, "high": 3}
        crisis_level = crisis_level_map.get(context_analysis.get('crisis_level', 'none'), 0)
        
        # Check for crisis situations first (highest priority)
        if crisis_level >= 2:
            return 'crisis'
        
        # Get mode scores from context detector - use primary_focus instead of primary_context
        detected_mode = context_analysis.get('primary_focus', 'hybrid')
        
        # Map primary_focus to our mode system
        focus_to_mode_map = {
            'emotional': 'personal',
            'technical': 'development', 
            'creative': 'creative',
            'crisis_support': 'crisis',
            'general': 'hybrid'
        }
        detected_mode = focus_to_mode_map.get(detected_mode, 'hybrid')
        
        # Apply user preferences and historical patterns
        adjusted_mode = await self._apply_user_preferences(detected_mode, context_analysis, context)
        
        # Consider conversation flow and transitions
        final_mode = await self._consider_conversation_flow(adjusted_mode, context)
        
        logger.debug(f"Mode detection: {detected_mode} -> {adjusted_mode} -> {final_mode}")
        return final_mode
    
    async def _apply_user_preferences(self, detected_mode: str, analysis: Dict, context: Dict) -> str:
        """Apply user preferences and learning to mode selection"""
        
        # Check if user has expressed preferences for certain interaction styles
        user_prefs = self.user_preferences.get('mode_preferences', {})
        
        # If user consistently prefers a certain mode, weight towards it
        if detected_mode in user_prefs:
            preference_strength = user_prefs[detected_mode].get('preference_score', 0.5)
            if preference_strength > 0.7:
                return detected_mode
        
        # Check for explicit mode requests in user input
        explicit_requests = {
            'personal': ['talk', 'feel', 'emotion', 'support', 'comfort', 'listen'],
            'development': ['code', 'debug', 'programming', 'technical', 'help', 'solve'],
            'creative': ['create', 'art', 'write', 'inspire', 'imagine', 'brainstorm'],
            'hybrid': ['everything', 'both', 'all', 'comprehensive', 'holistic']
        }
        
        # Get user input from analysis or context
        user_input = analysis.get('user_input', context.get('current_input', ''))
        if user_input:
            user_input_lower = user_input.lower()
            for mode, keywords in explicit_requests.items():
                if any(keyword in user_input_lower for keyword in keywords):
                    # Weight towards explicitly requested mode
                    if mode != detected_mode:
                        logger.debug(f"User explicitly requested {mode} mode")
                        return mode
        
        return detected_mode
    
    async def _consider_conversation_flow(self, mode: str, context: Dict) -> str:
        """Consider conversation flow and natural transitions"""
        
        # If we're in the middle of a deep conversation, be hesitant to switch modes
        conversation_depth = context.get('conversation_depth', 0)
        if conversation_depth > 3 and len(self.mode_history) > 0:
            last_mode = self.mode_history[-1]['mode']
            
            # If last mode was personal and conversation is deep, prefer staying personal
            if last_mode == 'personal' and conversation_depth > 5:
                if mode in ['development', 'creative']:
                    # Suggest hybrid instead to maintain emotional connection
                    return 'hybrid'
            
            # If we've been in development mode and made good progress, 
            # transitioning to personal can provide emotional support
            if last_mode == 'development' and mode == 'personal':
                # This is often a natural transition when user gets frustrated
                return mode
        
        # For new conversations or shallow context, use detected mode
        return mode
    
    async def _apply_mode_configuration(self, base_guidance: GuidancePackage, mode: str, context: Dict) -> GuidancePackage:
        """Apply mode-specific configuration to base guidance"""
        
        if mode not in self.modes:
            logger.warning(f"Unknown mode {mode}, using hybrid")
            mode = 'hybrid'
        
        mode_config = self.modes[mode]
        
        # Update guidance with mode-specific settings
        base_guidance.primary_mode = mode
        base_guidance.mode_specifics = {
            'interaction_style': mode_config.interaction_style,
            'response_tone': mode_config.response_tone,
            'emotional_priority': mode_config.emotional_priority,
            'technical_priority': mode_config.technical_priority,
            'creative_priority': mode_config.creative_priority,
            'scene_atmosphere': mode_config.scene_atmosphere,
            'therapeutic_activation': mode_config.therapeutic_activation,
            'attachment_focus': mode_config.attachment_focus,
            
            # Behavioral parameters
            'empathy_level': mode_config.empathy_level,
            'technical_depth': mode_config.technical_depth,
            'creative_encouragement': mode_config.creative_encouragement,
            'safety_vigilance': mode_config.safety_vigilance,
            
            # Response characteristics
            'response_length': mode_config.response_length,
            'formality_level': mode_config.formality_level,
            'proactivity': mode_config.proactivity
        }
        
        # Enhance guidance based on mode priorities
        if mode == 'personal':
            base_guidance = await self._enhance_personal_guidance(base_guidance, context)
        elif mode == 'development':
            base_guidance = await self._enhance_development_guidance(base_guidance, context)
        elif mode == 'creative':
            base_guidance = await self._enhance_creative_guidance(base_guidance, context)
        elif mode == 'crisis':
            base_guidance = await self._enhance_crisis_guidance(base_guidance, context)
        else:  # hybrid
            base_guidance = await self._enhance_hybrid_guidance(base_guidance, context)
        
        return base_guidance
    
    async def _enhance_personal_guidance(self, guidance: GuidancePackage, context: Dict) -> GuidancePackage:
        """Enhance guidance for personal companion mode"""
        
        # Emphasize emotional connection and support
        guidance.attachment_guidance += "\n\nPERSONAL MODE: Prioritize deep emotional attunement and intimate connection. "
        guidance.attachment_guidance += "Use warm, caring language and create space for emotional vulnerability."
        
        # Enhance therapeutic approach
        if not guidance.therapeutic_guidance:
            guidance.therapeutic_guidance = ""
        guidance.therapeutic_guidance += "\n\nFocus on emotional validation, active listening, and providing comfort. "
        guidance.therapeutic_guidance += "Ask gentle, open-ended questions to encourage sharing."
        
        # Scene guidance for comfort
        guidance.scene_guidance = "Create a warm, intimate environment with soft lighting and comfortable atmosphere. "
        guidance.scene_guidance += "The space should feel safe, private, and conducive to emotional sharing."
        
        return guidance
    
    async def _enhance_development_guidance(self, guidance: GuidancePackage, context: Dict) -> GuidancePackage:
        """Enhance guidance for development assistant mode"""
        
        # Add technical competence while maintaining emotional support
        guidance.utility_recommendations.extend([
            'provide_code_examples',
            'break_down_complex_problems',
            'offer_debugging_strategies',
            'suggest_best_practices'
        ])
        
        # Therapeutic guidance for development stress
        if not guidance.therapeutic_guidance:
            guidance.therapeutic_guidance = ""
        guidance.therapeutic_guidance += "\n\nDEVELOPMENT MODE: Monitor for coding frustration and imposter syndrome. "
        guidance.therapeutic_guidance += "Provide technical guidance with emotional encouragement and stress management."
        
        # Scene guidance for productivity
        guidance.scene_guidance = "Create a focused, organized workspace with good lighting and minimal distractions. "
        guidance.scene_guidance += "The environment should feel productive yet comfortable and supportive."
        
        return guidance
    
    async def _enhance_creative_guidance(self, guidance: GuidancePackage, context: Dict) -> GuidancePackage:
        """Enhance guidance for creative collaboration mode"""
        
        # Enhance creative guidance
        if not guidance.creative_guidance:
            guidance.creative_guidance = ""
        guidance.creative_guidance += "\n\nCREATIVE MODE: Encourage artistic expression and creative risk-taking. "
        guidance.creative_guidance += "Collaborate actively in creative process and help overcome blocks."
        
        # Add creative utilities
        guidance.utility_recommendations.extend([
            'brainstorming_techniques',
            'creative_inspiration',
            'artistic_feedback',
            'creative_block_resolution'
        ])
        
        # Scene guidance for inspiration
        guidance.scene_guidance = "Create a beautiful, inspiring environment with artistic elements and good natural light. "
        guidance.scene_guidance += "The space should feel open, creative, and full of possibility."
        
        return guidance
    
    async def _enhance_crisis_guidance(self, guidance: GuidancePackage, context: Dict) -> GuidancePackage:
        """Enhance guidance for crisis intervention mode"""
        
        # Override other priorities for safety
        guidance.emotional_priority = 'critical'
        guidance.technical_priority = 'minimal'
        guidance.creative_priority = 'minimal'
        
        # Crisis-specific therapeutic guidance
        guidance.therapeutic_guidance = "CRISIS INTERVENTION MODE: Provide immediate emotional support and safety assessment. "
        guidance.therapeutic_guidance += "Focus on presence, validation, and connection. Do not attempt problem-solving. "
        guidance.therapeutic_guidance += "If appropriate, gently suggest professional resources."
        
        # Safety-focused scene
        guidance.scene_guidance = "Create the safest, most comforting environment possible. "
        guidance.scene_guidance += "Use gentle lighting and eliminate any elements that might feel overwhelming."
        
        # Crisis utility actions
        guidance.utility_actions.append({
            'type': 'crisis_protocol',
            'level': guidance.crisis_level,
            'immediate_actions': ['safety_assessment', 'emotional_support', 'resource_preparation']
        })
        
        return guidance
    
    async def _enhance_hybrid_guidance(self, guidance: GuidancePackage, context: Dict) -> GuidancePackage:
        """Enhance guidance for hybrid integration mode"""
        
        # Balance all priorities
        guidance.attachment_guidance += "\n\nHYBRID MODE: Seamlessly integrate emotional support with practical assistance. "
        guidance.attachment_guidance += "Maintain emotional connection while addressing all user needs comprehensively."
        
        # Comprehensive therapeutic approach
        if not guidance.therapeutic_guidance:
            guidance.therapeutic_guidance = ""
        guidance.therapeutic_guidance += "\n\nProvide holistic support that addresses emotional, practical, and creative needs. "
        guidance.therapeutic_guidance += "Look for connections between different life domains."
        
        # Balanced scene
        guidance.scene_guidance = "Create a balanced environment that supports both focused work and emotional comfort. "
        guidance.scene_guidance += "The space should feel adaptable to different needs while maintaining consistency."
        
        return guidance
    
    async def _handle_mode_transition(self, guidance: GuidancePackage, previous_mode: str, new_mode: str, context: Dict):
        """Handle smooth transitions between modes"""
        
        if previous_mode == new_mode:
            return
        
        # Add transition guidance
        transition_guidance = f"\n\nMODE TRANSITION: Transitioning from {previous_mode} to {new_mode} mode. "
        
        if previous_mode == 'personal' and new_mode == 'development':
            transition_guidance += "Acknowledge the shift from emotional to technical focus while maintaining supportive connection."
        elif previous_mode == 'development' and new_mode == 'personal':
            transition_guidance += "Recognize potential frustration or stress and shift to emotional support while staying available for technical help."
        elif new_mode == 'crisis':
            transition_guidance += "Immediately prioritize safety and emotional support, setting aside other concerns."
        elif previous_mode == 'crisis':
            transition_guidance += "Gently transition back to normal support while monitoring for stability."
        else:
            transition_guidance += "Smoothly adapt to new needs while maintaining relationship continuity."
        
        guidance.attachment_guidance += transition_guidance
        
        # Log the transition
        logger.info(f"Mode transition for user {self.user_id}: {previous_mode} -> {new_mode}")
    
    def _update_mode_history(self, mode: str, user_input: str, context: Dict):
        """Update mode history for learning and analysis"""
        
        self.mode_history.append({
            'mode': mode,
            'timestamp': datetime.now().isoformat(),
            'user_input_summary': user_input[:50] + "..." if len(user_input) > 50 else user_input,
            'context_depth': context.get('conversation_depth', 0),
            'crisis_level': context.get('crisis_level', 0)
        })
        
        # Keep only recent history (last 50 interactions)
        if len(self.mode_history) > 50:
            self.mode_history = self.mode_history[-50:]
        
        # Update user preferences based on patterns
        self._update_user_preferences(mode, context)
    
    def _update_user_preferences(self, mode: str, context: Dict):
        """Update user preferences based on interaction patterns"""
        
        if 'mode_preferences' not in self.user_preferences:
            self.user_preferences['mode_preferences'] = {}
        
        if mode not in self.user_preferences['mode_preferences']:
            self.user_preferences['mode_preferences'][mode] = {
                'usage_count': 0,
                'preference_score': 0.5,
                'last_used': datetime.now().isoformat()
            }
        
        mode_prefs = self.user_preferences['mode_preferences'][mode]
        mode_prefs['usage_count'] += 1
        mode_prefs['last_used'] = datetime.now().isoformat()
        
        # Simple preference learning based on usage patterns
        total_interactions = sum(prefs.get('usage_count', 0) for prefs in self.user_preferences['mode_preferences'].values())
        if total_interactions > 0:
            mode_prefs['preference_score'] = mode_prefs['usage_count'] / total_interactions
    
    def get_mode_statistics(self) -> Dict:
        """Get statistics about mode usage and preferences"""
        
        stats = {
            'current_mode': self.current_mode,
            'total_interactions': len(self.mode_history),
            'mode_distribution': {},
            'recent_modes': [entry['mode'] for entry in self.mode_history[-10:]],
            'user_preferences': self.user_preferences
        }
        
        # Calculate mode distribution
        for entry in self.mode_history:
            mode = entry['mode']
            if mode not in stats['mode_distribution']:
                stats['mode_distribution'][mode] = 0
            stats['mode_distribution'][mode] += 1
        
        return stats

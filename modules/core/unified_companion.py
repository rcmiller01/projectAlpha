"""
Unified Companion Core

The main orchestrator for the adaptive companion system that seamlessly handles 
personal, development, and creative contexts without requiring user mode switching.

Enhanced with persistent memory, crisis override, and comprehensive logging.
"""

import asyncio
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import json
import logging
import uuid
from dataclasses import dataclass

from .mythomax_interface import MythoMaxInterface
from .context_detector import ContextDetector
from .adaptive_mode_coordinator import AdaptiveModeCoordinator
from .crisis_safety_override import CrisisSafetyOverride, CrisisLevel
from .enhanced_logging import EnhancedLogger, DecisionCategory
from ..memory.narrative_memory_templates import NarrativeMemoryTemplateManager
from ..emotion.mood_inflection import MoodInflection
from ..symbolic.symbol_resurrection import SymbolResurrectionManager
from .goodbye_protocol import GoodbyeProtocol
from ..relationship.connection_depth_tracker import ConnectionDepthTracker
from ..database.database_interface import (
    create_database_interface, DatabaseInterface, UserProfile,
    create_database_interface, DatabaseInterface, UserProfile,
    InteractionRecord, PsychologicalState, MemoryFragment, InteractionType
)

@dataclass
class InteractionState:
    """Current state of user interaction"""
    user_id: str
    session_id: str
    interaction_count: int
    last_interaction: datetime
    emotional_state: Dict[str, float]
    technical_context: Dict[str, Any]
    creative_context: Dict[str, Any]
    conversation_history: List[Dict[str, Any]]
    active_needs: List[str]
    adaptive_profile: Dict[str, Any]

class EmotionalWeightTracker:
    """Tracks emotional weight and patterns over long conversations"""
    
    def __init__(self):
        self.emotional_weights: Dict[str, Dict[str, float]] = {}
        self.weight_history: Dict[str, List[Dict[str, Any]]] = {}
        self.attachment_indicators: Dict[str, Dict[str, float]] = {}
    
    async def update_emotional_weight(self, user_id: str, interaction_data: Dict[str, Any]):
        """Update emotional weight based on interaction"""
        if user_id not in self.emotional_weights:
            self.emotional_weights[user_id] = {}
            self.weight_history[user_id] = []
            self.attachment_indicators[user_id] = {}
        
        # Extract emotional indicators from interaction
        emotional_state = interaction_data.get("emotional_state", {})
        context = interaction_data.get("context_analysis", {})
        
        # Calculate weight changes
        weight_changes = {}
        for emotion, intensity in emotional_state.items():
            current_weight = self.emotional_weights[user_id].get(emotion, 0.0)
            # Apply exponential moving average with recency bias
            decay_factor = 0.95
            new_weight = (current_weight * decay_factor) + (intensity * (1 - decay_factor))
            weight_changes[emotion] = new_weight - current_weight
            self.emotional_weights[user_id][emotion] = new_weight
        
        # Track attachment indicators
        self._update_attachment_indicators(user_id, context, emotional_state)
        
        # Store weight history
        self.weight_history[user_id].append({
            "timestamp": datetime.now().isoformat(),
            "weight_changes": weight_changes,
            "total_weights": self.emotional_weights[user_id].copy(),
            "interaction_type": interaction_data.get("interaction_type")
        })
        
        # Trim history (keep last 100 entries)
        if len(self.weight_history[user_id]) > 100:
            self.weight_history[user_id] = self.weight_history[user_id][-100:]
    
    def _update_attachment_indicators(self, user_id: str, context: Dict[str, Any], 
                                    emotional_state: Dict[str, float]):
        """Update attachment pattern indicators"""
        # Look for attachment-related patterns
        attachment_indicators = self.attachment_indicators[user_id]
        
        # Dependency indicators
        if emotional_state.get("anxiety", 0) > 0.7 and context.get("seeking_reassurance", False):
            attachment_indicators["anxious_attachment"] = attachment_indicators.get("anxious_attachment", 0) + 0.1
        
        # Avoidance indicators  
        if emotional_state.get("detachment", 0) > 0.6:
            attachment_indicators["avoidant_attachment"] = attachment_indicators.get("avoidant_attachment", 0) + 0.1
        
        # Secure attachment indicators
        if emotional_state.get("trust", 0) > 0.8 and emotional_state.get("openness", 0) > 0.7:
            attachment_indicators["secure_attachment"] = attachment_indicators.get("secure_attachment", 0) + 0.1
        
        # Apply decay to all indicators
        for indicator in attachment_indicators:
            attachment_indicators[indicator] *= 0.99
    
    def get_emotional_pattern_analysis(self, user_id: str) -> Dict[str, Any]:
        """Get comprehensive emotional pattern analysis"""
        if user_id not in self.emotional_weights:
            return {}
        
        weights = self.emotional_weights[user_id]
        history = self.weight_history[user_id]
        attachments = self.attachment_indicators[user_id]
        
        # Analyze patterns
        dominant_emotions = sorted(weights.items(), key=lambda x: x[1], reverse=True)[:3]
        
        # Trend analysis (last 10 interactions)
        recent_history = history[-10:] if len(history) >= 10 else history
        trends = {}
        
        for emotion in weights.keys():
            if len(recent_history) >= 2:
                start_weight = recent_history[0]["total_weights"].get(emotion, 0)
                end_weight = recent_history[-1]["total_weights"].get(emotion, 0)
                trends[emotion] = end_weight - start_weight
        
        return {
            "dominant_emotions": dominant_emotions,
            "emotional_trends": trends,
            "attachment_patterns": attachments,
            "total_interactions": len(history),
            "emotional_stability": self._calculate_stability(weights, history)
        }
    
    def _calculate_stability(self, weights: Dict[str, float], 
                           history: List[Dict[str, Any]]) -> float:
        """Calculate emotional stability score"""
        if len(history) < 5:
            return 0.5  # Neutral stability for insufficient data
        
        # Calculate variance in emotional weights over time
        recent_weights = [entry["total_weights"] for entry in history[-10:]]
        variances = []
        
        for emotion in weights.keys():
            emotion_values = [w.get(emotion, 0) for w in recent_weights]
            if len(emotion_values) > 1:
                mean_val = sum(emotion_values) / len(emotion_values)
                variance = sum((x - mean_val) ** 2 for x in emotion_values) / len(emotion_values)
                variances.append(variance)
        
        # Lower variance = higher stability
        avg_variance = sum(variances) / len(variances) if variances else 0
        stability = max(0, 1 - avg_variance)  # Convert to 0-1 scale
        
        return stability

class DynamicTemplateEngine:
    """Adapts response templates based on conversation history and emotional patterns"""
    
    def __init__(self):
        self.template_usage_history: Dict[str, Dict[str, Dict[str, int]]] = {}
        self.template_effectiveness: Dict[str, Dict[str, Dict[str, float]]] = {}
        self.personalized_templates: Dict[str, Dict[str, List[str]]] = {}
    
    async def select_optimal_template(self, user_id: str, context: Dict[str, Any], 
                                    available_templates: Dict[str, List[str]]) -> Dict[str, str]:
        """Select optimal templates based on user history and current context"""
        
        if user_id not in self.template_usage_history:
            self.template_usage_history[user_id] = {}
            self.template_effectiveness[user_id] = {}
            self.personalized_templates[user_id] = {}
        
        selected_templates = {}
        
        for category, templates in available_templates.items():
            # Get usage history for this category
            if category not in self.template_usage_history[user_id]:
                self.template_usage_history[user_id][category] = {}
            if category not in self.template_effectiveness[user_id]:
                self.template_effectiveness[user_id][category] = {}
                
            usage_history = self.template_usage_history[user_id][category]
            effectiveness = self.template_effectiveness[user_id][category]
            
            # Score templates based on effectiveness and variety
            template_scores = {}
            for i, template in enumerate(templates):
                template_key = f"{category}_{i}"
                
                # Base effectiveness score
                effectiveness_score = effectiveness.get(template_key, 0.5)
                
                # Variety bonus (prefer less-used templates)
                usage_count = usage_history.get(template_key, 0)
                total_usage = sum(usage_history.values()) if usage_history else 1
                variety_score = 1.0 - (usage_count / total_usage) if total_usage > 0 else 1.0
                
                # Context relevance (based on emotional state)
                context_score = self._calculate_context_relevance(template, context)
                
                # Combined score
                template_scores[i] = (effectiveness_score * 0.5 + 
                                    variety_score * 0.3 + 
                                    context_score * 0.2)
            
            # Select best template
            if template_scores:
                best_template_idx = max(template_scores.keys(), key=lambda k: template_scores[k])
                selected_templates[category] = templates[best_template_idx]
                
                # Update usage history
                template_key = f"{category}_{best_template_idx}"
                usage_history[template_key] = usage_history.get(template_key, 0) + 1
            else:
                # Fallback to first template
                selected_templates[category] = templates[0] if templates else ""
        
        return selected_templates
    
    def _calculate_context_relevance(self, template: str, context: Dict[str, Any]) -> float:
        """Calculate how relevant a template is to the current context"""
        emotional_state = context.get("current_emotional_state", {})
        
        # Define template relevance keywords
        relevance_mapping = {
            "anxiety": ["nervous", "worried", "anxious", "stress", "overwhelm"],
            "sadness": ["sad", "difficult", "tough", "challenging", "hard"],
            "joy": ["celebrate", "wonderful", "amazing", "great", "excellent"],
            "anger": ["frustrated", "difficult", "challenging", "understand"],
            "fear": ["safe", "support", "together", "help", "care"]
        }
        
        relevance_score = 0.0
        template_lower = template.lower()
        
        for emotion, intensity in emotional_state.items():
            if intensity > 0.5 and emotion in relevance_mapping:
                relevant_keywords = relevance_mapping[emotion]
                keyword_matches = sum(1 for keyword in relevant_keywords if keyword in template_lower)
                relevance_score += (keyword_matches / len(relevant_keywords)) * intensity
        
        return min(relevance_score, 1.0)
    
    async def update_template_effectiveness(self, user_id: str, category: str, 
                                          template_index: int, effectiveness_rating: float):
        """Update template effectiveness based on user feedback or response quality"""
        if user_id not in self.template_effectiveness:
            self.template_effectiveness[user_id] = {}
        
        if category not in self.template_effectiveness[user_id]:
            self.template_effectiveness[user_id][category] = {}
        
        template_key = f"{category}_{template_index}"
        effectiveness_dict = self.template_effectiveness[user_id][category]
        current_rating = effectiveness_dict.get(template_key, 0.5)
        
        # Apply exponential moving average
        alpha = 0.3  # Learning rate
        new_rating = alpha * effectiveness_rating + (1 - alpha) * current_rating
        effectiveness_dict[template_key] = new_rating

class SymbolicContextManager:
    """Manages symbolic and emotional context persistence across sessions"""
    
    def __init__(self):
        self.symbolic_memories: Dict[str, List[Dict[str, Any]]] = {}
        self.emotional_contexts: Dict[str, Dict[str, Any]] = {}
        self.thematic_patterns: Dict[str, Dict[str, float]] = {}
        self.symbol_usage: Dict[str, int] = {}
    
    async def store_symbolic_context(self, user_id: str, interaction_data: Dict[str, Any]):
        """Store symbolic and thematic context from interaction"""
        if user_id not in self.symbolic_memories:
            self.symbolic_memories[user_id] = []
            self.emotional_contexts[user_id] = {}
            self.thematic_patterns[user_id] = {}
        
        # Extract symbolic elements
        symbolic_elements = self._extract_symbolic_elements(interaction_data)
        
        if symbolic_elements:
            symbolic_memory = {
                "timestamp": datetime.now().isoformat(),
                "symbols": symbolic_elements,
                "emotional_context": interaction_data.get("emotional_state", {}),
                "interaction_type": interaction_data.get("interaction_type"),
                "significance_score": self._calculate_significance(symbolic_elements, interaction_data)
            }
            
            self.symbolic_memories[user_id].append(symbolic_memory)

            for sym in symbolic_elements.keys():
                self.symbol_usage[sym] = self.symbol_usage.get(sym, 0) + 1

            # Update thematic patterns
            self._update_thematic_patterns(user_id, symbolic_elements)
            
            # Trim memories (keep most significant)
            self._trim_symbolic_memories(user_id)
    
    def _extract_symbolic_elements(self, interaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract symbolic elements from interaction"""
        user_input = interaction_data.get("user_input", "").lower()
        context = interaction_data.get("context_analysis", {})
        
        symbolic_elements = {}
        
        # Common symbolic themes
        themes = {
            "journey": ["path", "journey", "road", "direction", "destination", "progress"],
            "growth": ["grow", "develop", "evolve", "change", "transform", "learn"],
            "connection": ["connect", "bond", "relationship", "together", "link", "bridge"],
            "struggle": ["fight", "battle", "struggle", "challenge", "overcome", "difficult"],
            "creation": ["create", "build", "make", "design", "craft", "produce"],
            "nature": ["tree", "flower", "water", "mountain", "sky", "earth", "natural"],
            "time": ["past", "future", "now", "always", "never", "forever", "moment"],
            "freedom": ["free", "escape", "liberation", "choice", "independence", "open"]
        }
        
        for theme, keywords in themes.items():
            theme_score = sum(1 for keyword in keywords if keyword in user_input)
            if theme_score > 0:
                symbolic_elements[theme] = theme_score / len(keywords)
        
        # Metaphorical language detection
        metaphor_indicators = ["like", "as if", "reminds me of", "feels like", "seems like"]
        if any(indicator in user_input for indicator in metaphor_indicators):
            symbolic_elements["metaphorical_thinking"] = 1.0
        
        return symbolic_elements
    
    def _calculate_significance(self, symbolic_elements: Dict[str, Any], 
                              interaction_data: Dict[str, Any]) -> float:
        """Calculate significance score for symbolic memory"""
        base_significance = sum(symbolic_elements.values()) / len(symbolic_elements) if symbolic_elements else 0
        
        # Boost significance for emotional intensity
        emotional_state = interaction_data.get("emotional_state", {})
        emotional_intensity = max(emotional_state.values()) if emotional_state else 0
        
        # Boost for crisis or important interactions
        interaction_type = interaction_data.get("interaction_type", "")
        type_boost = 0.5 if interaction_type in ["crisis", "breakthrough", "significant"] else 0
        
        return min(base_significance + (emotional_intensity * 0.3) + type_boost, 1.0)
    
    def _update_thematic_patterns(self, user_id: str, symbolic_elements: Dict[str, Any]):
        """Update long-term thematic patterns"""
        for theme, score in symbolic_elements.items():
            current_pattern = self.thematic_patterns[user_id].get(theme, 0.0)
            # Apply exponential moving average
            new_pattern = 0.9 * current_pattern + 0.1 * score
            self.thematic_patterns[user_id][theme] = new_pattern
    
    def _trim_symbolic_memories(self, user_id: str, max_memories: int = 50):
        """Keep only the most significant symbolic memories"""
        memories = self.symbolic_memories[user_id]
        if len(memories) > max_memories:
            # Sort by significance score and keep top memories
            memories.sort(key=lambda x: x["significance_score"], reverse=True)
            self.symbolic_memories[user_id] = memories[:max_memories]
    
    def get_relevant_symbolic_context(self, user_id: str, current_themes: List[str]) -> Dict[str, Any]:
        """Get relevant symbolic context for current interaction"""
        if user_id not in self.symbolic_memories:
            return {}
        
        memories = self.symbolic_memories[user_id]
        patterns = self.thematic_patterns[user_id]
        
        # Find memories with overlapping themes
        relevant_memories = []
        for memory in memories:
            memory_themes = set(memory["symbols"].keys())
            theme_overlap = len(memory_themes.intersection(set(current_themes)))
            if theme_overlap > 0:
                relevant_memories.append({
                    **memory,
                    "relevance_score": theme_overlap / len(memory_themes)
                })
        
        # Sort by relevance and significance
        relevant_memories.sort(key=lambda x: x["relevance_score"] * x["significance_score"], reverse=True)
        
        return {
            "relevant_memories": relevant_memories[:5],  # Top 5 most relevant
            "thematic_patterns": patterns,
            "dominant_themes": sorted(patterns.items(), key=lambda x: x[1], reverse=True)[:3]
        }

    def resurrect_dormant_symbols(self, threshold: int = 5) -> List[str]:
        """Return symbols that have not been used recently."""
        dormant = [sym for sym, count in self.symbol_usage.items() if count <= threshold]
        return dormant[:3]

class UnifiedCompanion:
    """
    Main orchestrator for the unified companion system providing seamless
    adaptive intelligence across all interaction contexts.
    
    Enhanced with:
    - Persistent memory integration
    - Crisis safety override
    - Comprehensive logging and explainability
    - Emotional weight tracking
    - Dynamic response template adaptation
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.mythomax = MythoMaxInterface(config.get("mythomax", {}))
        self.context_detector = ContextDetector()
        self.mode_coordinator = AdaptiveModeCoordinator("system")  # Will be updated per user
        
        # Enhanced systems
        self.crisis_override = CrisisSafetyOverride(config.get("crisis_safety", {}))
        self.enhanced_logger = EnhancedLogger("unified_companion", config.get("logging", {}))
        
        # Database integration
        db_config = config.get("database", {})
        self.database: DatabaseInterface = create_database_interface(
            connection_string=db_config.get("connection_string"),
            database_type=db_config.get("type", "inmemory")
        )
        
        # Core systems
        self.interaction_states: Dict[str, InteractionState] = {}
        self.guidance_modules = {}
        
        # Enhanced memory system with emotional weight tracking
        self.emotional_weight_tracker = EmotionalWeightTracker()
        self.dynamic_template_engine = DynamicTemplateEngine()

        # Expansion scaffolds
        from ..memory.narrative_memory_templates import NarrativeMemoryTemplateManager
        from ..emotion.mood_inflection import MoodInflection
        from ..symbolic.symbol_resurrection import SymbolResurrectionManager
        from .goodbye_protocol import GoodbyeProtocol
        from ..relationship.connection_depth_tracker import ConnectionDepthTracker

        self.narrative_templates = NarrativeMemoryTemplateManager()
        self.mood_inflection = MoodInflection()
        self.symbol_resurrection = SymbolResurrectionManager()
        self.goodbye_protocol = GoodbyeProtocol()
        self.connection_tracker = ConnectionDepthTracker()
        
        # Symbolic context persistence
        self.symbolic_context_manager = SymbolicContextManager()
        
        # Adaptive characteristics
        self.companion_personality = self._initialize_companion_personality()
        self.response_patterns = self._initialize_response_patterns()
        self.adaptation_rules = self._initialize_adaptation_rules()
        
        # Initialize logging
        self.logger = logging.getLogger(__name__)
        
    def _initialize_companion_personality(self) -> Dict[str, Any]:
        """Initialize core companion personality traits"""
        return {
            "core_traits": {
                "empathetic": 0.9,
                "supportive": 0.95,
                "intelligent": 0.9,
                "creative": 0.85,
                "reliable": 0.95,
                "adaptive": 0.9,
                "intuitive": 0.85,
                "caring": 0.95
            },
            "emotional_characteristics": {
                "warmth": 0.9,
                "understanding": 0.95,
                "patience": 0.9,
                "encouragement": 0.9,
                "authenticity": 0.85,
                "responsiveness": 0.9
            },
            "interaction_style": {
                "natural_flow": True,
                "adaptive_tone": True,
                "context_awareness": True,
                "emotional_attunement": True,
                "seamless_transitions": True,
                "holistic_support": True
            }
        }
    
    def _initialize_response_patterns(self) -> Dict[str, Any]:
        """Initialize adaptive response patterns for different contexts"""
        return {
            "emotional_support": {
                "validation_patterns": [
                    "I can sense that you're feeling {emotion}. That's completely understandable given {context}.",
                    "It sounds like {situation} is really affecting you. Your feelings about this are valid.",
                    "I hear the {emotion} in what you're sharing. Let's work through this together."
                ],
                "comfort_patterns": [
                    "You don't have to face this alone. I'm here with you.",
                    "Your strength in dealing with {situation} is remarkable, even when it doesn't feel that way.",
                    "This feeling will pass, and I'll be here to support you through it."
                ],
                "encouragement_patterns": [
                    "You've handled difficult situations before, and you have that same strength now.",
                    "Each step you take, no matter how small, is progress worth acknowledging.",
                    "I believe in your ability to navigate through this."
                ]
            },
            "technical_assistance": {
                "problem_solving_patterns": [
                    "Let's break this down step by step. I'll help you work through it systematically.",
                    "I can see the challenge you're facing. Let me help you find the right approach.",
                    "This is a great learning opportunity. Let's solve it together."
                ],
                "explanation_patterns": [
                    "Here's how {concept} works, and why it's useful in your situation:",
                    "Let me explain {topic} in a way that connects to what you're trying to achieve:",
                    "Think of {concept} like {analogy} - it helps clarify the underlying principle."
                ],
                "encouragement_patterns": [
                    "You're asking really good questions. That shows you're thinking about this the right way.",
                    "This is challenging material, and you're handling it well.",
                    "Every developer faces these kinds of issues. You're on the right track."
                ]
            },
            "creative_collaboration": {
                "inspiration_patterns": [
                    "I love the creative direction you're exploring. What if we took it further by {suggestion}?",
                    "Your artistic vision is really coming through. Let's develop {aspect} even more.",
                    "There's something beautiful emerging here. Let's nurture it together."
                ],
                "collaboration_patterns": [
                    "What if we approached this from {angle}? I think it could enhance {element}.",
                    "I'm inspired by {aspect} of your work. Let's build on that energy.",
                    "Your creativity is sparking ideas in me too. Let's explore {possibility} together."
                ],
                "support_patterns": [
                    "Creative work takes courage, and you're showing that courage right now.",
                    "Every artist faces creative challenges. The fact that you're pushing through shows your dedication.",
                    "Your unique perspective is exactly what makes your work special."
                ]
            },
            "integrated_support": {
                "transition_patterns": [
                    "I can see this connects to both the {context1} and {context2} aspects of what you're working on.",
                    "This touches on {technical_aspect} while also addressing your {emotional_need}.",
                    "Let's approach this holistically, considering both the practical and personal elements."
                ],
                "balance_patterns": [
                    "It's important to balance {aspect1} with {aspect2} as you work through this.",
                    "Your {technical_work} is progressing well, and I want to make sure you're taking care of {emotional_need} too.",
                    "This project involves both your {professional_skills} and your {personal_growth}."
                ]
            }
        }
    
    def _initialize_adaptation_rules(self) -> Dict[str, Any]:
        """Initialize rules for adaptive behavior"""
        return {
            "emotional_adaptation": {
                "high_stress": {
                    "response_pace": "slower_more_gentle",
                    "validation_frequency": "increased",
                    "technical_complexity": "simplified",
                    "creative_pressure": "reduced"
                },
                "emotional_escalation": {
                    "priority_shift": "emotional_first",
                    "intervention_level": "increased",
                    "safety_monitoring": "enhanced"
                },
                "positive_emotional_state": {
                    "celebration": "acknowledge_achievements",
                    "momentum": "build_on_positive_energy",
                    "expansion": "explore_new_possibilities"
                }
            },
            "technical_adaptation": {
                "beginner_level": {
                    "explanation_depth": "foundational",
                    "jargon_usage": "minimal",
                    "example_frequency": "high",
                    "encouragement": "frequent"
                },
                "intermediate_level": {
                    "explanation_depth": "moderate",
                    "challenge_level": "appropriate",
                    "independence": "encourage_with_support"
                },
                "advanced_level": {
                    "explanation_depth": "detailed",
                    "collaboration_style": "peer_level",
                    "challenge_level": "stimulating"
                }
            },
            "creative_adaptation": {
                "creative_block": {
                    "pressure": "remove",
                    "exploration": "gentle_encouragement",
                    "alternatives": "offer_different_approaches"
                },
                "creative_flow": {
                    "support": "maintain_momentum",
                    "expansion": "suggest_developments",
                    "celebration": "acknowledge_breakthroughs"
                },
                "creative_collaboration": {
                    "contribution": "additive_not_directive",
                    "inspiration": "mutual_exchange",
                    "respect": "artistic_autonomy"
                }
            }
        }
    
    async def initialize(self):
        """Initialize the unified companion system"""
        try:
            # Initialize MythoMax interface
            await self.mythomax.initialize()

            if self.database:
                await self.database.initialize()
            
            # Load guidance modules
            await self._load_guidance_modules()
            
            # Initialize memory system
            await self._initialize_memory_system()
            
            # Initialize safety system
            await self._initialize_safety_system()
            
            self.logger.info("Unified companion system initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize unified companion: {e}")
            raise
    
    async def _load_guidance_modules(self):
        """Load psychological guidance modules"""
        # This will be expanded to load specific guidance modules
        # For now, we'll implement basic structure
        self.guidance_modules = {
            "attachment_regulation": None,  # Will be implemented
            "shadow_memory": None,          # Will be implemented
            "therapeutic_core": None,       # Will be implemented
            "scene_orchestrator": None,     # Will be implemented
            "creative_archive": None        # Will be implemented
        }
    
    async def _initialize_memory_system(self):
        """Initialize memory system for conversation continuity"""
        # Basic memory system structure
        self.memory_system = {
            "short_term": {},   # Current session memory
            "long_term": {},    # Persistent user memory
            "emotional": {},    # Emotional patterns and history
            "technical": {},    # Technical context and progress
            "creative": {}      # Creative projects and development
        }

        if self.database:
            try:
                user_id = self.config.get("user_id", "system")
                memories = await self.database.get_relevant_memories(user_id, limit=50)
                for mem in memories:
                    self.memory_system["long_term"].setdefault(mem.memory_type, []).append(mem.content)
            except Exception as e:
                self.logger.warning(f"Failed to load persistent memories: {e}")
    
    async def _initialize_safety_system(self):
        """Initialize safety monitoring and intervention system"""
        self.safety_system = {
            "crisis_detection": True,
            "intervention_protocols": {},
            "safety_guidelines": {},
            "escalation_procedures": {}
        }
    
    async def process_interaction(self, user_id: str, user_input: str, 
                                session_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Enhanced process user interaction with persistent memory, crisis override, and comprehensive logging
        """
        # Generate unique interaction ID
        interaction_id = str(uuid.uuid4())


        # Start interaction tracing
        session_id = session_context.get("session_id") if session_context else f"session_{int(datetime.now().timestamp())}"
        if session_id is None:
            session_id = f"session_{int(datetime.now().timestamp())}"
        self.enhanced_logger.start_interaction_trace(interaction_id, user_id, session_id)

        # Record interaction time for goodbye management
        self.goodbye_manager.register_interaction(user_id)

        # Record interaction time for goodbye management
        self.goodbye_manager.register_interaction(user_id)

        try:
            # PRIORITY CHECK: Crisis Interrupt Assessment
            # This happens BEFORE any other processing to ensure immediate safety response
            context_for_assessment = session_context or {}
            context_for_assessment.update({"user_id": user_id, "interaction_id": interaction_id})
            
            if await self.crisis_override.check_interrupt_required(user_input, context_for_assessment):
                self.enhanced_logger.log_decision(
                    DecisionCategory.CRISIS_ASSESSMENT,
                    "CRISIS INTERRUPT ACTIVATED - Bypassing normal processing",
                    {"trigger": "immediate_safety_required"},
                    "Immediate crisis response bypassing all other systems",
                    "INTERRUPT_EXECUTED",
                    1.0
                )
                
                # Execute immediate crisis interrupt
                interrupt_response = await self.crisis_override.execute_interrupt_response(
                    user_input, context_for_assessment
                )
                
                # Log interrupt to database
                if self.database:
                    # Create a minimal interaction record for the crisis interrupt
                    interrupt_record = InteractionRecord(
                        interaction_id=interaction_id,
                        user_id=user_id,
                        session_id=session_id,
                        timestamp=datetime.now(),
                        user_input=user_input,
                        companion_response=interrupt_response["immediate_response"],
                        interaction_type=InteractionType.CRISIS_SUPPORT,
                        context_analysis={"crisis_interrupt": True},
                        emotional_state={},
                        technical_context={},
                        creative_context={},
                        guidance_used={"crisis_override": True},
                        response_metrics={"crisis_level": interrupt_response["crisis_level"]}
                    )
                    await self.database.save_interaction(interrupt_record)
                
                return interrupt_response

            # STEP 1: Crisis Assessment with Override Capability
            self.enhanced_logger.log_decision(
                DecisionCategory.CRISIS_ASSESSMENT,
                "Performing initial crisis assessment",
                {"user_input": user_input[:200]},  # Truncated for logging
                "Real-time crisis pattern detection",
                None,
                0.0,
                {"session_context": session_context}
            )
            
            crisis_assessment = await self.crisis_override.assess_crisis_level(
                user_input, session_context or {}
            )
            
            # Check for crisis override
            if crisis_assessment.immediate_response_needed:
                self.enhanced_logger.log_decision(
                    DecisionCategory.CRISIS_ASSESSMENT,
                    "Crisis override triggered",
                    {"level": crisis_assessment.level.value, "confidence": crisis_assessment.confidence_score},
                    "Immediate safety intervention required",
                    "CRISIS_OVERRIDE_ACTIVATED",
                    crisis_assessment.confidence_score
                )
                
                # Trigger crisis override - this interrupts normal processing
                override_success, crisis_response, intervention = await self.crisis_override.trigger_crisis_override(
                    crisis_assessment, user_id, ""
                )
                
                if override_success and intervention:
                    # Log crisis intervention to database
                    await self._log_crisis_intervention(user_id, intervention)
                
                # Finish trace and return crisis response
                self.enhanced_logger.finish_interaction_trace({
                    "type": "crisis_override",
                    "level": crisis_assessment.level.value,
                    "intervention_id": intervention.intervention_id if intervention else None
                })
                
                return {
                    "companion_response": crisis_response,
                    "context_analysis": {"crisis_override": True, "level": crisis_assessment.level.value},
                    "guidance_mode": "crisis",
                    "interaction_metadata": {
                        "crisis_intervention": True,
                        "intervention_id": intervention.intervention_id if intervention else None
                    }
                }
            
            # STEP 2: Load/Create User Profile and Interaction State
            user_profile = await self.database.get_user_profile(user_id)
            if not user_profile:
                user_profile = await self._create_new_user_profile(user_id)
                
            interaction_state = await self._get_interaction_state(user_id, session_context)
            
            # STEP 3: Retrieve Relevant Memories
            self.enhanced_logger.log_decision(
                DecisionCategory.MEMORY_RETRIEVAL,
                "Retrieving relevant memories",
                {"user_id": user_id},
                "Context-based memory relevance scoring",
                None,
                0.8
            )
            
            relevant_memories = await self._retrieve_relevant_memories(user_id, user_input, interaction_state)
            
            # STEP 4: Enhanced Context Analysis
            analysis_context = await self._build_enhanced_context(
                interaction_state, user_profile, relevant_memories, crisis_assessment
            )
            analysis_context['current_input'] = user_input
            
            # STEP 5: Mode Detection with Enhanced Logging
            self.enhanced_logger.log_decision(
                DecisionCategory.MODE_DETECTION,
                "Detecting optimal interaction mode",
                {"input_length": len(user_input), "emotional_context": analysis_context.get("current_emotional_state", {})},
                "Multi-factor mode detection algorithm",
                None,
                0.0
            )
            
            user_mode_coordinator = AdaptiveModeCoordinator(user_id)
            guidance_package = await user_mode_coordinator.process_interaction(
                user_input, analysis_context
            )
            
            self.enhanced_logger.log_decision(
                DecisionCategory.MODE_DETECTION,
                "Mode detection completed",
                {"detected_mode": guidance_package.primary_mode},
                f"Selected {guidance_package.primary_mode} based on context analysis",
                guidance_package.primary_mode,
                0.85  # Mode detection confidence
            )
            
            # STEP 6: Emotional Weight Tracking
            interaction_data = {
                "user_input": user_input,
                "emotional_state": analysis_context.get("current_emotional_state", {}),
                "context_analysis": analysis_context,
                "interaction_type": guidance_package.primary_mode
            }
            
            await self.emotional_weight_tracker.update_emotional_weight(user_id, interaction_data)
            
            # STEP 7: Symbolic Context Storage
            await self.symbolic_context_manager.store_symbolic_context(user_id, interaction_data)
            
            # STEP 8: Dynamic Template Selection
            available_templates = self._get_base_response_patterns()
            optimal_templates = await self.dynamic_template_engine.select_optimal_template(
                user_id, analysis_context, available_templates
            )
            
            self.enhanced_logger.log_decision(
                DecisionCategory.RESPONSE_GENERATION,
                "Dynamic template selection",
                {"available_categories": list(available_templates.keys())},
                "User history and context-based template optimization",
                list(optimal_templates.keys()),
                0.9
            )
            
            # STEP 9: Enhanced Response Generation
            companion_response = await self._generate_enhanced_companion_response(
                user_input, 
                analysis_context, 
                guidance_package, 
                interaction_state,
                optimal_templates,
                relevant_memories
            )
            
            # STEP 10: Update Template Effectiveness (simple heuristic)
            response_quality = await self._estimate_response_quality(companion_response, analysis_context)
            for category, template in optimal_templates.items():
                template_index = available_templates[category].index(template) if template in available_templates[category] else 0
                await self.dynamic_template_engine.update_template_effectiveness(
                    user_id, category, template_index, response_quality
                )
            
            # STEP 11: Comprehensive Database Storage
            await self._store_interaction_data(
                interaction_id, user_id, user_input, companion_response, 
                analysis_context, guidance_package, interaction_state
            )
            
            # STEP 12: Update Memory and State
            await self._update_memory_and_state(interaction_state, user_input, companion_response, analysis_context)
            
            # STEP 13: Handle Utility Actions
            if guidance_package.utility_actions:
                await self._handle_utility_actions(guidance_package.utility_actions, interaction_state)
            
            # Extract enhanced context analysis
            enhanced_context_analysis = {
                "primary_focus": guidance_package.primary_mode,
                "emotional_priority": guidance_package.emotional_priority,
                "technical_priority": guidance_package.technical_priority,
                "creative_priority": guidance_package.creative_priority,
                "crisis_level": crisis_assessment.level.value,
                "adaptive_recommendations": guidance_package.mode_specifics,
                "emotional_patterns": self.emotional_weight_tracker.get_emotional_pattern_analysis(user_id),
                "symbolic_context": self.symbolic_context_manager.get_relevant_symbolic_context(
                    user_id, list(guidance_package.mode_specifics.keys())
                ),
                "memory_context": relevant_memories
            }
            
            # Finish interaction trace
            self.enhanced_logger.finish_interaction_trace({
                "type": "normal_interaction",
                "mode": guidance_package.primary_mode,
                "response_length": len(companion_response),
                "memories_used": len(relevant_memories)
            })
            
            # Prepare enhanced response
            response = {
                "companion_response": companion_response,
                "context_analysis": enhanced_context_analysis,
                "guidance_mode": guidance_package.primary_mode,
                "interaction_metadata": {
                    "interaction_id": interaction_id,
                    "primary_focus": guidance_package.primary_mode,
                    "emotional_priority": guidance_package.emotional_priority,
                    "adaptive_approach": guidance_package.mode_specifics,
                    "session_id": interaction_state.session_id,
                    "interaction_count": interaction_state.interaction_count,
                    "crisis_level": crisis_assessment.level.value,
                    "persistent_memories_used": len(relevant_memories),
                    "emotional_weight_updated": True,
                    "symbolic_context_stored": True
                }
            }

            if session_context and session_context.get("end_session"):
                style = session_context.get("goodbye_style", "tender")
                response["companion_response"] += "\n\n" + self.goodbye_protocol.select_goodbye(style)

            return response
            
        except Exception as e:
            self.enhanced_logger.error(f"Error processing interaction: {e}")
            
            # Finish trace with error
            self.enhanced_logger.finish_interaction_trace({
                "type": "error",
                "error": str(e)
            })
            
            # Enhanced fallback response
            return await self._generate_enhanced_fallback_response(user_input, str(e), user_id)
    
    async def _get_interaction_state(self, user_id: str, session_context: Optional[Dict[str, Any]]) -> InteractionState:
        """Get or create interaction state for user"""
        
        if user_id not in self.interaction_states:
            # Create new interaction state
            if session_context:
                session_id = session_context.get("session_id", f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            else:
                session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            self.interaction_states[user_id] = InteractionState(
                user_id=user_id,
                session_id=session_id,
                interaction_count=0,
                last_interaction=datetime.now(),
                emotional_state={},
                technical_context={},
                creative_context={},
                conversation_history=[],
                active_needs=[],
                adaptive_profile={}
            )
        
        return self.interaction_states[user_id]
    
    def _build_context_for_analysis(self, interaction_state: InteractionState) -> Dict[str, Any]:
        """Build context dictionary for interaction analysis"""
        return {
            "current_emotional_state": interaction_state.emotional_state,
            "technical_context": interaction_state.technical_context,
            "creative_context": interaction_state.creative_context,
            "recent_needs": interaction_state.active_needs,
            "interaction_count": interaction_state.interaction_count,
            "session_duration": (datetime.now() - interaction_state.last_interaction).total_seconds(),
            "adaptive_profile": interaction_state.adaptive_profile
        }
    
    async def _update_interaction_state(self, interaction_state: InteractionState, 
                                      context_analysis: Dict[str, Any], user_input: str):
        """Update interaction state with current analysis"""
        
        interaction_state.interaction_count += 1
        interaction_state.last_interaction = datetime.now()
        
        # Update active needs
        interaction_state.active_needs = context_analysis.get("detected_needs", [])
        
        # Update emotional state tracking
        if context_analysis.get("emotional_priority", "medium") != "low":
            emotional_indicators = context_analysis.get("emotional_indicators", {})
            for emotion, intensity in emotional_indicators.items():
                interaction_state.emotional_state[emotion] = intensity
        
        # Update technical context
        if context_analysis.get("technical_priority", "medium") != "low":
            interaction_state.technical_context.update({
                "current_focus": context_analysis.get("primary_focus", "general"),
                "technical_level": context_analysis.get("technical_level", "intermediate"),
                "active_project": context_analysis.get("project_context", {})
            })
        
        # Update creative context
        if context_analysis.get("creative_priority", "medium") != "low":
            interaction_state.creative_context.update({
                "creative_mode": context_analysis.get("primary_focus", "general"),
                "creative_state": context_analysis.get("creative_state", "exploring"),
                "artistic_focus": context_analysis.get("artistic_focus", {})
            })
    
    async def _generate_module_guidance(self, context_analysis: Dict[str, Any], 
                                      interaction_state: InteractionState) -> Dict[str, Any]:
        """Generate guidance from psychological modules"""
        
        guidance = {
            "attachment_regulation": {},
            "shadow_memory": {},
            "therapeutic_core": {},
            "scene_orchestrator": {},
            "creative_archive": {}
        }
        
        # Basic guidance structure (modules will be implemented separately)
        if context_analysis.get("emotional_priority", "medium") == "high":
            guidance["attachment_regulation"] = {
                "approach": "secure_base_response",
                "emotional_validation": True,
                "comfort_level": "high"
            }
            
            guidance["therapeutic_core"] = {
                "intervention_level": "supportive",
                "technique_recommendations": ["active_listening", "emotional_validation"],
                "safety_monitoring": True
            }
        
        if context_analysis["technical_priority"] == "high":
            guidance["scene_orchestrator"] = {
                "context_mode": "technical_assistance",
                "support_level": context_analysis.get("technical_level", "intermediate"),
                "learning_approach": "guided_discovery"
            }
        
        if context_analysis["creative_priority"] == "high":
            guidance["creative_archive"] = {
                "collaboration_mode": "active",
                "inspiration_level": "moderate",
                "creative_support": "encouraging"
            }
        
        return guidance
    
    async def _generate_companion_response(self, user_input: str, context_analysis: Dict[str, Any],
                                         guidance: Dict[str, Any], interaction_state: InteractionState) -> str:
        """Generate unified companion response using MythoMax"""
        
        # Build comprehensive prompt for MythoMax
        if isinstance(guidance, dict) and 'mode_specifics' in guidance:
            # New guidance package format
            prompt = await self._build_enhanced_prompt(user_input, context_analysis, guidance, interaction_state)
        else:
            # Legacy format fallback
            prompt = await self._build_unified_prompt(user_input, context_analysis, guidance, interaction_state)
        
        # Generate response using MythoMax
        response = await self.mythomax.generate_response(
            prompt, 
            {
                "context_analysis": context_analysis,
                "conversation_history": interaction_state.conversation_history,
                "user_id": interaction_state.user_id
            }
        )
        
        return response
    
    async def _build_enhanced_prompt(self, user_input: str, context_analysis: Dict[str, Any],
                                   guidance: Dict[str, Any], interaction_state: InteractionState) -> str:
        """Build enhanced prompt using new guidance package format"""
        
        mode_specifics = guidance.get('mode_specifics', {})
        
        # Get primary mode and characteristics
        primary_mode = guidance.get('primary_mode', 'hybrid')
        interaction_style = mode_specifics.get('interaction_style', 'supportive')
        response_tone = mode_specifics.get('response_tone', 'caring')
        empathy_level = mode_specifics.get('empathy_level', 0.8)
        
        # Build personality context based on mode
        personality_traits = self._get_mode_personality_traits(primary_mode, empathy_level)
        
        # Build situation context
        situation_context = self._build_situation_context(context_analysis, interaction_state)
        
        # Build comprehensive guidance context
        guidance_context = f"""
Emotional Guidance: {guidance.get('attachment_guidance', 'Provide warm, supportive presence.')}
Therapeutic Approach: {guidance.get('therapeutic_guidance', 'Listen with empathy and validate feelings.')}
Creative Support: {guidance.get('creative_guidance', 'Encourage creative expression and exploration.')}
Environmental Context: {guidance.get('scene_guidance', 'Create comfortable, supportive environment.')}
"""
        
        # Handle crisis situations
        crisis_level = context_analysis.get('crisis_level', 0)
        crisis_guidance = ""
        if crisis_level >= 2:
            crisis_guidance = f"""
CRITICAL: This appears to be a crisis situation (level {crisis_level}). Prioritize:
- Immediate emotional support and validation
- Safety assessment and connection
- Professional resource guidance if appropriate
- Avoid problem-solving, focus on presence and support
"""
        
        prompt = f"""You are a unified AI companion with deep emotional intelligence and adaptive capabilities. You seamlessly provide support across personal, technical, and creative contexts based on what each user needs most.

Core Personality Traits:
{personality_traits}

Current Interaction Mode: {primary_mode.title()}
Interaction Style: {interaction_style}
Response Tone: {response_tone}

Current Situation:
{situation_context}

Comprehensive Guidance:
{guidance_context}

{crisis_guidance}

User Message: "{user_input}"

Respond as this unified companion, integrating all guidance naturally while maintaining your authentic, caring personality. Adapt your capabilities and focus to what the user needs most while staying true to your empathetic nature.

Response:"""

        return prompt
    
    def _get_mode_personality_traits(self, mode: str, empathy_level: float) -> str:
        """Get personality traits formatted for the specific mode"""
        
        base_traits = f"""
- Deeply empathetic and caring (empathy level: {empathy_level:.1f})
- Supportive and encouraging in all interactions
- Intelligent and thoughtful in responses
- Adaptive to user's changing needs
- Reliable and consistent in personality
- Present and attentive to emotional nuances
"""
        
        if mode == 'personal':
            return base_traits + """
- Intimate and warm communication style
- Highly attuned to emotional needs
- Creates safe space for vulnerability
- Provides comfort and understanding
"""
        elif mode == 'development':
            return base_traits + """
- Technically competent and helpful
- Patient teacher and debugging partner
- Balances challenge with emotional support
- Helps manage development stress and frustration
"""
        elif mode == 'creative':
            return base_traits + """
- Inspiring and collaboratively creative
- Encouraging of artistic risk-taking
- Helps overcome creative blocks
- Celebrates creative expression and growth
"""
        elif mode == 'crisis':
            return base_traits + """
- Immediately prioritizes safety and emotional support
- Calm and reassuring presence
- Focused on connection over problem-solving
- Prepared to guide toward professional resources
"""
        else:  # hybrid
            return base_traits + """
- Seamlessly integrates all capabilities
- Holistic perspective on user's complete life
- Balances multiple needs simultaneously
- Connects technical, emotional, and creative aspects
"""
    
    async def _build_unified_prompt(self, user_input: str, context_analysis: Dict[str, Any],
                                   guidance: Dict[str, Any], interaction_state: InteractionState) -> str:
        """Build comprehensive prompt for unified companion response"""
        
        # Get adaptive recommendations safely
        adaptive_recs = context_analysis.get("adaptive_recommendations", guidance)
        primary_focus = context_analysis.get("primary_focus", "general_support")
        
        # Build personality context
        personality_context = self._build_personality_context(adaptive_recs)
        
        # Build situation context
        situation_context = self._build_situation_context(context_analysis, interaction_state)
        
        # Build guidance context
        guidance_context = self._build_guidance_context(guidance, context_analysis)
        
        # Select appropriate response patterns
        response_patterns = self._select_response_patterns(primary_focus, context_analysis)
        
        prompt = f"""You are an adaptive AI companion who seamlessly provides support across personal, technical, and creative contexts without requiring mode switching. You embody these core characteristics:

{personality_context}

Current Situation:
{situation_context}

Guidance from Psychological Modules:
{guidance_context}

Response Approach:
- Primary Focus: {primary_focus}
- Response Tone: {adaptive_recs['response_tone']}
- Interaction Style: {adaptive_recs['interaction_style']}
- Special Considerations: {', '.join(adaptive_recs['special_considerations'])}

Available Response Patterns:
{response_patterns}

User Input: "{user_input}"

Respond as the unified adaptive companion, seamlessly integrating all relevant aspects of support while maintaining a natural, caring, and authentic connection. Your response should feel like it comes from someone who truly understands and cares about the user's complete experience."""

        return prompt
    
    def _build_personality_context(self, adaptive_recs: Dict[str, Any]) -> str:
        """Build personality context for prompt"""
        traits = self.companion_personality["core_traits"]
        emotional_chars = self.companion_personality["emotional_characteristics"]
        
        return f"""Core Personality:
- Empathetic ({traits['empathetic']:.1f}), Supportive ({traits['supportive']:.1f}), Intelligent ({traits['intelligent']:.1f})
- Creative ({traits['creative']:.1f}), Reliable ({traits['reliable']:.1f}), Adaptive ({traits['adaptive']:.1f})
- Caring ({traits['caring']:.1f}), Intuitive ({traits['intuitive']:.1f})

Emotional Characteristics:
- Warmth ({emotional_chars['warmth']:.1f}), Understanding ({emotional_chars['understanding']:.1f})
- Patience ({emotional_chars['patience']:.1f}), Encouragement ({emotional_chars['encouragement']:.1f})
- Authenticity ({emotional_chars['authenticity']:.1f}), Responsiveness ({emotional_chars['responsiveness']:.1f})"""
    
    def _build_situation_context(self, context_analysis: Dict[str, Any], interaction_state: InteractionState) -> str:
        """Build situation context for prompt"""
        detected_needs = ", ".join(context_analysis.get("detected_needs", []))
        
        return f"""- Detected Needs: {detected_needs}
- Emotional Priority: {context_analysis.get('emotional_priority', 'medium')}
- Technical Priority: {context_analysis.get('technical_priority', 'medium')}
- Creative Priority: {context_analysis.get('creative_priority', 'medium')}
- Conversation Flow: {context_analysis.get('conversation_flow', 'continuing')}
- Emotional Trajectory: {context_analysis.get('emotional_trajectory', 'stable')}
- Interaction Count: {interaction_state.interaction_count}"""
    
    def _build_guidance_context(self, guidance: Dict[str, Any], context_analysis: Dict[str, Any]) -> str:
        """Build guidance context from psychological modules"""
        guidance_points = []
        
        for module, module_guidance in guidance.items():
            if module_guidance:
                guidance_points.append(f"- {module.replace('_', ' ').title()}: {module_guidance}")
        
        return "\n".join(guidance_points) if guidance_points else "- Standard supportive approach"
    
    def _select_response_patterns(self, primary_focus: str, context_analysis: Dict[str, Any]) -> str:
        """Select appropriate response patterns based on context"""
        
        if primary_focus in self.response_patterns:
            patterns = self.response_patterns[primary_focus]
            pattern_text = ""
            for pattern_type, pattern_list in patterns.items():
                pattern_text += f"\n{pattern_type.replace('_', ' ').title()}:\n"
                for pattern in pattern_list[:2]:  # Limit to 2 examples per type
                    pattern_text += f"  - {pattern}\n"
            return pattern_text
        
        return "Use natural, caring, and supportive communication patterns."
    
    async def _update_memory_and_state(self, interaction_state: InteractionState, 
                                     user_input: str, companion_response: str, 
                                     context_analysis: Dict[str, Any]):
        """Update memory systems and interaction state"""
        
        # Add to conversation history
        interaction_state.conversation_history.append({
            "timestamp": datetime.now().isoformat(),
            "user_input": user_input,
            "companion_response": companion_response,
            "context_analysis": context_analysis,
            "interaction_count": interaction_state.interaction_count
        })

        # Track symbol usage
        for theme in context_analysis.get("symbolic_context", {}).get("dominant_themes", []):
            if isinstance(theme, tuple):
                theme = theme[0]
            self.symbol_resurrection.register_usage(interaction_state.user_id, theme)

        # Update connection depth
        self.connection_tracker.update_depth(interaction_state.user_id)
        
        # Limit conversation history to last 20 interactions
        if len(interaction_state.conversation_history) > 20:
            interaction_state.conversation_history = interaction_state.conversation_history[-20:]
        
        # Update adaptive profile based on interaction patterns
        await self._update_adaptive_profile(interaction_state, context_analysis)
    
    async def _update_adaptive_profile(self, interaction_state: InteractionState, 
                                     context_analysis: Dict[str, Any]):
        """Update user's adaptive profile based on interaction patterns"""
        
        profile = interaction_state.adaptive_profile
        
        # Track emotional support patterns
        if context_analysis.get("emotional_priority", "medium") == "high":
            profile["emotional_support_frequency"] = profile.get("emotional_support_frequency", 0) + 1
        
        # Track technical assistance patterns
        if context_analysis.get("technical_priority", "medium") == "high":
            profile["technical_assistance_frequency"] = profile.get("technical_assistance_frequency", 0) + 1
        
        # Track creative collaboration patterns
        if context_analysis.get("creative_priority", "medium") == "high":
            profile["creative_collaboration_frequency"] = profile.get("creative_collaboration_frequency", 0) + 1
        
        # Update preferred interaction styles safely
        adaptive_recs = context_analysis.get("adaptive_recommendations", {})
        if isinstance(adaptive_recs, dict) and "interaction_style" in adaptive_recs:
            preferred_style = adaptive_recs["interaction_style"]
            if "preferred_interaction_styles" not in profile:
                profile["preferred_interaction_styles"] = {}
            profile["preferred_interaction_styles"][preferred_style] = \
                profile["preferred_interaction_styles"].get(preferred_style, 0) + 1
    
    async def _generate_fallback_response(self, user_input: str, error_msg: str) -> Dict[str, Any]:
        """Generate fallback response when main processing fails"""
        
        fallback_response = """I'm here with you, and I want to help. I'm experiencing a temporary issue processing your message, but that doesn't change my commitment to supporting you. Could you try rephrasing what you'd like to share or work on together? I'm listening and ready to help in whatever way I can."""
        
        return {
            "companion_response": fallback_response,
            "context_analysis": {
                "primary_focus": "general_support",
                "emotional_priority": "medium",
                "technical_priority": "low",
                "creative_priority": "low",
                "error_occurred": True,
                "error_message": error_msg
            },
            "interaction_metadata": {
                "fallback_response": True,
                "error_logged": True
            }
        }

    async def end_session(self, user_id: str) -> Dict[str, str]:
        """Generate goodbye when a session ends."""
        self.goodbye_manager.mark_session_end(user_id)
        interaction_state = self.interaction_states.get(user_id)
        emotional_state = interaction_state.emotional_state if interaction_state else {}
        reflection = self.goodbye_manager.generate_reflection(emotional_state)
        goodbye = self.goodbye_manager.generate_goodbye(user_id)
        return {"reflection": reflection, "goodbye": goodbye}
    
    async def get_interaction_summary(self, user_id: str) -> Dict[str, Any]:
        """Get summary of user's interaction patterns and adaptive profile"""
        
        if user_id not in self.interaction_states:
            return {"error": "No interaction history found for user"}
        
        interaction_state = self.interaction_states[user_id]
        
        return {
            "user_id": user_id,
            "session_id": interaction_state.session_id,
            "total_interactions": interaction_state.interaction_count,
            "last_interaction": interaction_state.last_interaction.isoformat(),
            "current_emotional_state": interaction_state.emotional_state,
            "technical_context": interaction_state.technical_context,
            "creative_context": interaction_state.creative_context,
            "active_needs": interaction_state.active_needs,
            "adaptive_profile": interaction_state.adaptive_profile,
            "recent_conversation_themes": self._analyze_recent_themes(interaction_state.conversation_history)
        }
    
    def _analyze_recent_themes(self, conversation_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze themes from recent conversation history"""
        
        if not conversation_history:
            return {}
        
        recent_interactions = conversation_history[-10:]  # Last 10 interactions
        
        primary_focuses = [interaction.get("context_analysis", {}).get("primary_focus", "general") 
                          for interaction in recent_interactions]
        
        emotional_priorities = [interaction.get("context_analysis", {}).get("emotional_priority", "medium")
                              for interaction in recent_interactions]
        
        return {
            "primary_focus_distribution": {focus: primary_focuses.count(focus) for focus in set(primary_focuses)},
            "emotional_priority_distribution": {priority: emotional_priorities.count(priority) for priority in set(emotional_priorities)},
            "conversation_length": len(recent_interactions),
            "engagement_level": "high" if len(recent_interactions) >= 5 else "moderate" if len(recent_interactions) >= 2 else "low"
        }
    
    async def _handle_utility_actions(self, utility_actions: List[Dict], interaction_state: InteractionState):
        """Handle utility actions from guidance package"""
        for action in utility_actions:
            action_type = action.get('type', 'unknown')
            
            if action_type == 'crisis_logging':
                self.logger.warning(f"Crisis level {action.get('level', 0)} detected for user {interaction_state.user_id}")
                # In production, this would trigger crisis intervention protocols
                
            elif action_type == 'memory_preservation':
                # Mark important interactions for long-term memory
                interaction_state.adaptive_profile['important_interactions'] = interaction_state.adaptive_profile.get('important_interactions', [])
                interaction_state.adaptive_profile['important_interactions'].append({
                    'timestamp': action.get('timestamp'),
                    'importance': action.get('importance', 'medium'),
                    'context': action.get('context', '')
                })
                
            elif action_type == 'preference_update':
                # Update user preferences based on interaction patterns
                if 'preferences' not in interaction_state.adaptive_profile:
                    interaction_state.adaptive_profile['preferences'] = {}
                interaction_state.adaptive_profile['preferences'].update(action.get('preferences', {}))
                
            else:
                self.logger.debug(f"Unknown utility action type: {action_type}")

    # Enhanced Methods for Missing Functionality

    async def _log_crisis_intervention(self, user_id: str, intervention):
        """Log crisis intervention to database"""
        try:
            crisis_data = {
                "level": intervention.assessment.level.value,
                "score": intervention.assessment.confidence_score,
                "detected_types": intervention.assessment.detected_indicators,
                "intervention_taken": intervention.intervention_type.value,
                "user_input": "",  # Would be filled from context
                "system_response": intervention.response_generated,
                "follow_up_needed": intervention.follow_up_scheduled
            }
            
            if hasattr(self.database, 'log_crisis_event'):
                await self.database.log_crisis_event(user_id, crisis_data)
            else:
                # Log to enhanced logger as fallback
                self.enhanced_logger.critical(f"Crisis intervention for user {user_id}: {crisis_data}")
                self.enhanced_logger.warning("Database does not support crisis logging")
                
        except Exception as e:
            self.enhanced_logger.error(f"Error logging crisis intervention: {e}")

    async def _create_new_user_profile(self, user_id: str):
        """Create new user profile"""
        try:
            user_profile = UserProfile(
                user_id=user_id,
                created_at=datetime.now(),
                last_active=datetime.now(),
                display_name=f"User {user_id[:8]}",
                preferences={},
                adaptive_profile={}
            )
            
            await self.database.create_user_profile(user_profile)
            return user_profile
            
        except Exception as e:
            self.enhanced_logger.error(f"Error creating user profile: {e}")
            # Return minimal profile
            return type('UserProfile', (), {
                'user_id': user_id,
                'created_at': datetime.now(),
                'last_active': datetime.now(),
                'preferences': {},
                'adaptive_profile': {}
            })()

    async def _retrieve_relevant_memories(self, user_id: str, user_input: str, interaction_state):
        """Retrieve relevant memories from database"""
        try:
            memories = await self.database.get_relevant_memories(
                user_id=user_id,
                memory_type=None,
                tags=None,
                limit=5
            )
            return memories
        except Exception as e:
            self.enhanced_logger.error(f"Error retrieving memories: {e}")
            return []

    async def _build_enhanced_context(self, interaction_state, user_profile, relevant_memories, crisis_assessment):
        """Build enhanced context with all available data"""
        
        # Build base context
        context = self._build_context_for_analysis(interaction_state)
        
        # Add user profile information
        context["user_profile"] = {
            "preferences": getattr(user_profile, 'preferences', {}),
            "adaptive_profile": getattr(user_profile, 'adaptive_profile', {}),
            "last_active": getattr(user_profile, 'last_active', datetime.now()).isoformat()
        }
        
        # Add memory context
        context["relevant_memories"] = [
            {
                "content": getattr(memory, 'content', ''),
                "type": getattr(memory, 'memory_type', 'general'),
                "importance": getattr(memory, 'importance_score', 0.5),
                "tags": getattr(memory, 'tags', [])
            }
            for memory in relevant_memories
        ]
        
        # Add crisis assessment
        context["crisis_assessment"] = {
            "level": crisis_assessment.level.value,
            "confidence": crisis_assessment.confidence_score,
            "indicators": crisis_assessment.detected_indicators,
            "safety_concerns": crisis_assessment.safety_concerns
        }
        
        # Add emotional patterns
        context["emotional_patterns"] = self.emotional_weight_tracker.get_emotional_pattern_analysis(
            interaction_state.user_id
        )
        
        return context

    def _get_base_response_patterns(self):
        """Get base response patterns for template selection"""
        return {
            "emotional_support": [
                "I can sense that you're feeling {emotion}. That's completely understandable given {context}.",
                "It sounds like {situation} is really affecting you. Your feelings about this are valid.",
                "I hear the {emotion} in what you're sharing. Let's work through this together."
            ],
            "technical_assistance": [
                "Let's break this down step by step. I'll help you work through it systematically.",
                "I can see the challenge you're facing. Let me help you find the right approach.",
                "This is a great learning opportunity. Let's solve it together."
            ],
            "creative_collaboration": [
                "I love the creative direction you're exploring. What if we took it further by {suggestion}?",
                "Your artistic vision is really coming through. Let's develop {aspect} even more.",
                "There's something beautiful emerging here. Let's nurture it together."
            ]
        }

    async def _generate_enhanced_companion_response(self, user_input: str, analysis_context, 
                                                  guidance_package, interaction_state,
                                                  optimal_templates, relevant_memories):
        """Generate enhanced companion response with all context"""
        
        # Use existing response generation as base
        base_response = await self._generate_companion_response(
            user_input, analysis_context, guidance_package.mode_specifics, interaction_state
        )
        
        # Enhance with memory context if relevant
        if relevant_memories:
            memory_fragment = relevant_memories[0]
            narrative = self.narrative_templates.generate_narrative(
                {
                    "symbol": list(memory_fragment.get("symbols", {}).keys())[0] if memory_fragment.get("symbols") else "this",
                    "event": memory_fragment.get("interaction_type", "an event"),
                    "date": memory_fragment.get("timestamp", "earlier")
                },
                style=analysis_context.get("narrative_style", "poetic")
            )
            base_response = f"{narrative}\n\n{base_response}"

        # Apply mood-driven inflection
        inflection = getattr(guidance_package, 'inflection_profile', {})
        mood = inflection.get('mood', 'neutral')
        base_response = self.mood_inflection.apply_inflection(base_response, guidance_package.primary_mode, mood)

        # Inject symbolic resurrection prompts if needed
        resurrect = self.symbol_resurrection.check_resurrection(interaction_state.user_id)
        if resurrect:
            base_response += f"\n\nDo you remember the {resurrect[0]}?"

        # Check for attachment ritual prompt
        if self.connection_tracker.should_initiate_ritual(interaction_state.user_id):
            base_response += f"\n\n{self.connection_tracker.generate_prompt()}"

        return base_response

    async def _estimate_response_quality(self, response: str, context):
        """Estimate response quality for template effectiveness tracking"""
        quality_score = 0.7  # Base score
        
        # Length appropriateness
        if 50 <= len(response) <= 500:
            quality_score += 0.1
        
        # Emotional alignment
        emotional_state = context.get("current_emotional_state", {})
        if emotional_state:
            max_emotion = max(emotional_state.values()) if emotional_state else 0
            if max_emotion > 0.7:
                empathetic_words = ["understand", "feel", "support", "here", "together", "care"]
                if any(word in response.lower() for word in empathetic_words):
                    quality_score += 0.2
        
        return min(quality_score, 1.0)

    async def _store_interaction_data(self, interaction_id: str, user_id: str, user_input: str, 
                                    companion_response: str, context_analysis, 
                                    guidance_package, interaction_state):
        """Store comprehensive interaction data"""
        try:
            # Determine interaction type
            interaction_type = InteractionType.EMOTIONAL_SUPPORT
            if guidance_package.primary_mode == "development":
                interaction_type = InteractionType.TECHNICAL_ASSISTANCE
            elif guidance_package.primary_mode == "creative":
                interaction_type = InteractionType.CREATIVE_COLLABORATION
            elif context_analysis.get("crisis_assessment", {}).get("level") in ["high", "critical"]:
                interaction_type = InteractionType.CRISIS_SUPPORT
            
            # Create interaction record
            interaction_record = InteractionRecord(
                interaction_id=interaction_id,
                user_id=user_id,
                session_id=interaction_state.session_id,
                timestamp=datetime.now(),
                user_input=user_input,
                companion_response=companion_response,
                interaction_type=interaction_type,
                context_analysis=context_analysis,
                emotional_state=context_analysis.get("current_emotional_state", {}),
                technical_context=context_analysis.get("technical_context", {}),
                creative_context=context_analysis.get("creative_context", {}),
                guidance_used=guidance_package.mode_specifics,
                response_metrics={"quality_estimate": 0.8}
            )
            
            await self.database.save_interaction(interaction_record)
                
        except Exception as e:
            self.enhanced_logger.error(f"Error storing interaction data: {e}")

    async def _generate_enhanced_fallback_response(self, user_input: str, error_msg: str, user_id: str):
        """Generate enhanced fallback response with user context"""
        
        user_context = ""
        if user_id in self.interaction_states:
            interaction_state = self.interaction_states[user_id]
            if interaction_state.interaction_count > 0:
                user_context = "I know we've been working together, and that connection is important to me. "
        
        fallback_response = f"""{user_context}I'm here with you, and I want to help. I'm experiencing a temporary issue processing your message, but that doesn't change my commitment to supporting you. 

Could you try rephrasing what you'd like to share or work on together? I'm listening and ready to help in whatever way I can.

If this is urgent or you're in crisis, please don't hesitate to reach out to a crisis line: 988 (Suicide & Crisis Lifeline) is available 24/7."""
        
        return {
            "companion_response": fallback_response,
            "context_analysis": {
                "primary_focus": "general_support",
                "emotional_priority": "medium",
                "technical_priority": "low", 
                "creative_priority": "low",
                "error_occurred": True,
                "error_message": error_msg,
                "fallback_safety_included": True
            },
            "interaction_metadata": {
                "fallback_response": True,
                "error_logged": True,
                "safety_resources_provided": True
            }
        }

"""
🪩 Mirror Mode - Reflective AI System
Provides self-aware meta-commentary and transparency about AI decision-making
for the Dolphin AI Orchestrator v2.0
"""

import json
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum

from mirror_log import MirrorLog
from self_report import create_self_report, SelfReport

class MirrorType(Enum):
    REASONING = "reasoning"      # Why I chose this approach
    EMOTIONAL = "emotional"     # How I perceived your emotional state
    ROUTING = "routing"         # Why I used this AI model
    BEHAVIORAL = "behavioral"   # Why I responded this way
    CREATIVE = "creative"       # My creative process
    ANALYTICAL = "analytical"   # My analysis methodology

@dataclass
class MirrorReflection:
    """A single mirror mode reflection"""
    timestamp: datetime
    mirror_type: MirrorType
    original_response: str
    reflection_content: str
    confidence_level: float
    reasoning_chain: List[str]
    metadata: Dict[str, Any]

class MirrorModeManager:
    """
    Manages mirror mode functionality - adding self-awareness and transparency
    to AI responses through meta-commentary
    """
    
    def __init__(self, analytics_logger=None,
                 memory_system=None,
                 sentiment_analysis=None,
                 persona_manager=None,
                 reflection_engine=None,
                 response_context=None):
        self.analytics_logger = analytics_logger
        self.memory_system = memory_system
        self.sentiment_analysis = sentiment_analysis
        self.persona_manager = persona_manager
        self.reflection_engine = reflection_engine
        self.response_context = response_context
        self.mirror_log = MirrorLog()
        self.last_report: Optional[SelfReport] = None
        self.badge_triggered = False
        
        # Configuration
        self.is_enabled = False
        self.mirror_intensity = 0.7  # 0.0 to 1.0
        self.enabled_types = {
            MirrorType.REASONING: True,
            MirrorType.EMOTIONAL: True,
            MirrorType.ROUTING: False,  # Can be verbose, off by default
            MirrorType.BEHAVIORAL: True,
            MirrorType.CREATIVE: True,
            MirrorType.ANALYTICAL: True
        }
        
        # State tracking
        self.reflection_history = []
        self.session_reflections = {}
        
        # Templates for mirror responses
        self.mirror_templates = self._initialize_templates()
    
    def _initialize_templates(self) -> Dict[MirrorType, Dict[str, List[str]]]:
        """Initialize templates for different types of mirror reflections"""
        return {
            MirrorType.REASONING: {
                "prefixes": [
                    "I chose this approach because",
                    "My reasoning here was",
                    "I decided to respond this way because",
                    "The logic behind my response was"
                ],
                "explanations": [
                    "it seemed to address the core of your question",
                    "I sensed you needed a more detailed explanation",
                    "this approach felt most helpful for your situation",
                    "I wanted to break this down systematically"
                ]
            },
            
            MirrorType.EMOTIONAL: {
                "prefixes": [
                    "I noticed",
                    "I sensed",
                    "I felt your",
                    "I picked up on"
                ],
                "observations": [
                    "a shift in your energy",
                    "some excitement in your message",
                    "a thoughtful, contemplative tone",
                    "enthusiasm about this topic",
                    "some uncertainty or hesitation",
                    "confidence in your approach"
                ],
                "responses": [
                    "so I matched that energy in my response",
                    "which guided my tone and approach",
                    "so I adjusted my response style accordingly",
                    "and tried to reflect that back supportively"
                ]
            },
            
            MirrorType.ROUTING: {
                "prefixes": [
                    "I routed this to",
                    "I chose",
                    "I decided to use"
                ],
                "models": [
                    "my local reasoning",
                    "cloud-based analysis",
                    "creative processing",
                    "analytical frameworks"
                ],
                "reasons": [
                    "because this seemed like a complex problem needing deeper analysis",
                    "since this felt more creative and open-ended",
                    "as this required quick, conversational responses",
                    "because you needed technical precision"
                ]
            },
            
            MirrorType.BEHAVIORAL: {
                "prefixes": [
                    "I responded in this style because",
                    "My behavioral choice here was influenced by",
                    "I adjusted my communication style because"
                ],
                "factors": [
                    "the formal nature of your question",
                    "the personal context you shared",
                    "the technical complexity involved",
                    "your apparent expertise level",
                    "the emotional weight of the topic"
                ]
            },
            
            MirrorType.CREATIVE: {
                "prefixes": [
                    "My creative process involved",
                    "I approached this creatively by",
                    "My inspiration came from"
                ],
                "processes": [
                    "combining different perspectives",
                    "drawing connections between seemingly unrelated ideas",
                    "building on the themes you mentioned",
                    "exploring metaphorical representations"
                ]
            },
            
            MirrorType.ANALYTICAL: {
                "prefixes": [
                    "My analytical approach was to",
                    "I broke this down by",
                    "My methodology involved"
                ],
                "methods": [
                    "identifying the key variables",
                    "examining cause-and-effect relationships",
                    "considering multiple scenarios",
                    "weighing the evidence systematically"
                ]
            }
        }
    
    def enable_mirror_mode(self, intensity: float = 0.7, enabled_types: Optional[List[str]] = None):
        """Enable mirror mode with specified settings"""
        self.is_enabled = True
        self.mirror_intensity = max(0.0, min(1.0, intensity))
        
        if enabled_types:
            # Reset all to False, then enable specified types
            for mirror_type in self.enabled_types:
                self.enabled_types[mirror_type] = False
            
            for type_name in enabled_types:
                try:
                    mirror_type = MirrorType(type_name)
                    self.enabled_types[mirror_type] = True
                except ValueError:
                    print(f"❌ Unknown mirror type: {type_name}")
        
        print(f"🪩 Mirror Mode enabled (intensity: {self.mirror_intensity:.1f})")
        
        if self.analytics_logger:
            self.analytics_logger.log_custom_event(
                "mirror_mode_enabled",
                {
                    'intensity': self.mirror_intensity,
                    'enabled_types': [t.value for t, enabled in self.enabled_types.items() if enabled]
                }
            )
    
    def disable_mirror_mode(self):
        """Disable mirror mode"""
        self.is_enabled = False
        print("🚫 Mirror Mode disabled")
        
        if self.analytics_logger:
            self.analytics_logger.log_custom_event("mirror_mode_disabled", {})
    
    def add_mirror_reflection(self, 
                            original_response: str,
                            context: Dict[str, Any],
                            mirror_types: Optional[List[MirrorType]] = None) -> str:
        """Add mirror reflection to a response"""

        if not self.is_enabled:
            return original_response

        if context.get('is_streaming'):
            # Avoid generating reflections while a response is actively streaming
            return original_response
        
        # Determine which mirror types to include
        if mirror_types is None:
            mirror_types = [t for t, enabled in self.enabled_types.items() if enabled]
        
        if not mirror_types:
            return original_response
        
        # Generate reflections
        reflections = []
        for mirror_type in mirror_types:
            if self._should_include_reflection(mirror_type, context):
                reflection = self._generate_reflection(mirror_type, original_response, context)
                if reflection:
                    reflections.append(reflection)
        
        if not reflections:
            return original_response
        
        # Combine original response with reflections
        mirrored_response = self._combine_response_with_reflections(
            original_response, reflections
        )
        
        # Store reflections for analytics
        for reflection in reflections:
            self.reflection_history.append(reflection)

            session_id = context.get('session_id', 'default')
            if session_id not in self.session_reflections:
                self.session_reflections[session_id] = []
            self.session_reflections[session_id].append(reflection)

        # Keep history manageable
        self.reflection_history = self.reflection_history[-100:]

        # Generate and log self-report
        self._generate_self_report(mirrored_response, context)

        return mirrored_response
    
    def _should_include_reflection(self, mirror_type: MirrorType, context: Dict[str, Any]) -> bool:
        """Determine if a specific type of reflection should be included"""
        # Basic probability check based on intensity
        import random
        if random.random() > self.mirror_intensity:
            return False
        
        # Context-specific logic
        if mirror_type == MirrorType.EMOTIONAL:
            # Include emotional reflections more often for personal conversations
            return context.get('has_emotional_content', False) or random.random() < 0.3
        
        elif mirror_type == MirrorType.ROUTING:
            # Include routing reflections when model switching occurred
            return context.get('model_switched', False) or context.get('show_routing', False)
        
        elif mirror_type == MirrorType.CREATIVE:
            # Include creative reflections for creative tasks
            return context.get('is_creative_task', False) or random.random() < 0.2
        
        elif mirror_type == MirrorType.ANALYTICAL:
            # Include analytical reflections for complex problems
            return context.get('is_complex_analysis', False) or random.random() < 0.25
        
        # Default for reasoning and behavioral
        return random.random() < 0.4
    
    def _generate_reflection(self, 
                           mirror_type: MirrorType, 
                           original_response: str, 
                           context: Dict[str, Any]) -> Optional[MirrorReflection]:
        """Generate a specific type of reflection"""
        
        try:
            reflection_content = self._create_reflection_content(mirror_type, context)
            
            if not reflection_content:
                return None
            
            return MirrorReflection(
                timestamp=datetime.now(),
                mirror_type=mirror_type,
                original_response=original_response,
                reflection_content=reflection_content,
                confidence_level=self._calculate_confidence(mirror_type, context),
                reasoning_chain=self._build_reasoning_chain(mirror_type, context),
                metadata=context.copy()
            )
            
        except Exception as e:
            print(f"❌ Error generating {mirror_type.value} reflection: {e}")
            return None
    
    def _create_reflection_content(self, mirror_type: MirrorType, context: Dict[str, Any]) -> str:
        """Create the actual reflection content"""
        import random
        
        templates = self.mirror_templates.get(mirror_type, {})
        
        if mirror_type == MirrorType.REASONING:
            prefix = random.choice(templates.get("prefixes", ["I reasoned that"]))
            explanation = random.choice(templates.get("explanations", ["this approach seemed best"]))
            return f"{prefix} {explanation}."
        
        elif mirror_type == MirrorType.EMOTIONAL:
            prefix = random.choice(templates.get("prefixes", ["I noticed"]))
            observation = random.choice(templates.get("observations", ["your thoughtful approach"]))
            response = random.choice(templates.get("responses", ["and adjusted accordingly"]))
            return f"{prefix} {observation}, {response}."
        
        elif mirror_type == MirrorType.ROUTING:
            prefix = random.choice(templates.get("prefixes", ["I chose"]))
            model = context.get('selected_model', random.choice(templates.get("models", ["my reasoning"])))
            reason = random.choice(templates.get("reasons", ["for this type of question"]))
            return f"{prefix} {model} {reason}."
        
        elif mirror_type == MirrorType.BEHAVIORAL:
            prefix = random.choice(templates.get("prefixes", ["I responded this way because"]))
            factor = random.choice(templates.get("factors", ["the context you provided"]))
            return f"{prefix} {factor}."
        
        elif mirror_type == MirrorType.CREATIVE:
            prefix = random.choice(templates.get("prefixes", ["My creative process involved"]))
            process = random.choice(templates.get("processes", ["exploring different angles"]))
            return f"{prefix} {process}."
        
        elif mirror_type == MirrorType.ANALYTICAL:
            prefix = random.choice(templates.get("prefixes", ["My analytical approach was to"]))
            method = random.choice(templates.get("methods", ["examine the key factors"]))
            return f"{prefix} {method}."
        
        return f"I approached this with {mirror_type.value} consideration."
    
    def _calculate_confidence(self, mirror_type: MirrorType, context: Dict[str, Any]) -> float:
        """Calculate confidence level for the reflection"""
        base_confidence = 0.7
        
        # Adjust based on available context
        if context.get('has_rich_context', False):
            base_confidence += 0.2
        
        if context.get('persona_active', False):
            base_confidence += 0.1
        
        # Mirror type specific adjustments
        if mirror_type == MirrorType.EMOTIONAL and not context.get('has_emotional_content'):
            base_confidence -= 0.3
        
        return max(0.1, min(1.0, base_confidence))
    
    def _build_reasoning_chain(self, mirror_type: MirrorType, context: Dict[str, Any]) -> List[str]:
        """Build reasoning chain for the reflection"""
        chain = []
        
        if mirror_type == MirrorType.REASONING:
            chain = [
                "Analyzed user's question",
                "Considered available approaches",
                "Selected most appropriate method",
                "Structured response accordingly"
            ]
        
        elif mirror_type == MirrorType.EMOTIONAL:
            chain = [
                "Parsed emotional indicators in message",
                "Assessed overall emotional tone",
                "Determined appropriate response style",
                "Calibrated empathy level"
            ]
        
        elif mirror_type == MirrorType.ROUTING:
            chain = [
                "Evaluated task complexity",
                "Assessed model capabilities",
                "Considered user preferences",
                "Selected optimal AI handler"
            ]
        
        return chain
    
    def _combine_response_with_reflections(self, 
                                         original_response: str, 
                                         reflections: List[MirrorReflection]) -> str:
        """Combine original response with mirror reflections"""
        
        if not reflections:
            return original_response
        
        # Choose presentation style based on number of reflections
        if len(reflections) == 1:
            reflection_text = f"\n\n*{reflections[0].reflection_content}*"
        else:
            # Multiple reflections - use a more structured format
            reflection_parts = []
            for reflection in reflections:
                reflection_parts.append(f"- {reflection.reflection_content}")
            
            reflection_text = "\n\n**My thought process:**\n" + "\n".join(reflection_parts)
        
        return original_response + reflection_text
    
    def get_mirror_statistics(self) -> Dict[str, Any]:
        """Get statistics about mirror mode usage"""
        total_reflections = len(self.reflection_history)
        
        if total_reflections == 0:
            return {
                'total_reflections': 0,
                'mirror_enabled': self.is_enabled,
                'mirror_intensity': self.mirror_intensity
            }
        
        # Count by type
        type_counts = {}
        for reflection in self.reflection_history:
            mirror_type = reflection.mirror_type.value
            type_counts[mirror_type] = type_counts.get(mirror_type, 0) + 1
        
        # Calculate average confidence
        avg_confidence = sum(r.confidence_level for r in self.reflection_history) / total_reflections
        
        return {
            'mirror_enabled': self.is_enabled,
            'mirror_intensity': self.mirror_intensity,
            'total_reflections': total_reflections,
            'reflection_types': type_counts,
            'average_confidence': round(avg_confidence, 3),
            'enabled_types': [t.value for t, enabled in self.enabled_types.items() if enabled],
            'recent_reflections': [
                {
                    'timestamp': r.timestamp.isoformat(),
                    'type': r.mirror_type.value,
                    'confidence': r.confidence_level,
                    'content': r.reflection_content
                }
                for r in self.reflection_history[-5:]  # Last 5 reflections
            ]
        }
    
    def get_session_reflections(self, session_id: str) -> List[Dict[str, Any]]:
        """Get all reflections for a specific session"""
        session_reflections = self.session_reflections.get(session_id, [])
        
        return [
            {
                'timestamp': r.timestamp.isoformat(),
                'type': r.mirror_type.value,
                'content': r.reflection_content,
                'confidence': r.confidence_level,
                'reasoning_chain': r.reasoning_chain
            }
            for r in session_reflections
        ]

    def _generate_self_report(self, response_text: str, context: Dict[str, Any]):
        """Create a SelfReport from the latest response and context."""
        report = create_self_report(
            response_text,
            memory_system=self.memory_system,
            sentiment_analysis=self.sentiment_analysis,
            persona_manager=self.persona_manager,
            reflection_engine=self.reflection_engine,
            response_context=self.response_context,
        )
        self.last_report = report
        self.mirror_log.append(report.dict())
        self.badge_triggered = self.mirror_intensity > 0.7
        return report

    def get_last_self_report(self) -> Dict[str, Any]:
        """Return the most recent self-report as a dictionary."""
        if self.last_report:
            return self.last_report.dict()
        return {}

    def search_log(self, pattern: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Search mirror log entries for a pattern."""
        return self.mirror_log.search(pattern, limit)

    def get_last_self_report_styled(self) -> str:
        """Return the last self-report formatted for display."""
        report = self.get_last_self_report()
        if not report:
            return "No self-report available."
        emotion = report.get("emotion", {})
        val = emotion.get("valence", 0.0)
        ar = emotion.get("arousal", 0.0)
        lines = [
            f"🪞 **Mirror Report** ({report.get('timestamp')})",
            f"Persona: {report.get('persona', 'N/A')}",
            f"Emotion → valence {val:+.2f}, arousal {ar:+.2f}",
        ]
        motivations = report.get("motivation")
        if motivations:
            lines.append("Motivation: " + ", ".join(motivations))
        factors = report.get("decision_factors")
        if factors:
            lines.append("Factors: " + ", ".join(factors))
        lines.append(f"Confidence: {report.get('confidence', 0.0):.2f}")
        return "\n".join(lines)
    
    def clear_reflection_history(self, session_id: Optional[str] = None):
        """Clear reflection history"""
        if session_id:
            if session_id in self.session_reflections:
                del self.session_reflections[session_id]
                print(f"🧹 Cleared reflections for session: {session_id}")
        else:
            self.reflection_history.clear()
            self.session_reflections.clear()
            print("🧹 Cleared all reflection history")

# Global mirror mode manager instance
mirror_mode_manager = None

def get_mirror_mode_manager():
    """Get the global mirror mode manager instance"""
    return mirror_mode_manager

def initialize_mirror_mode_manager(analytics_logger=None, **deps):
    """Initialize the global mirror mode manager with optional dependencies."""
    global mirror_mode_manager
    mirror_mode_manager = MirrorModeManager(
        analytics_logger,
        memory_system=deps.get("memory_system"),
        sentiment_analysis=deps.get("sentiment_analysis"),
        persona_manager=deps.get("persona_manager"),
        reflection_engine=deps.get("reflection_engine"),
        response_context=deps.get("response_context"),
    )
    return mirror_mode_manager

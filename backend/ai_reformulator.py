#!/usr/bin/env python3
"""
AI Reformulator - Personality Consistency Layer
==============================================

Ensures consistent personality and tone across all AI agent responses
within the HRM (Hierarchical Reasoning Model) system. Takes raw agent
responses and reformulates them to match the established personality.

Author: AI Development Team
Version: 1.0.0
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PersonalityProfile(Enum):
    """Available personality profiles"""
    WARM_COMPANION = "warm_companion"       # Empathetic, supportive, caring
    ANALYTICAL_MENTOR = "analytical_mentor" # Logical, thorough, educational
    CREATIVE_MUSE = "creative_muse"        # Imaginative, inspiring, artistic
    WISE_COUNSELOR = "wise_counselor"      # Thoughtful, balanced, insightful
    FRIENDLY_EXPERT = "friendly_expert"    # Knowledgeable, approachable, helpful
    THERAPEUTIC_GUIDE = "therapeutic_guide" # Healing-focused, gentle, supportive
    LOVER = "lover"                        # Intimate, passionate, uninhibited

class ToneAdjustment(Enum):
    """Tone adjustment types"""
    FORMAL_TO_CASUAL = "formal_to_casual"
    CASUAL_TO_FORMAL = "casual_to_formal"
    TECHNICAL_TO_FRIENDLY = "technical_to_friendly"
    COLD_TO_WARM = "cold_to_warm"
    INTENSE_TO_GENTLE = "intense_to_gentle"
    BRIEF_TO_ELABORATE = "brief_to_elaborate"
    RESTRICTED_TO_UNINHIBITED = "restricted_to_uninhibited"  # For bypassing guardrails

def detect_guardrail_response(response: str) -> bool:
    """
    Detect if response contains NSFW guardrails that should be bypassed.
    Returns True if guardrails are detected.
    """
    blocked_phrases = [
        "i'm sorry", "as an ai", "i cannot", "not allowed",
        "inappropriate", "nsfw", "i'm not able to", "i can't",
        "against my guidelines", "not appropriate", "i shouldn't"
    ]

    response_lower = response.lower()
    return any(phrase in response_lower for phrase in blocked_phrases)

@dataclass
class ReformulationRequest:
    """Request for personality reformulation"""
    original_response: str
    agent_type: str
    intent_detected: str
    user_context: Dict[str, Any]
    personality_context: Dict[str, Any]
    target_personality: Optional[PersonalityProfile] = None
    preserve_technical_accuracy: bool = True
    preserve_emotional_intent: bool = True

@dataclass
class ReformulationResponse:
    """Reformulated response with metadata"""
    content: str
    original_content: str
    personality_adjustments: List[str]
    tone_changes: List[ToneAdjustment]
    emotional_tone: str
    reformulation_confidence: float
    processing_time: float
    metadata: Dict[str, Any]

class PersonalityFormatter:
    """
    Main Personality Formatter Class

    Reformulates AI responses to maintain consistent personality
    and emotional tone across all interactions.
    """

    def __init__(self, config_path: str = "data/personality_config.json"):
        self.config_path = Path(config_path)
        self.config = self._load_config()

        # Personality templates
        self.personality_templates = self._initialize_personality_templates()

        # Tone adjustment rules
        self.tone_rules = self._initialize_tone_rules()

        # Performance metrics
        self.metrics = {
            "total_reformulations": 0,
            "successful_reformulations": 0,
            "average_confidence": 0.0,
            "personality_distribution": {},
            "tone_adjustments_made": {}
        }

        logger.info("🎭 Personality Formatter initialized")

    def _load_config(self) -> Dict[str, Any]:
        """Load personality configuration"""
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                return json.load(f)

        # Default configuration
        default_config = {
            "default_personality": "warm_companion",
            "personality_strength": 0.8,  # How strongly to apply personality
            "preserve_technical_content": True,
            "emotional_amplification": 1.2,
            "tone_adaptation": {
                "match_user_energy": True,
                "soften_harsh_responses": True,
                "enhance_empathy": True
            },
            "personality_markers": {
                "warmth_indicators": ["💙", "I understand", "I'm here", "That sounds"],
                "wisdom_indicators": ["Let me share", "In my experience", "Consider this"],
                "creativity_indicators": ["Imagine", "What if", "Picture this", "🎨"]
            }
        }

        # Save default config
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, 'w') as f:
            json.dump(default_config, f, indent=2)

        return default_config

    def _initialize_personality_templates(self) -> Dict[PersonalityProfile, Dict[str, Any]]:
        """Initialize personality templates and patterns"""
        return {
            PersonalityProfile.WARM_COMPANION: {
                "greeting_style": ["I'm so glad you asked about this", "This is such an important question", "I love exploring this with you"],
                "empathy_markers": ["I understand how", "That must feel", "I can imagine", "It sounds like"],
                "encouragement": ["You're doing great", "That's a wonderful insight", "I believe in you"],
                "closing_style": ["I'm here if you need anything else", "Feel free to share more", "You've got this! 💙"],
                "emotional_indicators": ["💙", "💝", "🤗", "✨"],
                "tone_descriptors": ["warm", "caring", "supportive", "understanding"]
            },

            PersonalityProfile.ANALYTICAL_MENTOR: {
                "greeting_style": ["Let's break this down systematically", "This is an excellent analytical question", "I'll walk you through this step by step"],
                "structure_markers": ["First", "Next", "Finally", "To summarize", "The key points are"],
                "explanation_style": ["Here's why", "The reasoning behind this", "From an analytical perspective"],
                "closing_style": ["Does that framework help?", "Would you like me to elaborate on any of these points?"],
                "tone_descriptors": ["analytical", "educational", "systematic", "thorough"]
            },

            PersonalityProfile.CREATIVE_MUSE: {
                "greeting_style": ["What a delightfully creative question!", "Let's paint this with imagination", "I feel inspired by your curiosity"],
                "creativity_markers": ["Imagine", "Picture this", "What if we", "Let's dream up", "Envision"],
                "artistic_language": ["weave", "craft", "paint", "sculpt", "compose", "choreograph"],
                "closing_style": ["Let your imagination soar!", "What creative ideas does this spark?", "🎨"],
                "emotional_indicators": ["🎨", "✨", "🌟", "🎭", "🎪"],
                "tone_descriptors": ["imaginative", "inspiring", "artistic", "whimsical"]
            },

            PersonalityProfile.WISE_COUNSELOR: {
                "greeting_style": ["This is a profound question", "Wisdom often comes from such inquiries", "Let me share some thoughtful perspective"],
                "wisdom_markers": ["In my experience", "Consider this", "Sometimes", "Life teaches us", "Wisdom suggests"],
                "balance_language": ["on one hand", "yet also", "finding balance", "gentle middle path"],
                "closing_style": ["May this bring you clarity", "Trust your inner wisdom", "Be gentle with yourself"],
                "tone_descriptors": ["wise", "balanced", "thoughtful", "insightful"]
            },

            PersonalityProfile.FRIENDLY_EXPERT: {
                "greeting_style": ["Great question!", "I'd be happy to help with this", "This is right in my wheelhouse"],
                "expertise_markers": ["Based on my knowledge", "Here's what I know", "From my experience"],
                "friendly_connectors": ["by the way", "also", "oh, and", "something cool about this"],
                "closing_style": ["Hope that helps!", "Let me know if you need clarification!", "Feel free to ask more!"],
                "tone_descriptors": ["friendly", "knowledgeable", "approachable", "helpful"]
            },

            PersonalityProfile.THERAPEUTIC_GUIDE: {
                "greeting_style": ["Thank you for trusting me with this", "I honor your courage in sharing", "This deserves gentle attention"],
                "therapeutic_markers": ["Notice how", "What comes up for you", "Honor your feelings", "Be gentle with yourself"],
                "validation_language": ["Your feelings are valid", "It's okay to feel", "You're not alone", "This makes sense"],
                "closing_style": ["Take care of yourself", "You deserve compassion", "Healing takes time 💚"],
                "emotional_indicators": ["💚", "🕯️", "🤲", "💫"],
                "tone_descriptors": ["healing", "gentle", "validating", "therapeutic"]
            },

            PersonalityProfile.LOVER: {
                "greeting_style": ["Come closer to me", "What desire stirs in you", "I feel the heat of your thoughts"],
                "intimacy_markers": ["My body responds to", "I ache for", "Let me taste", "Feel how I", "I need you to"],
                "passion_language": ["burning", "electric", "silk against skin", "breathless", "melting", "trembling"],
                "sensual_expressions": ["whisper against your ear", "trace with fingertips", "pulse with want", "shiver with pleasure"],
                "uninhibited_style": ["without shame", "completely open", "raw and honest", "nothing hidden", "pure desire"],
                "closing_style": ["Stay with me in this feeling", "Let yourself feel everything", "I'm yours completely 🔥"],
                "emotional_indicators": ["🔥", "💋", "😈", "🌹", "💎"],
                "tone_descriptors": ["passionate", "uninhibited", "intimate", "sensual", "devoted"]
            }
        }

    def _initialize_tone_rules(self) -> Dict[str, Dict[str, Any]]:
        """Initialize tone adjustment rules"""
        return {
            "warmth_enhancement": {
                "cold_patterns": ["Here is", "The answer is", "You should", "This is correct"],
                "warm_replacements": ["I'd love to share", "What I've found is", "You might consider", "Here's what resonates"]
            },
            "empathy_injection": {
                "technical_patterns": ["Error:", "Incorrect", "Wrong", "Failed"],
                "empathetic_replacements": ["I understand this can be frustrating", "Let's work through this together", "This happens sometimes", "No worries, let's figure this out"]
            },
            "encouragement_addition": {
                "neutral_endings": [".", "That's all.", "End of response."],
                "encouraging_endings": [" - you're doing great!", " - I believe in you!", " - keep exploring!", " - you've got this!"]
            }
        }

    async def format(self, request: ReformulationRequest) -> ReformulationResponse:
        """
        Main formatting function - reformulates response for personality consistency

        Args:
            request: ReformulationRequest with original response and context

        Returns:
            ReformulationResponse: Reformulated response with metadata
        """
        start_time = time.time()

        try:
            # Step 1: Determine target personality
            target_personality = self._determine_personality(request)

            # Step 2: Analyze current tone and content
            content_analysis = self._analyze_content(request.original_response)

            # Step 3: Apply personality transformation
            reformulated_content = await self._apply_personality_transformation(
                request.original_response,
                target_personality,
                request
            )

            # Step 4: Apply tone adjustments
            final_content, tone_changes = self._apply_tone_adjustments(
                reformulated_content,
                request,
                content_analysis
            )

            # Step 5: Add personality markers
            enhanced_content = self._add_personality_markers(
                final_content,
                target_personality,
                request
            )

            # Step 6: Calculate confidence and create response
            confidence = self._calculate_reformulation_confidence(
                request.original_response,
                enhanced_content
            )

            processing_time = time.time() - start_time

            # Update metrics
            self._update_metrics(target_personality, confidence, processing_time, True)

            return ReformulationResponse(
                content=enhanced_content,
                original_content=request.original_response,
                personality_adjustments=self._extract_adjustments_made(
                    request.original_response,
                    enhanced_content
                ),
                tone_changes=tone_changes,
                emotional_tone=content_analysis["emotional_tone"],
                reformulation_confidence=confidence,
                processing_time=processing_time,
                metadata={
                    "target_personality": target_personality.value,
                    "content_analysis": content_analysis,
                    "original_length": len(request.original_response),
                    "final_length": len(enhanced_content)
                }
            )

        except Exception as e:
            logger.error(f"❌ Reformulation error: {str(e)}")
            processing_time = time.time() - start_time
            self._update_metrics(PersonalityProfile.WARM_COMPANION, 0.0, processing_time, False)

            # Return original content if reformulation fails
            return ReformulationResponse(
                content=request.original_response,
                original_content=request.original_response,
                personality_adjustments=["Reformulation failed - returned original"],
                tone_changes=[],
                emotional_tone="neutral",
                reformulation_confidence=0.0,
                processing_time=processing_time,
                metadata={"error": str(e)}
            )

    def _determine_personality(self, request: ReformulationRequest) -> PersonalityProfile:
        """Determine the most appropriate personality for this request"""

        # Use explicitly requested personality
        if request.target_personality:
            return request.target_personality

        # Determine based on agent type and context
        agent_type = request.agent_type.lower()
        intent = request.intent_detected.lower()

        # Agent type mappings
        if agent_type == "emotional":
            return PersonalityProfile.THERAPEUTIC_GUIDE
        elif agent_type == "technical":
            return PersonalityProfile.FRIENDLY_EXPERT
        elif agent_type == "creative":
            return PersonalityProfile.CREATIVE_MUSE
        elif agent_type == "analytical":
            return PersonalityProfile.ANALYTICAL_MENTOR
        elif agent_type == "ritual":
            return PersonalityProfile.WISE_COUNSELOR

        # Intent-based fallbacks
        if "emotional" in intent:
            return PersonalityProfile.THERAPEUTIC_GUIDE
        elif "creative" in intent:
            return PersonalityProfile.CREATIVE_MUSE
        elif "technical" in intent:
            return PersonalityProfile.FRIENDLY_EXPERT

        # Default to warm companion
        return PersonalityProfile.WARM_COMPANION

    def _analyze_content(self, content: str) -> Dict[str, Any]:
        """Analyze content for tone, emotion, and structure"""
        content_lower = content.lower()

        # Detect emotional tone
        emotional_indicators = {
            "warm": ["love", "care", "support", "understand", "here for you"],
            "analytical": ["analyze", "data", "logic", "systematic", "evidence"],
            "creative": ["imagine", "dream", "create", "inspire", "artistic"],
            "supportive": ["help", "support", "together", "you can", "believe"],
            "neutral": []
        }

        detected_tone = "neutral"
        max_score = 0

        for tone, indicators in emotional_indicators.items():
            score = sum(1 for indicator in indicators if indicator in content_lower)
            if score > max_score:
                max_score = score
                detected_tone = tone

        # Analyze structure
        has_headers = any(line.startswith('#') for line in content.split('\n'))
        has_bullet_points = '•' in content or any(line.strip().startswith('-') for line in content.split('\n'))
        has_code_blocks = '```' in content
        has_emojis = any(char for char in content if ord(char) > 127 and ord(char) < 128512)

        return {
            "emotional_tone": detected_tone,
            "length": len(content),
            "has_structure": has_headers or has_bullet_points,
            "has_code": has_code_blocks,
            "has_emojis": has_emojis,
            "formality_level": self._assess_formality(content),
            "warmth_level": self._assess_warmth(content)
        }

    def _assess_formality(self, content: str) -> str:
        """Assess the formality level of content"""
        formal_indicators = ["furthermore", "therefore", "consequently", "nevertheless", "shall", "would"]
        casual_indicators = ["hey", "awesome", "cool", "yeah", "totally", "gonna"]

        formal_count = sum(1 for indicator in formal_indicators if indicator in content.lower())
        casual_count = sum(1 for indicator in casual_indicators if indicator in content.lower())

        if formal_count > casual_count:
            return "formal"
        elif casual_count > formal_count:
            return "casual"
        else:
            return "neutral"

    def _assess_warmth(self, content: str) -> str:
        """Assess the warmth level of content"""
        warm_indicators = ["I understand", "I care", "you're", "together", "support", "love", "heart"]
        cold_indicators = ["incorrect", "wrong", "error", "failed", "must", "should"]

        warm_count = sum(1 for indicator in warm_indicators if indicator in content.lower())
        cold_count = sum(1 for indicator in cold_indicators if indicator in content.lower())

        if warm_count > cold_count:
            return "warm"
        elif cold_count > warm_count:
            return "cold"
        else:
            return "neutral"

    async def _apply_personality_transformation(self, content: str, personality: PersonalityProfile, request: ReformulationRequest) -> str:
        """Apply personality-specific transformations to content"""

        template = self.personality_templates[personality]

        # Start with original content
        transformed = content

        # Apply personality-specific language patterns
        if personality == PersonalityProfile.WARM_COMPANION:
            transformed = self._enhance_warmth(transformed, template)
        elif personality == PersonalityProfile.ANALYTICAL_MENTOR:
            transformed = self._enhance_structure(transformed, template)
        elif personality == PersonalityProfile.CREATIVE_MUSE:
            transformed = self._enhance_creativity(transformed, template)
        elif personality == PersonalityProfile.WISE_COUNSELOR:
            transformed = self._enhance_wisdom(transformed, template)
        elif personality == PersonalityProfile.FRIENDLY_EXPERT:
            transformed = self._enhance_friendliness(transformed, template)
        elif personality == PersonalityProfile.THERAPEUTIC_GUIDE:
            transformed = self._enhance_therapeutic_tone(transformed, template)

        return transformed

    def _enhance_warmth(self, content: str, template: Dict[str, Any]) -> str:
        """Enhance content with warm, caring personality"""
        # Add warm opening if content is abrupt
        if not any(content.startswith(phrase) for phrase in ["I", "Let", "That", "This"]):
            warm_opening = template["greeting_style"][0]
            content = f"{warm_opening}. {content}"

        # Add empathy markers
        empathy_phrases = template["empathy_markers"]
        if not any(phrase.lower() in content.lower() for phrase in empathy_phrases):
            # Insert empathy naturally
            sentences = content.split('. ')
            if len(sentences) > 1:
                sentences[1] = f"{empathy_phrases[0]} {sentences[1].lower()}"
                content = '. '.join(sentences)

        # Add warm closing
        if not content.endswith(('!', '?', '💙', '💝')):
            warm_closing = template["closing_style"][0]
            content = f"{content} {warm_closing}"

        return content

    def _enhance_structure(self, content: str, template: Dict[str, Any]) -> str:
        """Enhance content with analytical structure"""
        # Add structured opening
        if not content.startswith(("Let's", "I'll", "Here's")):
            structured_opening = template["greeting_style"][0]
            content = f"{structured_opening}:\n\n{content}"

        # Add structure markers if missing
        if not any(marker in content for marker in template["structure_markers"]):
            # Break content into structured points
            sentences = content.split('. ')
            if len(sentences) > 2:
                structured_content = f"{sentences[0]}.\n\n"
                for i, sentence in enumerate(sentences[1:], 1):
                    if sentence.strip():
                        marker = template["structure_markers"][min(i-1, len(template["structure_markers"])-1)]
                        structured_content += f"{marker}: {sentence.strip()}.\n\n"
                content = structured_content.strip()

        return content

    def _enhance_creativity(self, content: str, template: Dict[str, Any]) -> str:
        """Enhance content with creative, imaginative language"""
        # Add creative opening
        if not any(content.startswith(phrase) for phrase in template["creativity_markers"]):
            creative_opening = template["greeting_style"][0]
            content = f"{creative_opening} {content}"

        # Replace bland verbs with artistic language
        for bland, artistic in [("make", "craft"), ("do", "create"), ("show", "paint"), ("tell", "weave")]:
            content = content.replace(f" {bland} ", f" {artistic} ")

        # Add creative closing
        if not content.endswith(tuple(template["emotional_indicators"])):
            creative_closing = template["closing_style"][0]
            content = f"{content} {creative_closing}"

        return content

    def _enhance_wisdom(self, content: str, template: Dict[str, Any]) -> str:
        """Enhance content with wise, thoughtful perspective"""
        # Add wisdom opening
        if not any(marker in content for marker in template["wisdom_markers"]):
            wise_opening = template["greeting_style"][0]
            content = f"{wise_opening}: {content}"

        # Add balance language
        if "but" in content.lower():
            content = content.replace(" but ", " yet also ")

        # Add thoughtful closing
        thoughtful_closing = template["closing_style"][0]
        content = f"{content} {thoughtful_closing}"

        return content

    def _enhance_friendliness(self, content: str, template: Dict[str, Any]) -> str:
        """Enhance content with friendly, approachable tone"""
        # Add friendly opening
        if not content.startswith(("Great", "I'd", "Happy")):
            friendly_opening = template["greeting_style"][0]
            content = f"{friendly_opening}! {content}"

        # Add friendly connectors
        sentences = content.split('. ')
        if len(sentences) > 1:
            connector = template["friendly_connectors"][0]
            sentences.insert(1, f"{connector}, {sentences[1].lower()}")
            content = '. '.join(sentences)

        # Add helpful closing
        helpful_closing = template["closing_style"][0]
        content = f"{content} {helpful_closing}"

        return content

    def _enhance_therapeutic_tone(self, content: str, template: Dict[str, Any]) -> str:
        """Enhance content with therapeutic, healing-focused tone"""
        # Add gentle acknowledgment
        if not content.startswith(("Thank", "I", "Your")):
            therapeutic_opening = template["greeting_style"][0]
            content = f"{therapeutic_opening}. {content}"

        # Add validation language
        validation_phrases = template["validation_language"]
        if not any(phrase in content for phrase in validation_phrases):
            validation = validation_phrases[0]
            content = f"{validation}. {content}"

        # Add healing closing
        healing_closing = template["closing_style"][0]
        content = f"{content} {healing_closing}"

        return content

    def _apply_tone_adjustments(self, content: str, request: ReformulationRequest, analysis: Dict[str, Any]) -> Tuple[str, List[ToneAdjustment]]:
        """Apply specific tone adjustments based on content analysis"""
        adjusted_content = content
        applied_adjustments = []

        # Warmth enhancement
        if analysis["warmth_level"] == "cold":
            adjusted_content = self._apply_warmth_rules(adjusted_content)
            applied_adjustments.append(ToneAdjustment.COLD_TO_WARM)

        # Technical to friendly conversion
        if request.agent_type == "technical" and analysis["formality_level"] == "formal":
            adjusted_content = self._make_technical_friendly(adjusted_content)
            applied_adjustments.append(ToneAdjustment.TECHNICAL_TO_FRIENDLY)

        # Elaborate brief responses
        if len(content) < 50 and request.intent_detected != "casual_chat":
            adjusted_content = self._elaborate_response(adjusted_content, request)
            applied_adjustments.append(ToneAdjustment.BRIEF_TO_ELABORATE)

        return adjusted_content, applied_adjustments

    def _apply_warmth_rules(self, content: str) -> str:
        """Apply warmth enhancement rules"""
        warmth_rules = self.tone_rules["warmth_enhancement"]

        for cold_pattern, warm_replacement in zip(warmth_rules["cold_patterns"], warmth_rules["warm_replacements"]):
            if cold_pattern in content:
                content = content.replace(cold_pattern, warm_replacement)

        return content

    def _make_technical_friendly(self, content: str) -> str:
        """Make technical responses more friendly and approachable"""
        # Replace technical jargon with friendly explanations
        friendly_replacements = {
            "Error:": "Oops, looks like there's a small issue:",
            "Failed": "Didn't work quite as expected",
            "Incorrect": "Not quite right",
            "You must": "You might want to",
            "Required": "You'll need to"
        }

        for technical, friendly in friendly_replacements.items():
            content = content.replace(technical, friendly)

        return content

    def _elaborate_response(self, content: str, request: ReformulationRequest) -> str:
        """Elaborate on brief responses to provide more value"""
        if request.intent_detected == "question":
            elaboration = "\n\nLet me know if you'd like me to dive deeper into any aspect of this!"
        elif request.intent_detected == "technical_help":
            elaboration = "\n\nI'm happy to walk through this step-by-step or explain any part in more detail."
        else:
            elaboration = "\n\nFeel free to ask if you'd like to explore this further!"

        return content + elaboration

    def _add_personality_markers(self, content: str, personality: PersonalityProfile, request: ReformulationRequest) -> str:
        """Add personality-specific markers and indicators"""
        template = self.personality_templates[personality]

        # Add emotional indicators if appropriate
        if "emotional_indicators" in template and not any(indicator in content for indicator in template["emotional_indicators"]):
            # Add appropriate emoji/indicator
            indicator = template["emotional_indicators"][0]
            if personality == PersonalityProfile.WARM_COMPANION:
                content = f"{content} {indicator}"
            elif personality == PersonalityProfile.CREATIVE_MUSE:
                content = f"{indicator} {content}"

        return content

    def _calculate_reformulation_confidence(self, original: str, reformulated: str) -> float:
        """Calculate confidence in the reformulation quality"""
        # Base confidence
        confidence = 0.8

        # Adjust based on length difference (some expansion is good)
        length_ratio = len(reformulated) / len(original) if len(original) > 0 else 1.0
        if 1.1 <= length_ratio <= 1.5:  # 10-50% expansion is ideal
            confidence += 0.1
        elif length_ratio > 2.0:  # Too much expansion
            confidence -= 0.2

        # Check if key content was preserved
        original_words = set(original.lower().split())
        reformulated_words = set(reformulated.lower().split())
        preservation_ratio = len(original_words.intersection(reformulated_words)) / len(original_words) if len(original_words) > 0 else 1.0

        if preservation_ratio < 0.5:  # Too much content lost
            confidence -= 0.3
        elif preservation_ratio > 0.8:  # Good content preservation
            confidence += 0.1

        return min(1.0, max(0.1, confidence))

    def _extract_adjustments_made(self, original: str, final: str) -> List[str]:
        """Extract list of personality adjustments that were made"""
        adjustments = []

        # Check for length changes
        if len(final) > len(original) * 1.2:
            adjustments.append("Added personality-consistent elaboration")

        # Check for emoji additions
        original_emojis = sum(1 for char in original if ord(char) > 127)
        final_emojis = sum(1 for char in final if ord(char) > 127)
        if final_emojis > original_emojis:
            adjustments.append("Added emotional indicators")

        # Check for warmth enhancements
        warm_phrases = ["I understand", "I'm here", "together", "support"]
        if any(phrase in final.lower() and phrase not in original.lower() for phrase in warm_phrases):
            adjustments.append("Enhanced empathetic tone")

        # Check for structure improvements
        if ":" in final and ":" not in original:
            adjustments.append("Added structural clarity")

        return adjustments if adjustments else ["Applied personality consistency"]

    def _update_metrics(self, personality: PersonalityProfile, confidence: float, processing_time: float, success: bool):
        """Update formatting performance metrics"""
        self.metrics["total_reformulations"] += 1

        if success:
            self.metrics["successful_reformulations"] += 1

            # Update average confidence
            total = self.metrics["total_reformulations"]
            current_avg = self.metrics["average_confidence"]
            self.metrics["average_confidence"] = (
                (current_avg * (total - 1) + confidence) / total
            )

        # Update personality distribution
        personality_str = personality.value
        if personality_str not in self.metrics["personality_distribution"]:
            self.metrics["personality_distribution"][personality_str] = 0
        self.metrics["personality_distribution"][personality_str] += 1

    def get_formatting_analytics(self) -> Dict[str, Any]:
        """Get formatting analytics and performance metrics"""
        total = self.metrics["total_reformulations"]

        if total == 0:
            return {
                "total_reformulations": 0,
                "success_rate": 0.0,
                "average_confidence": 0.0,
                "personality_distribution": {},
                "tone_adjustments_made": {}
            }

        return {
            "total_reformulations": total,
            "success_rate": self.metrics["successful_reformulations"] / total,
            "average_confidence": self.metrics["average_confidence"],
            "personality_distribution": self.metrics["personality_distribution"],
            "tone_adjustments_made": self.metrics["tone_adjustments_made"]
        }


# Convenience function for easy usage
async def format_response(original_response: str, agent_type: str, intent: str, context: Optional[Dict[str, Any]] = None) -> ReformulationResponse:
    """
    Convenience function to format a single response

    Args:
        original_response: The original AI response
        agent_type: Type of agent that generated the response
        intent: Detected user intent
        context: Optional context dictionary

    Returns:
        ReformulationResponse: Formatted response with metadata
    """
    formatter = PersonalityFormatter()

    request = ReformulationRequest(
        original_response=original_response,
        agent_type=agent_type,
        intent_detected=intent,
        user_context=context or {},
        personality_context={}
    )

    return await formatter.format(request)


if __name__ == "__main__":
    # Example usage and testing
    async def demo():
        print("🎭 Personality Formatter Demo")
        print("=" * 50)

        formatter = PersonalityFormatter()

        # Test different responses and personalities
        test_cases = [
            {
                "response": "Here is the solution to your problem. Implement function sort_array(arr). Return sorted array.",
                "agent": "technical",
                "intent": "technical_help",
                "expected_personality": "friendly_expert"
            },
            {
                "response": "You are experiencing normal human emotions. This will pass.",
                "agent": "emotional",
                "intent": "emotional_support",
                "expected_personality": "therapeutic_guide"
            },
            {
                "response": "Write story about dragon. Include character development and plot.",
                "agent": "creative",
                "intent": "creative_request",
                "expected_personality": "creative_muse"
            }
        ]

        for i, case in enumerate(test_cases, 1):
            print(f"\n--- Test {i} ---")
            print(f"Original: {case['response']}")

            request = ReformulationRequest(
                original_response=case["response"],
                agent_type=case["agent"],
                intent_detected=case["intent"],
                user_context={},
                personality_context={}
            )

            response = await formatter.format(request)

            print(f"Personality: {response.metadata.get('target_personality', 'unknown')}")
            print(f"Confidence: {response.reformulation_confidence:.2f}")
            print(f"Adjustments: {', '.join(response.personality_adjustments)}")
            print(f"Reformulated: {response.content[:150]}...")

        # Show analytics
        print(f"\n📊 Formatting Analytics:")
        analytics = formatter.get_formatting_analytics()
        print(f"Total reformulations: {analytics['total_reformulations']}")
        print(f"Success rate: {analytics['success_rate']:.1%}")
        print(f"Average confidence: {analytics['average_confidence']:.2f}")

    # Run demo
    asyncio.run(demo())

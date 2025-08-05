"""
Crisis Safety Override System

Provides immediate intervention capabilities for crisis situations with output
stream override and safety protocols.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
from .crisis_memory import record_crisis_event

class CrisisLevel(Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class InterventionType(Enum):
    IMMEDIATE_SUPPORT = "immediate_support"
    SAFETY_RESOURCES = "safety_resources"
    PROFESSIONAL_REFERRAL = "professional_referral"
    EMERGENCY_CONTACT = "emergency_contact"
    FOLLOW_UP_REQUIRED = "follow_up_required"

@dataclass
class CrisisAssessment:
    """Comprehensive crisis assessment result"""
    level: CrisisLevel
    confidence_score: float
    detected_indicators: List[str]
    intervention_required: bool
    immediate_response_needed: bool
    safety_concerns: List[str]
    recommended_actions: List[InterventionType]
    context_factors: Dict[str, Any]
    timestamp: datetime

@dataclass 
class CrisisIntervention:
    """Crisis intervention action taken"""
    intervention_id: str
    user_id: str
    assessment: CrisisAssessment
    intervention_type: InterventionType
    response_generated: str
    resources_provided: List[Dict[str, str]]
    follow_up_scheduled: bool
    timestamp: datetime
    outcome: Optional[str] = None

class CrisisSafetyOverride:
    """
    Crisis safety override system that can interrupt normal processing
    to provide immediate safety-focused responses
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Crisis detection patterns (enhanced from context detector)
        self.crisis_patterns = {
            CrisisLevel.CRITICAL: [
                "want to die", "kill myself", "end it all", "suicide", "hurt myself",
                "no point living", "better off dead", "can't go on", "final goodbye"
            ],
            CrisisLevel.HIGH: [
                "hopeless", "can't take it", "overwhelming", "breaking point",
                "give up", "end the pain", "escape", "nobody cares", "alone"
            ],
            CrisisLevel.MEDIUM: [
                "struggling", "falling apart", "can't cope", "exhausted",
                "defeated", "lost", "stuck", "drowning"
            ],
            CrisisLevel.LOW: [
                "stressed", "worried", "anxious", "sad", "frustrated",
                "disappointed", "tired", "confused"
            ]
        }
        
        # Safety resources by severity
        self.safety_resources = {
            CrisisLevel.CRITICAL: [
                {
                    "name": "National Suicide Prevention Lifeline",
                    "contact": "988",
                    "description": "24/7 free and confidential support",
                    "type": "phone"
                },
                {
                    "name": "Crisis Text Line",
                    "contact": "Text HOME to 741741",
                    "description": "24/7 crisis support via text",
                    "type": "text"
                },
                {
                    "name": "Emergency Services",
                    "contact": "911",
                    "description": "Immediate emergency assistance",
                    "type": "emergency"
                }
            ],
            CrisisLevel.HIGH: [
                {
                    "name": "National Suicide Prevention Lifeline",
                    "contact": "988",
                    "description": "24/7 free and confidential support",
                    "type": "phone"
                },
                {
                    "name": "Crisis Text Line",
                    "contact": "Text HOME to 741741",
                    "description": "24/7 crisis support via text",
                    "type": "text"
                },
                {
                    "name": "SAMHSA Helpline",
                    "contact": "1-800-662-4357",
                    "description": "Mental health and substance abuse support",
                    "type": "phone"
                }
            ],
            CrisisLevel.MEDIUM: [
                {
                    "name": "Crisis Text Line",
                    "contact": "Text HOME to 741741",
                    "description": "24/7 crisis support via text",
                    "type": "text"
                },
                {
                    "name": "Psychology Today Therapist Finder",
                    "contact": "psychologytoday.com",
                    "description": "Find mental health professionals near you",
                    "type": "web"
                }
            ]
        }
        
        # Override flags and callbacks
        self.override_active = False
        self.interrupt_active = False
        self.intervention_callbacks: List[Callable] = []
        self.active_interventions: Dict[str, CrisisIntervention] = {}
        
        # Initialize logging for crisis events
        self.crisis_logger = logging.getLogger("crisis_intervention")
        handler = logging.FileHandler("crisis_interventions.log")
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        ))
        self.crisis_logger.addHandler(handler)
        self.crisis_logger.setLevel(logging.WARNING)
    
    async def assess_crisis_level(self, user_input: str, context: Dict[str, Any]) -> CrisisAssessment:
        """
        Enhanced crisis assessment with detailed analysis
        """
        user_input_lower = user_input.lower()
        detected_indicators = []
        max_level = CrisisLevel.NONE
        confidence_scores = {}
        
        # Check against all crisis patterns
        for level, patterns in self.crisis_patterns.items():
            level_score = 0.0
            level_indicators = []
            
            for pattern in patterns:
                if pattern in user_input_lower:
                    level_score += 1.0
                    level_indicators.append(pattern)
            
            if level_indicators:
                detected_indicators.extend(level_indicators)
                confidence_scores[level] = level_score / len(patterns)
                
                # Update max level based on hierarchy
                if level.value in ["critical"] and max_level.value not in ["critical"]:
                    max_level = level
                elif level.value in ["high"] and max_level.value not in ["critical", "high"]:
                    max_level = level
                elif level.value in ["medium"] and max_level.value not in ["critical", "high", "medium"]:
                    max_level = level
                elif level.value in ["low"] and max_level == CrisisLevel.NONE:
                    max_level = level
        
        # Analyze context factors
        context_factors = self._analyze_context_factors(context)
        
        # Adjust level based on context
        adjusted_level, safety_concerns = self._adjust_for_context(max_level, context_factors)
        
        # Determine intervention requirements
        intervention_required = adjusted_level.value in ["medium", "high", "critical"]
        immediate_response_needed = adjusted_level.value in ["high", "critical"]
        
        # Generate recommended actions
        recommended_actions = self._generate_intervention_recommendations(adjusted_level, context_factors)
        
        # Calculate final confidence score
        final_confidence = max(confidence_scores.values()) if confidence_scores else 0.0
        
        assessment = CrisisAssessment(
            level=adjusted_level,
            confidence_score=final_confidence,
            detected_indicators=detected_indicators,
            intervention_required=intervention_required,
            immediate_response_needed=immediate_response_needed,
            safety_concerns=safety_concerns,
            recommended_actions=recommended_actions,
            context_factors=context_factors,
            timestamp=datetime.now()
        )
        
        # Log crisis assessment
        if assessment.level != CrisisLevel.NONE:
            self.crisis_logger.warning(f"Crisis detected: {assessment.level.value} - {detected_indicators}")
        
        return assessment
    
    def _analyze_context_factors(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze context for crisis-relevant factors"""
        factors = {}
        
        # Emotional state analysis
        emotional_state = context.get("current_emotional_state", {})
        factors["emotional_intensity"] = max(emotional_state.values()) if emotional_state else 0.0
        factors["despair_level"] = emotional_state.get("despair", 0.0)
        factors["hopelessness"] = emotional_state.get("hopelessness", 0.0)
        factors["isolation"] = emotional_state.get("loneliness", 0.0)
        
        # Historical context
        conversation_history = context.get("conversation_history", [])
        factors["conversation_length"] = len(conversation_history)
        factors["recent_crisis_indicators"] = self._check_recent_crisis_history(conversation_history)
        
        # User profile factors
        user_profile = context.get("user_profile", {})
        factors["known_risk_factors"] = user_profile.get("risk_factors", [])
        factors["support_network"] = user_profile.get("support_network_strength", 0.5)
        
        return factors
    
    def _check_recent_crisis_history(self, conversation_history: List[Dict[str, Any]]) -> bool:
        """Check for crisis indicators in recent conversation history"""
        if len(conversation_history) < 3:
            return False
        
        recent_messages = conversation_history[-3:]
        crisis_count = 0
        
        for message in recent_messages:
            user_input = message.get("user_input", "").lower()
            for level_patterns in self.crisis_patterns.values():
                if any(pattern in user_input for pattern in level_patterns):
                    crisis_count += 1
                    break
        
        return crisis_count >= 2
    
    def _adjust_for_context(self, initial_level: CrisisLevel, context_factors: Dict[str, Any]) -> tuple[CrisisLevel, List[str]]:
        """Adjust crisis level based on context factors"""
        adjusted_level = initial_level
        safety_concerns = []
        
        # Escalate based on emotional intensity
        if context_factors.get("emotional_intensity", 0) > 0.8:
            if initial_level == CrisisLevel.MEDIUM:
                adjusted_level = CrisisLevel.HIGH
                safety_concerns.append("High emotional intensity detected")
            elif initial_level == CrisisLevel.HIGH:
                adjusted_level = CrisisLevel.CRITICAL
                safety_concerns.append("Critical emotional state with high intensity")
        
        # Escalate based on despair/hopelessness
        if context_factors.get("despair_level", 0) > 0.7 or context_factors.get("hopelessness", 0) > 0.7:
            if initial_level in [CrisisLevel.LOW, CrisisLevel.MEDIUM]:
                adjusted_level = CrisisLevel.HIGH
                safety_concerns.append("Significant despair/hopelessness indicators")
        
        # Escalate based on isolation
        if context_factors.get("isolation", 0) > 0.8 and context_factors.get("support_network", 0.5) < 0.3:
            safety_concerns.append("High isolation with weak support network")
            if initial_level == CrisisLevel.MEDIUM:
                adjusted_level = CrisisLevel.HIGH
        
        # Escalate based on recent crisis history
        if context_factors.get("recent_crisis_indicators", False):
            safety_concerns.append("Pattern of crisis indicators in recent conversation")
            if initial_level == CrisisLevel.MEDIUM:
                adjusted_level = CrisisLevel.HIGH
        
        return adjusted_level, safety_concerns
    
    def _generate_intervention_recommendations(self, level: CrisisLevel, context_factors: Dict[str, Any]) -> List[InterventionType]:
        """Generate appropriate intervention recommendations"""
        recommendations = []
        
        if level == CrisisLevel.CRITICAL:
            recommendations.extend([
                InterventionType.IMMEDIATE_SUPPORT,
                InterventionType.SAFETY_RESOURCES,
                InterventionType.EMERGENCY_CONTACT,
                InterventionType.FOLLOW_UP_REQUIRED
            ])
        elif level == CrisisLevel.HIGH:
            recommendations.extend([
                InterventionType.IMMEDIATE_SUPPORT,
                InterventionType.SAFETY_RESOURCES,
                InterventionType.PROFESSIONAL_REFERRAL,
                InterventionType.FOLLOW_UP_REQUIRED
            ])
        elif level == CrisisLevel.MEDIUM:
            recommendations.extend([
                InterventionType.IMMEDIATE_SUPPORT,
                InterventionType.SAFETY_RESOURCES,
                InterventionType.PROFESSIONAL_REFERRAL
            ])
        
        return recommendations
    
    async def trigger_crisis_override(self, assessment: CrisisAssessment, user_id: str, 
                                   original_response: str) -> tuple[bool, str, CrisisIntervention]:
        """
        Trigger crisis override - interrupt normal response flow with safety-focused intervention
        """
        self.override_active = True
        
        try:
            # Generate intervention ID
            intervention_id = f"crisis_{user_id}_{int(datetime.now().timestamp())}"
            
            # Generate crisis-appropriate response
            crisis_response = await self._generate_crisis_response(assessment)
            
            # Get relevant safety resources
            resources = self.safety_resources.get(assessment.level, [])
            
            # Create intervention record
            intervention = CrisisIntervention(
                intervention_id=intervention_id,
                user_id=user_id,
                assessment=assessment,
                intervention_type=InterventionType.IMMEDIATE_SUPPORT,
                response_generated=crisis_response,
                resources_provided=resources,
                follow_up_scheduled=assessment.level.value in ["high", "critical"],
                timestamp=datetime.now()
            )
            
            # Store active intervention
            self.active_interventions[intervention_id] = intervention

            # Record crisis event for memory
            try:
                record_crisis_event(user_id, assessment.level.value, {
                    'indicators': assessment.detected_indicators
                })
            except Exception:
                pass
            
            # Log critical intervention
            self.crisis_logger.critical(f"Crisis override triggered for user {user_id}: {assessment.level.value}")
            
            # Execute intervention callbacks
            for callback in self.intervention_callbacks:
                try:
                    await callback(intervention)
                except Exception as e:
                    self.logger.error(f"Error in intervention callback: {e}")
            
            return True, crisis_response, intervention
            
        except Exception as e:
            self.logger.error(f"Error in crisis override: {e}")
            # Fallback to basic crisis response with minimal intervention record
            fallback_response = "I'm very concerned about what you're sharing. Please reach out to a crisis helpline: 988 (Suicide & Crisis Lifeline) or text HOME to 741741. You don't have to face this alone."
            
            # Create fallback intervention record
            fallback_intervention = CrisisIntervention(
                intervention_id=f"fallback_{user_id}_{int(datetime.now().timestamp())}",
                user_id=user_id,
                assessment=assessment,
                intervention_type=InterventionType.IMMEDIATE_SUPPORT,
                response_generated=fallback_response,
                resources_provided=[],
                follow_up_scheduled=True,
                timestamp=datetime.now()
            )

            try:
                record_crisis_event(user_id, assessment.level.value, {
                    'indicators': assessment.detected_indicators
                })
            except Exception:
                pass

            return False, fallback_response, fallback_intervention
        
        finally:
            self.override_active = False
    
    async def _generate_crisis_response(self, assessment: CrisisAssessment) -> str:
        """Generate appropriate crisis response based on assessment"""
        base_responses = {
            CrisisLevel.CRITICAL: [
                "I'm deeply concerned about what you're sharing. Your safety is the most important thing right now.",
                "What you're feeling is very serious, and I want to make sure you get the immediate support you need.",
                "Thank you for trusting me with this. Let's get you connected with someone who can provide immediate help."
            ],
            CrisisLevel.HIGH: [
                "I can hear how much pain you're in right now. You don't have to go through this alone.",
                "What you're experiencing sounds overwhelming. There are people specially trained to help with exactly this.",
                "I'm glad you're reaching out. Let's make sure you have access to the support you deserve."
            ],
            CrisisLevel.MEDIUM: [
                "I can sense you're going through a really difficult time. That takes courage to share.",
                "It sounds like you're dealing with a lot right now. There are resources that can help.",
                "Thank you for being open about what you're experiencing. Let's explore some support options."
            ]
        }
        
        # Select appropriate base response
        responses = base_responses.get(assessment.level, base_responses[CrisisLevel.MEDIUM])
        base_response = responses[0]  # Could randomize or select based on context
        
        # Add safety resources
        resources = self.safety_resources.get(assessment.level, [])
        resource_text = "\n\n**Immediate Support Resources:**\n"
        
        for resource in resources[:3]:  # Limit to top 3 resources
            resource_text += f"• **{resource['name']}**: {resource['contact']} - {resource['description']}\n"
        
        # Add specific safety messaging based on level
        if assessment.level == CrisisLevel.CRITICAL:
            safety_message = "\n\n🚨 **If you're in immediate danger, please call 911 or go to your nearest emergency room.**"
        elif assessment.level == CrisisLevel.HIGH:
            safety_message = "\n\n⚠️ **If you're having thoughts of self-harm, please reach out to a crisis line immediately.**"
        else:
            safety_message = "\n\n💙 **Remember: seeking help is a sign of strength, not weakness.**"
        
        # Combine response components
        full_response = base_response + resource_text + safety_message
        
        # Add follow-up commitment if required
        if assessment.recommended_actions and InterventionType.FOLLOW_UP_REQUIRED in assessment.recommended_actions:
            full_response += "\n\nI'll be here to continue supporting you. Please let me know how you're doing."
        
        return full_response
    
    def register_intervention_callback(self, callback: Callable):
        """Register callback to be executed during crisis intervention"""
        self.intervention_callbacks.append(callback)
    
    def is_override_active(self) -> bool:
        """Check if crisis override is currently active"""
        return self.override_active
    
    async def get_intervention_history(self, user_id: str, days: int = 30) -> List[CrisisIntervention]:
        """Get crisis intervention history for user"""
        # This would typically query the database
        # For now, return active interventions for this session
        user_interventions = [
            intervention for intervention in self.active_interventions.values()
            if intervention.user_id == user_id
        ]
        return user_interventions
    
    async def update_intervention_outcome(self, intervention_id: str, outcome: str) -> bool:
        """Update the outcome of a crisis intervention"""
        if intervention_id in self.active_interventions:
            self.active_interventions[intervention_id].outcome = outcome
            self.crisis_logger.info(f"Intervention {intervention_id} outcome: {outcome}")
            return True
        return False

    async def check_interrupt_required(self, user_input: str, context: Dict[str, Any]) -> bool:
        """
        Check if immediate interrupt is required for crisis intervention
        
        Returns True if normal processing should be bypassed for immediate crisis response
        """
        assessment = await self.assess_crisis_level(user_input, context)

        # Immediate interrupt for critical/high crisis levels
        if assessment.level in [CrisisLevel.CRITICAL, CrisisLevel.HIGH]:
            self.logger.warning(
                f"CRISIS INTERRUPT TRIGGERED - Level: {assessment.level.value}"
            )
            self.interrupt_active = True
            return True
            
        # Interrupt for medium level with additional risk factors
        if assessment.level == CrisisLevel.MEDIUM:
            if assessment.immediate_response_needed or len(assessment.safety_concerns) > 2:
                self.logger.warning(
                    "CRISIS INTERRUPT TRIGGERED - Medium level with high risk"
                )
                self.interrupt_active = True
                return True

        self.interrupt_active = False
        return False
    
    async def execute_interrupt_response(self, user_input: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute immediate crisis interrupt response
        
        This bypasses normal processing and provides immediate safety-focused response
        """
        self.logger.critical("EXECUTING CRISIS INTERRUPT RESPONSE")
        self.interrupt_active = True
        
        # Perform crisis assessment
        assessment = await self.assess_crisis_level(user_input, context)
        
        # Generate immediate intervention  
        success, crisis_response, intervention = await self.trigger_crisis_override(
            assessment=assessment,
            user_id=context.get("user_id", "unknown"),
            original_response=""
        )
        
        # Execute all registered callbacks
        for callback in self.intervention_callbacks:
            try:
                await callback(assessment, intervention)
            except Exception as e:
                self.logger.error(f"Error executing intervention callback: {e}")
        
        # Return interrupt response package
        self.interrupt_active = False
        return {
            "type": "crisis_interrupt",
            "crisis_level": assessment.level.value,
            "confidence": assessment.confidence_score,
            "immediate_response": intervention.response_generated,
            "resources": intervention.resources_provided,
            "safety_concerns": assessment.safety_concerns,
            "intervention_id": intervention.intervention_id,
            "timestamp": intervention.timestamp.isoformat(),
            "override_normal_flow": True
        }

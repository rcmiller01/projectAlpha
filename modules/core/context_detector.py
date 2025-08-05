"""
Context Detector

Analyzes user input to determine interaction characteristics and adaptive focus
without requiring manual mode switching from the user.
"""

import re
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import asyncio

class ContextDetector:
    """
    Detects interaction context and characteristics for adaptive companion responses
    """
    
    def __init__(self):
        self.emotional_indicators = self._initialize_emotional_indicators()
        self.technical_indicators = self._initialize_technical_indicators()
        self.creative_indicators = self._initialize_creative_indicators()
        self.crisis_indicators = self._initialize_crisis_indicators()
        
    def _initialize_emotional_indicators(self) -> Dict[str, List[str]]:
        """Initialize emotional context indicators"""
        return {
            "high_emotional_need": [
                "feeling", "feel", "heart", "love", "afraid", "scared", "lonely", "overwhelmed",
                "anxious", "depressed", "sad", "hurt", "pain", "crying", "tears", "emotional",
                "vulnerable", "need you", "help me", "support", "comfort", "understand"
            ],
            "relationship_focused": [
                "relationship", "together", "close", "intimate", "connect", "connection",
                "bond", "trust", "care", "caring", "love", "affection", "attachment"
            ],
            "stress_indicators": [
                "stress", "stressed", "pressure", "overwhelmed", "burned out", "exhausted",
                "can't handle", "too much", "breaking down", "falling apart"
            ],
            "positive_emotional": [
                "happy", "joy", "excited", "grateful", "thankful", "proud", "accomplished",
                "celebrating", "amazing", "wonderful", "beautiful", "love this"
            ]
        }
    
    def _initialize_technical_indicators(self) -> Dict[str, List[str]]:
        """Initialize technical context indicators"""
        return {
            "code_related": [
                "code", "programming", "function", "variable", "class", "method", "API",
                "database", "sql", "python", "javascript", "html", "css", "react", "node",
                "debug", "error", "bug", "syntax", "compile", "deploy", "git", "github"
            ],
            "problem_solving": [
                "problem", "issue", "error", "bug", "not working", "broken", "fix", "solve",
                "debug", "troubleshoot", "help with", "figure out", "stuck on"
            ],
            "learning_technical": [
                "learn", "tutorial", "how to", "explain", "understand", "concept",
                "documentation", "example", "practice", "study", "course"
            ],
            "project_management": [
                "project", "deadline", "timeline", "planning", "organize", "structure",
                "architecture", "design", "requirements", "specifications"
            ]
        }
    
    def _initialize_creative_indicators(self) -> Dict[str, List[str]]:
        """Initialize creative context indicators"""
        return {
            "artistic_creation": [
                "write", "writing", "create", "creative", "art", "artistic", "poem", "poetry",
                "story", "novel", "song", "music", "paint", "draw", "design", "craft"
            ],
            "inspiration_seeking": [
                "inspiration", "inspire", "ideas", "brainstorm", "imagine", "dream",
                "vision", "possibility", "potential", "explore", "experiment"
            ],
            "creative_blocks": [
                "blocked", "stuck", "can't create", "no ideas", "uninspired",
                "creative block", "writer's block", "empty", "blank"
            ],
            "collaboration": [
                "collaborate", "together", "co-create", "partner", "work with",
                "help me create", "create together", "artistic partner"
            ]
        }
    
    def _initialize_crisis_indicators(self) -> Dict[str, List[str]]:
        """Initialize crisis-level indicators requiring immediate attention"""
        return {
            "severe_distress": [
                "can't go on", "want to die", "kill myself", "end it all", "suicide",
                "self-harm", "hurt myself", "not worth living", "hopeless", "no point"
            ],
            "panic_anxiety": [
                "panic attack", "can't breathe", "heart racing", "going crazy",
                "losing control", "feel like dying", "terror", "overwhelming fear"
            ],
            "dissociation": [
                "not real", "floating", "outside myself", "watching myself",
                "disconnected", "numb", "empty", "void", "nothing matters"
            ]
        }
    
    async def analyze_interaction(self, user_input: str, context: Dict[str, Any], 
                                conversation_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze user interaction to determine characteristics and adaptive focus
        """
        
        analysis = {
            "primary_focus": "general",
            "emotional_priority": "medium",
            "technical_priority": "low",
            "creative_priority": "low",
            "crisis_level": "none",
            "detected_needs": [],
            "conversation_flow": "natural",
            "emotional_trajectory": "stable",
            "technical_context": "none",
            "adaptive_recommendations": {}
        }
        
        # Crisis assessment first (highest priority)
        crisis_analysis = await self._assess_crisis_level(user_input, context)
        if crisis_analysis["level"] != "none":
            analysis.update(crisis_analysis)
            analysis["primary_focus"] = "crisis_support"
            return analysis
        
        # Analyze content for different dimensions
        emotional_analysis = self._analyze_emotional_content(user_input, context)
        technical_analysis = self._analyze_technical_content(user_input, context)
        creative_analysis = self._analyze_creative_content(user_input, context)
        
        # Determine priority levels
        analysis["emotional_priority"] = emotional_analysis["priority"]
        analysis["technical_priority"] = technical_analysis["priority"]
        analysis["creative_priority"] = creative_analysis["priority"]
        
        # Determine primary focus based on highest priority and context
        analysis["primary_focus"] = self._determine_primary_focus(
            emotional_analysis, technical_analysis, creative_analysis
        )
        
        # Detect specific needs
        analysis["detected_needs"] = self._detect_specific_needs(
            user_input, emotional_analysis, technical_analysis, creative_analysis
        )
        
        # Analyze conversation flow and trajectory
        if conversation_history:
            flow_analysis = self._analyze_conversation_flow(user_input, conversation_history)
            analysis.update(flow_analysis)
        
        # Generate adaptive recommendations
        analysis["adaptive_recommendations"] = self._generate_adaptive_recommendations(
            analysis, emotional_analysis, technical_analysis, creative_analysis
        )
        
        return analysis
    
    async def _assess_crisis_level(self, user_input: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Assess if this is a crisis-level interaction requiring immediate support"""
        
        user_input_lower = user_input.lower()
        crisis_score = 0.0
        detected_crisis_types = []
        
        for crisis_type, indicators in self.crisis_indicators.items():
            for indicator in indicators:
                if indicator in user_input_lower:
                    crisis_score += 0.8 if crisis_type == "severe_distress" else 0.6
                    detected_crisis_types.append(crisis_type)
        
        # Check emotional state from context
        emotional_state = context.get("current_emotional_state", {})
        if emotional_state.get("despair", 0) > 0.8 or emotional_state.get("hopelessness", 0) > 0.7:
            crisis_score += 0.5
        
        # Determine crisis level
        if crisis_score >= 0.8:
            level = "high"
        elif crisis_score >= 0.5:
            level = "medium"
        elif crisis_score > 0:
            level = "low"
        else:
            level = "none"
        
        return {
            "level": level,
            "score": crisis_score,
            "detected_types": detected_crisis_types,
            "immediate_support_needed": level in ["high", "medium"]
        }
    
    def _analyze_emotional_content(self, user_input: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze emotional content and needs"""
        
        user_input_lower = user_input.lower()
        emotional_score = 0.0
        detected_emotions = []
        
        for emotion_category, indicators in self.emotional_indicators.items():
            category_score = 0.0
            for indicator in indicators:
                if indicator in user_input_lower:
                    category_score += 1.0
            
            if category_score > 0:
                emotional_score += category_score
                detected_emotions.append(emotion_category)
        
        # Determine priority
        if emotional_score >= 3.0:
            priority = "high"
        elif emotional_score >= 1.0:
            priority = "medium"
        else:
            priority = "low"
        
        return {
            "priority": priority,
            "score": emotional_score,
            "detected_categories": detected_emotions,
            "needs_emotional_support": priority in ["high", "medium"],
            "relationship_focus": "relationship_focused" in detected_emotions
        }
    
    def _analyze_technical_content(self, user_input: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze technical content and assistance needs"""
        
        user_input_lower = user_input.lower()
        technical_score = 0.0
        detected_categories = []
        
        for tech_category, indicators in self.technical_indicators.items():
            category_score = 0.0
            for indicator in indicators:
                if indicator in user_input_lower:
                    category_score += 1.0
            
            if category_score > 0:
                technical_score += category_score
                detected_categories.append(tech_category)
        
        # Check for code snippets or technical patterns
        if re.search(r'```|`.*`|\b\w+\(\)|def \w+|class \w+|import \w+', user_input):
            technical_score += 2.0
            detected_categories.append("code_snippet_present")
        
        # Determine priority
        if technical_score >= 3.0:
            priority = "high"
        elif technical_score >= 1.0:
            priority = "medium"
        else:
            priority = "low"
        
        return {
            "priority": priority,
            "score": technical_score,
            "detected_categories": detected_categories,
            "needs_technical_help": priority in ["high", "medium"],
            "is_coding_session": "code_related" in detected_categories or "code_snippet_present" in detected_categories
        }
    
    def _analyze_creative_content(self, user_input: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze creative content and artistic needs"""
        
        user_input_lower = user_input.lower()
        creative_score = 0.0
        detected_categories = []
        
        for creative_category, indicators in self.creative_indicators.items():
            category_score = 0.0
            for indicator in indicators:
                if indicator in user_input_lower:
                    category_score += 1.0
            
            if category_score > 0:
                creative_score += category_score
                detected_categories.append(creative_category)
        
        # Determine priority
        if creative_score >= 2.0:
            priority = "high"
        elif creative_score >= 1.0:
            priority = "medium"
        else:
            priority = "low"
        
        return {
            "priority": priority,
            "score": creative_score,
            "detected_categories": detected_categories,
            "needs_creative_support": priority in ["high", "medium"],
            "has_creative_block": "creative_blocks" in detected_categories,
            "wants_collaboration": "collaboration" in detected_categories
        }
    
    def _determine_primary_focus(self, emotional_analysis: Dict, technical_analysis: Dict, 
                                creative_analysis: Dict) -> str:
        """Determine the primary focus for this interaction"""
        
        priorities = {
            "emotional": emotional_analysis["score"],
            "technical": technical_analysis["score"],
            "creative": creative_analysis["score"]
        }
        
        # Check for integrated needs (multiple high priorities)
        high_priority_count = sum(1 for analysis in [emotional_analysis, technical_analysis, creative_analysis]
                                if analysis["priority"] == "high")
        
        if high_priority_count > 1:
            return "integrated_support"
        
        # Determine single primary focus
        max_priority = max(priorities.items(), key=lambda x: x[1])
        
        if max_priority[1] == 0:
            return "general_conversation"
        
        if max_priority[0] == "emotional":
            return "emotional_support"
        elif max_priority[0] == "technical":
            return "technical_assistance"
        elif max_priority[0] == "creative":
            return "creative_collaboration"
        else:
            return "general_conversation"
    
    def _detect_specific_needs(self, user_input: str, emotional_analysis: Dict,
                             technical_analysis: Dict, creative_analysis: Dict) -> List[str]:
        """Detect specific user needs for this interaction"""
        
        needs = []
        
        # Emotional needs
        if emotional_analysis["needs_emotional_support"]:
            if "stress_indicators" in emotional_analysis["detected_categories"]:
                needs.append("stress_management")
            if "relationship_focused" in emotional_analysis["detected_categories"]:
                needs.append("relationship_support")
            if "positive_emotional" in emotional_analysis["detected_categories"]:
                needs.append("celebration_sharing")
            else:
                needs.append("emotional_validation")
        
        # Technical needs
        if technical_analysis["needs_technical_help"]:
            if "problem_solving" in technical_analysis["detected_categories"]:
                needs.append("problem_solving_assistance")
            if "learning_technical" in technical_analysis["detected_categories"]:
                needs.append("technical_education")
            if "project_management" in technical_analysis["detected_categories"]:
                needs.append("project_guidance")
            else:
                needs.append("technical_support")
        
        # Creative needs
        if creative_analysis["needs_creative_support"]:
            if creative_analysis["has_creative_block"]:
                needs.append("creative_block_resolution")
            if creative_analysis["wants_collaboration"]:
                needs.append("creative_collaboration")
            if "inspiration_seeking" in creative_analysis["detected_categories"]:
                needs.append("creative_inspiration")
            else:
                needs.append("artistic_expression_support")
        
        return needs
    
    def _analyze_conversation_flow(self, user_input: str, conversation_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze conversation flow and emotional trajectory"""
        
        if not conversation_history:
            return {
                "conversation_flow": "new_conversation",
                "emotional_trajectory": "establishing"
            }
        
        recent_interactions = conversation_history[-3:]
        
        # Analyze emotional trajectory
        emotional_priorities = [interaction.get("emotional_priority", "medium") 
                              for interaction in recent_interactions]
        
        if len(set(emotional_priorities)) == 1 and emotional_priorities[0] == "high":
            emotional_trajectory = "consistently_high_emotional_need"
        elif emotional_priorities[-1] == "high" and emotional_priorities[0] != "high":
            emotional_trajectory = "escalating_emotional_need"
        elif emotional_priorities[0] == "high" and emotional_priorities[-1] != "high":
            emotional_trajectory = "de_escalating_emotional_need"
        else:
            emotional_trajectory = "stable"
        
        # Analyze conversation flow
        primary_focuses = [interaction.get("primary_focus", "general") 
                         for interaction in recent_interactions]
        
        if len(set(primary_focuses)) == 1:
            conversation_flow = f"sustained_{primary_focuses[0]}"
        elif len(set(primary_focuses)) == len(primary_focuses):
            conversation_flow = "varied_topics"
        else:
            conversation_flow = "mixed_engagement"
        
        return {
            "conversation_flow": conversation_flow,
            "emotional_trajectory": emotional_trajectory,
            "recent_focus_pattern": primary_focuses
        }
    
    def _generate_adaptive_recommendations(self, analysis: Dict, emotional_analysis: Dict,
                                         technical_analysis: Dict, creative_analysis: Dict) -> Dict[str, Any]:
        """Generate recommendations for adaptive response approach"""
        
        recommendations = {
            "response_tone": "warm_and_supportive",
            "interaction_style": "natural_and_caring",
            "priority_balance": {},
            "special_considerations": []
        }
        
        # Determine response tone based on primary focus
        if analysis["primary_focus"] == "emotional_support":
            recommendations["response_tone"] = "gentle_and_empathetic"
            recommendations["interaction_style"] = "intimate_and_caring"
        elif analysis["primary_focus"] == "technical_assistance":
            recommendations["response_tone"] = "supportive_and_competent"
            recommendations["interaction_style"] = "helpful_and_encouraging"
        elif analysis["primary_focus"] == "creative_collaboration":
            recommendations["response_tone"] = "inspiring_and_encouraging"
            recommendations["interaction_style"] = "collaborative_and_artistic"
        elif analysis["primary_focus"] == "integrated_support":
            recommendations["response_tone"] = "balanced_and_comprehensive"
            recommendations["interaction_style"] = "holistically_supportive"
        
        # Priority balance recommendations
        recommendations["priority_balance"] = {
            "emotional": emotional_analysis["priority"],
            "technical": technical_analysis["priority"],
            "creative": creative_analysis["priority"]
        }
        
        # Special considerations
        if analysis["emotional_trajectory"] == "escalating_emotional_need":
            recommendations["special_considerations"].append("monitor_emotional_escalation")
        
        if emotional_analysis.get("relationship_focus"):
            recommendations["special_considerations"].append("emphasize_relationship_connection")
        
        if technical_analysis.get("is_coding_session"):
            recommendations["special_considerations"].append("provide_technical_precision")
        
        if creative_analysis.get("has_creative_block"):
            recommendations["special_considerations"].append("gentle_creative_encouragement")
        
        return recommendations

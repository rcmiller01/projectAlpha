    # Enhanced Methods for Missing Functionality

    async def _log_crisis_intervention(self, user_id: str, intervention: Any):
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
                self.enhanced_logger.warning("Database does not support crisis logging")
                
        except Exception as e:
            self.enhanced_logger.error(f"Error logging crisis intervention: {e}")

    async def _create_new_user_profile(self, user_id: str) -> Any:
        """Create new user profile"""
        try:
            from ..database.database_interface import UserProfile
            
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

    async def _retrieve_relevant_memories(self, user_id: str, user_input: str, interaction_state: Any) -> List[Any]:
        """Retrieve relevant memories from database"""
        try:
            # Get memories based on current context
            memories = await self.database.get_relevant_memories(
                user_id=user_id,
                memory_type=None,  # Get all types
                tags=None,
                limit=5
            )
            
            return memories
            
        except Exception as e:
            self.enhanced_logger.error(f"Error retrieving memories: {e}")
            return []

    async def _build_enhanced_context(self, interaction_state: Any, user_profile: Any, 
                                   relevant_memories: List[Any], crisis_assessment: Any) -> Dict[str, Any]:
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

    def _get_base_response_patterns(self) -> Dict[str, List[str]]:
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

    async def _generate_enhanced_companion_response(self, user_input: str, analysis_context: Dict[str, Any], 
                                                  guidance_package: Any, interaction_state: Any,
                                                  optimal_templates: Dict[str, str], 
                                                  relevant_memories: List[Any]) -> str:
        """Generate enhanced companion response with all context"""
        
        # Use existing response generation as base
        base_response = await self._generate_companion_response(
            user_input, analysis_context, guidance_package.mode_specifics, interaction_state
        )
        
        # Enhance with memory context if relevant
        if relevant_memories:
            memory_context = "I remember you mentioning similar themes before, which helps me understand the deeper context of what you're sharing."
            base_response = f"{memory_context}\n\n{base_response}"
        
        # Add emotional pattern awareness
        emotional_patterns = analysis_context.get("emotional_patterns", {})
        if emotional_patterns.get("dominant_emotions"):
            dominant_emotion = emotional_patterns["dominant_emotions"][0][0]
            # Subtly acknowledge emotional patterns without being too clinical
            if dominant_emotion in ["anxiety", "stress", "worry"]:
                base_response = base_response.replace("I understand", "I understand, and I notice this connects to feelings you've been working through")
        
        return base_response

    async def _estimate_response_quality(self, response: str, context: Dict[str, Any]) -> float:
        """Estimate response quality for template effectiveness tracking"""
        
        quality_score = 0.5  # Base score
        
        # Length appropriateness
        if 50 <= len(response) <= 500:
            quality_score += 0.1
        
        # Emotional alignment
        emotional_state = context.get("current_emotional_state", {})
        if emotional_state:
            max_emotion = max(emotional_state.values()) if emotional_state else 0
            if max_emotion > 0.7:
                # Check if response contains appropriate emotional language
                empathetic_words = ["understand", "feel", "support", "here", "together", "care"]
                if any(word in response.lower() for word in empathetic_words):
                    quality_score += 0.2
        
        # Crisis appropriateness
        crisis_level = context.get("crisis_assessment", {}).get("level", "none")
        if crisis_level in ["high", "critical"] and "help" in response.lower():
            quality_score += 0.2
        
        return min(quality_score, 1.0)

    async def _store_interaction_data(self, interaction_id: str, user_id: str, user_input: str, 
                                    companion_response: str, context_analysis: Dict[str, Any], 
                                    guidance_package: Any, interaction_state: Any):
        """Store comprehensive interaction data"""
        try:
            from ..database.database_interface import InteractionRecord, InteractionType
            
            # Determine interaction type
            interaction_type = InteractionType.PERSONAL_SUPPORT
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
            
            # Store memory fragments for important interactions
            if context_analysis.get("emotional_priority") == "high" or \
               context_analysis.get("crisis_assessment", {}).get("level") in ["medium", "high", "critical"]:
                await self._store_memory_fragment(user_id, user_input, companion_response, context_analysis)
                
        except Exception as e:
            self.enhanced_logger.error(f"Error storing interaction data: {e}")

    async def _store_memory_fragment(self, user_id: str, user_input: str, companion_response: str, 
                                   context_analysis: Dict[str, Any]):
        """Store important interaction as memory fragment"""
        try:
            from ..database.database_interface import MemoryFragment
            
            # Determine memory type
            memory_type = "emotional"
            if context_analysis.get("technical_priority") == "high":
                memory_type = "technical"
            elif context_analysis.get("creative_priority") == "high":
                memory_type = "creative"
            
            # Calculate importance score
            importance_score = 0.5
            if context_analysis.get("emotional_priority") == "high":
                importance_score += 0.3
            if context_analysis.get("crisis_assessment", {}).get("level") in ["medium", "high", "critical"]:
                importance_score += 0.4
            
            # Generate tags
            tags = []
            if context_analysis.get("primary_focus"):
                tags.append(context_analysis["primary_focus"])
            
            emotional_state = context_analysis.get("current_emotional_state", {})
            for emotion, intensity in emotional_state.items():
                if intensity > 0.6:
                    tags.append(emotion)

            tone = context_analysis.get("detected_tone")
            sentiment = context_analysis.get("sentiment_score", 0.0)
            hour = datetime.now().hour
            if hour < 6:
                time_of_day = "night"
            elif hour < 12:
                time_of_day = "morning"
            elif hour < 18:
                time_of_day = "afternoon"
            else:
                time_of_day = "evening"

            memory_fragment = MemoryFragment(
                memory_id=str(uuid.uuid4()),
                user_id=user_id,
                content=f"User: {user_input[:200]}... | Response: {companion_response[:200]}...",
                memory_type=memory_type,
                importance_score=min(importance_score, 1.0),
                created_at=datetime.now(),
                last_accessed=datetime.now(),
                access_count=1,
                related_interactions=[],
                tags=tags,
                tone=tone,
                sentiment=sentiment,
                time_of_day=time_of_day
            )
            
            await self.database.save_memory_fragment(memory_fragment)
            
        except Exception as e:
            self.enhanced_logger.error(f"Error storing memory fragment: {e}")

    async def _generate_enhanced_fallback_response(self, user_input: str, error_msg: str, user_id: str) -> Dict[str, Any]:
        """Generate enhanced fallback response with user context"""
        
        # Try to get some user context even during errors
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

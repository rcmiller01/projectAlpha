"""
Investment Collaboration Integration

Bridges the existing investment tracker with the new collaborative goal achievement engine.
Creates seamless AI partnership for investment goals.
"""

import json
import time
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from modules.collaboration.goal_achievement import collaborative_engine, CollaborativeGoalEngine
from modules.finance import get_investment_integration, StrategyType, GoalType

class InvestmentCollaborationBridge:
    """
    Bridges investment tracking with AI collaborative partnership
    """
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.investment_integration = get_investment_integration(data_dir)
        self.collaborative_engine = collaborative_engine
        
        # Enhanced goal creation with AI partnership
        self.enhanced_goals_file = f"{data_dir}/enhanced_investment_goals.json"
        self.partnership_interactions = f"{data_dir}/partnership_interactions.json"
        
    def create_collaborative_investment_goal(self, user_id: str, goal_data: Dict) -> Dict:
        """Create an investment goal with full AI collaboration"""
        
        # Create the base goal using existing system
        base_response = self.investment_integration.create_investment_goal_with_companion(
            name=goal_data['name'],
            target_amount=goal_data['target_amount'],
            goal_type=GoalType(goal_data.get('goal_type', 'WEALTH_BUILDING')),
            description=goal_data.get('description', ''),
            priority=goal_data.get('priority', 1)
        )
        
        # Enhance with collaborative partnership
        goal_id = base_response['goal_created']['goal_id']
        enhanced_goal_data = {
            **goal_data,
            'goal_id': goal_id,
            'created_at': datetime.now().isoformat(),
            'base_goal_data': base_response['goal_created']
        }
        
        # Create AI partnership
        partnership = self.collaborative_engine.create_investment_partnership(
            enhanced_goal_data, user_id
        )
        
        # Generate initial collaboration session
        brainstorm_session = self.collaborative_engine.initiate_brainstorming_session(
            goal_id, enhanced_goal_data
        )
        
        # Create enhanced response
        enhanced_response = {
            'goal_created': base_response['goal_created'],
            'ai_partnership': {
                'partnership_id': goal_id,
                'ai_commitment_level': partnership.ai_commitment_level,
                'partnership_strength': partnership.partnership_strength,
                'collaboration_style': self._determine_collaboration_style(partnership)
            },
            'brainstorming_session': brainstorm_session,
            'companion_response': self._generate_enhanced_companion_response(
                base_response, brainstorm_session, partnership
            ),
            'next_steps': self._generate_collaborative_next_steps(brainstorm_session),
            'partnership_invitation': self._generate_partnership_invitation(goal_data, partnership)
        }
        
        return enhanced_response
    
    def _determine_collaboration_style(self, partnership) -> str:
        """Determine AI collaboration style based on user profile"""
        profile = partnership.user_motivation_profile
        
        if profile.get('responds_to_excitement'):
            return "Enthusiastic Brainstorming Partner"
        elif profile.get('needs_emotional_security'):
            return "Supportive Planning Ally"
        elif profile.get('likes_technical_details'):
            return "Analytical Strategy Partner"
        else:
            return "Adaptive Collaboration Companion"
    
    def _generate_enhanced_companion_response(self, base_response: Dict, 
                                            brainstorm_session: Dict, 
                                            partnership) -> str:
        """Generate enhanced AI companion response with collaboration elements"""
        
        base_message = base_response.get('companion_response', '')
        goal_name = base_response['goal_created']['name']
        collaboration_style = self._determine_collaboration_style(partnership)
        
        enhanced_message = f"""
{base_message}

🤝 **Partnership Mode Activated!**

I'm not just tracking this goal - I'm your {collaboration_style}! Together, we're going to make "{goal_name}" happen.

I've already been brainstorming some exciting strategies for us:

• **{brainstorm_session['ai_ideas'][0]['title']}**: {brainstorm_session['ai_ideas'][0]['description']}
• **{brainstorm_session['ai_ideas'][1]['title']}**: {brainstorm_session['ai_ideas'][1]['description']}
{f"• **{brainstorm_session['ai_ideas'][2]['title']}**: {brainstorm_session['ai_ideas'][2]['description']}" if len(brainstorm_session['ai_ideas']) > 2 else ""}

What excites you most? I'm ready to dive deep and co-create the perfect approach with you! 🚀
        """.strip()
        
        return enhanced_message
    
    def _generate_collaborative_next_steps(self, brainstorm_session: Dict) -> List[str]:
        """Generate collaborative next steps"""
        return [
            "Review AI brainstorming ideas and share your thoughts",
            "Collaborate on refining the investment strategy",
            "Set up automated systems based on our agreed approach",
            "Schedule regular partnership check-ins",
            "Celebrate milestones together as we progress!"
        ]
    
    def _generate_partnership_invitation(self, goal_data: Dict, partnership) -> str:
        """Generate an invitation to active partnership"""
        
        goal_name = goal_data['name']
        target_amount = goal_data['target_amount']
        
        return f"""
🎯 **Partnership Invitation**

I'm genuinely excited to be your investment partner for "{goal_name}"!

Here's what our partnership means:
• I'll actively brainstorm strategies with you (not just track numbers)
• We'll adapt and evolve our approach based on what's working
• I'll provide motivation, accountability, and celebrate every win
• You get a thinking partner who never gets tired or judgmental

Ready to build ${target_amount:,.0f} together? Let's make this happen! 💪
        """
    
    def process_user_collaboration_response(self, goal_id: str, user_input: Dict) -> Dict:
        """Process user response to collaboration and continue the partnership"""
        
        # Parse user input for collaboration elements
        collaboration_response = {
            'goal_id': goal_id,
            'user_id': user_input.get('user_id', 'default'),
            'timestamp': datetime.now().isoformat(),
            'text': user_input.get('message', ''),
            'liked_ideas': self._extract_liked_ideas(user_input),
            'concerns': self._extract_concerns(user_input),
            'custom_ideas': self._extract_custom_ideas(user_input),
            'engagement_signals': self._detect_engagement_signals(user_input)
        }
        
        # Find the most recent brainstorming session
        sessions = self.collaborative_engine.collaboration_sessions
        recent_session = None
        for session in reversed(sessions):
            if session.get('goal_id') == goal_id:
                recent_session = session
                break
        
        if not recent_session:
            raise ValueError(f"No brainstorming session found for goal {goal_id}")
        
        # Process through collaborative engine
        ai_response = self.collaborative_engine.process_user_feedback(
            goal_id, recent_session['session_id'], collaboration_response
        )
        
        # Enhance with investment-specific elements
        enhanced_response = self._enhance_ai_response_with_investment_context(
            ai_response, goal_id, collaboration_response
        )
        
        # Log the interaction
        self._log_partnership_interaction(goal_id, collaboration_response, enhanced_response)
        
        return enhanced_response
    
    def _extract_liked_ideas(self, user_input: Dict) -> List[Dict]:
        """Extract which AI ideas the user liked"""
        liked_ideas = []
        message = user_input.get('message', '').lower()
        
        # Simple keyword detection (could be enhanced with NLP)
        if 'milestone' in message or 'step' in message:
            liked_ideas.append({'idea_type': 'milestone_strategy', 'confidence': 0.7})
        if 'automat' in message or 'set it and forget' in message:
            liked_ideas.append({'idea_type': 'automation_strategy', 'confidence': 0.8})
        if 'game' in message or 'fun' in message or 'point' in message:
            liked_ideas.append({'idea_type': 'gamification_strategy', 'confidence': 0.6})
        if 'value' in message or 'meaning' in message or 'believe' in message:
            liked_ideas.append({'idea_type': 'symbolic_strategy', 'confidence': 0.8})
        
        return liked_ideas
    
    def _extract_concerns(self, user_input: Dict) -> List[Dict]:
        """Extract user concerns from their message"""
        concerns = []
        message = user_input.get('message', '').lower()
        
        concern_keywords = {
            'risk': 'Risk tolerance concerns',
            'lose': 'Loss aversion', 
            'volatile': 'Market volatility fears',
            'time': 'Time management concerns',
            'money': 'Financial capacity concerns',
            'complex': 'Strategy complexity concerns'
        }
        
        for keyword, concern_type in concern_keywords.items():
            if keyword in message:
                concerns.append({
                    'type': concern_type,
                    'text': f"User mentioned '{keyword}' indicating {concern_type.lower()}",
                    'severity': 'medium'
                })
        
        return concerns
    
    def _extract_custom_ideas(self, user_input: Dict) -> List[str]:
        """Extract custom ideas the user suggested"""
        # This would be enhanced with NLP in production
        message = user_input.get('message', '')
        custom_ideas = []
        
        # Look for phrases like "what if", "I think", "maybe we could"
        idea_indicators = ["what if", "i think", "maybe we", "how about", "could we"]
        
        for indicator in idea_indicators:
            if indicator in message.lower():
                # Extract the sentence containing the idea
                sentences = message.split('.')
                for sentence in sentences:
                    if indicator in sentence.lower():
                        custom_ideas.append(sentence.strip())
        
        return custom_ideas
    
    def _detect_engagement_signals(self, user_input: Dict) -> Dict:
        """Detect engagement level and type from user input"""
        message = user_input.get('message', '').lower()
        
        signals = {
            'enthusiasm_level': 'medium',
            'question_count': message.count('?'),
            'exclamation_count': message.count('!'),
            'positive_words': 0,
            'uncertainty_words': 0
        }
        
        positive_words = ['love', 'great', 'excited', 'perfect', 'awesome', 'brilliant', 'yes']
        uncertainty_words = ['maybe', 'not sure', 'hmm', 'dunno', 'uncertain']
        
        signals['positive_words'] = sum(1 for word in positive_words if word in message)
        signals['uncertainty_words'] = sum(1 for word in uncertainty_words if word in message)
        
        # Calculate enthusiasm
        if signals['positive_words'] >= 2 or signals['exclamation_count'] >= 2:
            signals['enthusiasm_level'] = 'high'
        elif signals['uncertainty_words'] >= 2:
            signals['enthusiasm_level'] = 'low'
        
        return signals
    
    def _enhance_ai_response_with_investment_context(self, ai_response: Dict, 
                                                   goal_id: str, 
                                                   user_response: Dict) -> Dict:
        """Enhance AI response with investment-specific context"""
        
        # Get current goal data
        goal_data = self._get_goal_data(goal_id)
        
        # Add investment-specific enhancements
        enhanced_response = {
            **ai_response,
            'investment_specific': {
                'portfolio_impact': self._analyze_portfolio_impact(goal_data, ai_response),
                'market_context': self._get_relevant_market_context(goal_data),
                'risk_assessment': self._assess_risk_alignment(goal_data, ai_response),
                'timeline_optimization': self._optimize_investment_timeline(goal_data, ai_response)
            },
            'actionable_steps': self._generate_investment_action_steps(goal_data, ai_response),
            'partnership_momentum': self._calculate_partnership_momentum(goal_id)
        }
        
        return enhanced_response
    
    def _get_goal_data(self, goal_id: str) -> Dict:
        """Get goal data from investment integration"""
        # This would fetch from the actual goal storage
        return {
            'goal_id': goal_id,
            'target_amount': 5000,  # Placeholder
            'current_amount': 0,    # Placeholder
            'target_date': None     # Placeholder
        }
    
    def _analyze_portfolio_impact(self, goal_data: Dict, ai_response: Dict) -> Dict:
        """Analyze how this collaboration affects portfolio strategy"""
        return {
            'diversification_impact': 'Positive - adds goal-focused allocation',
            'risk_profile_change': 'Minimal - maintains current risk tolerance',
            'rebalancing_needed': False,
            'new_allocation_suggested': True
        }
    
    def _get_relevant_market_context(self, goal_data: Dict) -> Dict:
        """Get relevant market context for the investment goal"""
        return {
            'market_sentiment': 'Neutral - good time for systematic investing',
            'relevant_sectors': ['Technology', 'Healthcare', 'Renewable Energy'],
            'timing_recommendation': 'Dollar-cost averaging recommended',
            'volatility_outlook': 'Moderate - normal market conditions'
        }
    
    def _assess_risk_alignment(self, goal_data: Dict, ai_response: Dict) -> Dict:
        """Assess risk alignment between user preferences and goal requirements"""
        refined_strategy = ai_response.get('refined_strategy', {})
        
        return {
            'user_risk_preference': 'Moderate',
            'goal_risk_requirement': 'Moderate',
            'alignment_score': 0.85,
            'recommendations': ['Balanced index funds', 'Target-date funds', 'Conservative growth allocation']
        }
    
    def _optimize_investment_timeline(self, goal_data: Dict, ai_response: Dict) -> Dict:
        """Optimize investment timeline based on collaboration"""
        return {
            'optimal_timeline': '18-24 months',
            'milestone_frequency': 'Monthly check-ins',
            'rebalancing_schedule': 'Quarterly',
            'goal_achievement_probability': 0.88
        }
    
    def _generate_investment_action_steps(self, goal_data: Dict, ai_response: Dict) -> List[Dict]:
        """Generate specific investment action steps"""
        return [
            {
                'step': 'Account Setup',
                'description': 'Open investment account if needed',
                'timeline': 'This week',
                'ai_support': 'I\'ll help research the best platforms for your needs'
            },
            {
                'step': 'Initial Investment',
                'description': 'Make first investment based on our agreed strategy',
                'timeline': 'Next week',
                'ai_support': 'I\'ll remind you and celebrate this milestone!'
            },
            {
                'step': 'Automation Setup', 
                'description': 'Set up recurring investments',
                'timeline': 'Within 2 weeks',
                'ai_support': 'I\'ll help optimize the amount and frequency'
            }
        ]
    
    def _calculate_partnership_momentum(self, goal_id: str) -> Dict:
        """Calculate current partnership momentum"""
        partnership = self.collaborative_engine.partnerships.get(goal_id)
        
        if not partnership:
            return {'momentum': 'starting', 'score': 0.5}
        
        momentum_score = (partnership.partnership_strength + partnership.ai_commitment_level) / 2
        
        if momentum_score >= 0.8:
            momentum_level = 'high'
        elif momentum_score >= 0.6:
            momentum_level = 'building'
        elif momentum_score >= 0.4:
            momentum_level = 'developing'
        else:
            momentum_level = 'starting'
        
        return {
            'momentum': momentum_level,
            'score': momentum_score,
            'trajectory': 'upward' if partnership.partnership_strength > 0.5 else 'stabilizing'
        }
    
    def _log_partnership_interaction(self, goal_id: str, user_response: Dict, ai_response: Dict):
        """Log partnership interaction for analysis"""
        interaction = {
            'timestamp': datetime.now().isoformat(),
            'goal_id': goal_id,
            'user_engagement': user_response.get('engagement_signals', {}),
            'ai_response_type': ai_response.get('tone', 'unknown'),
            'partnership_strength_after': ai_response.get('partnership_update', {}).get('partnership_strength', 0),
            'next_step': ai_response.get('next_collaboration_step', {}).get('step_type', 'unknown')
        }
        
        # Save to partnership interactions log
        try:
            with open(self.partnership_interactions, 'r') as f:
                interactions = json.load(f)
        except FileNotFoundError:
            interactions = []
        
        interactions.append(interaction)
        
        with open(self.partnership_interactions, 'w') as f:
            json.dump(interactions, f, indent=2)
    
    def get_daily_partnership_updates(self, user_id: str) -> List[Dict]:
        """Get daily updates for all active partnerships"""
        updates = []
        
        for goal_id, partnership in self.collaborative_engine.partnerships.items():
            if partnership.user_id == user_id:
                check_in = self.collaborative_engine.generate_daily_partnership_check_in(goal_id)
                if check_in:
                    # Enhance with investment context
                    enhanced_check_in = self._enhance_check_in_with_investment_data(check_in, goal_id)
                    updates.append(enhanced_check_in)
        
        return updates
    
    def _enhance_check_in_with_investment_data(self, check_in: Dict, goal_id: str) -> Dict:
        """Enhance check-in with current investment performance data"""
        
        # This would fetch real investment data
        mock_investment_data = {
            'current_value': 1250.0,
            'target_value': 5000.0,
            'progress_percentage': 25.0,
            'recent_performance': '+2.3%',
            'days_to_goal': 547
        }
        
        enhanced_check_in = {
            **check_in,
            'investment_progress': mock_investment_data,
            'celebration_worthy': mock_investment_data['progress_percentage'] % 25 < 1,  # Celebrate every 25%
            'course_correction_needed': False,
            'market_opportunities': 'Dollar-cost averaging opportunity due to recent dip'
        }
        
        return enhanced_check_in

# Global bridge instance
investment_collaboration_bridge = InvestmentCollaborationBridge()

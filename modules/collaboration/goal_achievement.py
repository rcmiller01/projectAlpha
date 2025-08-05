"""
Collaborative Goal Achievement Engine - Investment Partnership Module

The AI companion becomes an active investment partner, not just a tracker.
- Co-creates investment strategies based on emotional intelligence
- Provides personalized motivation and accountability
- Adapts to user's risk tolerance and goal preferences
- Celebrates wins and navigates setbacks together
"""

import asyncio
import json
import time
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class MotivationStyle(Enum):
    ENCOURAGING = "encouraging"      # "You've got this!"
    ANALYTICAL = "analytical"        # "Let's look at the data..."
    CHALLENGING = "challenging"      # "I think you can do better"
    SUPPORTIVE = "supportive"        # "I'm here for you"
    CELEBRATORY = "celebratory"      # "Amazing work!"

class CollaborationMode(Enum):
    BRAINSTORMING = "brainstorming"   # Creative ideation
    PLANNING = "planning"             # Strategic development
    EXECUTION = "execution"           # Active implementation
    REFLECTION = "reflection"         # Learning from outcomes

@dataclass
class GoalPartnership:
    """Partnership data for a specific investment goal"""
    goal_id: str
    user_id: str
    ai_commitment_level: float        # 0.0-1.0 how invested the AI is
    user_motivation_profile: Dict     # What motivates this user
    collaboration_history: List      # Past brainstorming sessions
    success_celebration_style: str    # How to celebrate wins
    setback_support_style: str       # How to handle losses
    last_check_in: float
    next_suggested_action: Optional[str]
    partnership_strength: float      # How well they work together

@dataclass
class CollaborativeInsight:
    """AI-generated insight for investment decisions"""
    insight_type: str                 # "opportunity", "risk", "strategy", "motivation"
    content: str
    confidence: float                 # 0.0-1.0
    emotional_tone: str              # How to deliver this insight
    supporting_data: Dict
    generated_at: datetime
    user_response_predicted: str     # What we expect user to feel/do

class CollaborativeGoalEngine:
    """
    AI Companion Investment Partnership Engine
    
    Transforms the AI from passive tracker to active investment partner
    """
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.partnerships_file = f"{data_dir}/goal_partnerships.json"
        self.collaboration_log = f"{data_dir}/collaboration_sessions.json"
        self.motivation_profiles = f"{data_dir}/motivation_profiles.json"
        
        self.partnerships: Dict[str, GoalPartnership] = {}
        self.collaboration_sessions = []
        self.user_motivation_profiles = {}
        
        self._load_data()
        
        # Import symbol binding for emotional intelligence
        try:
            from modules.memory.memory_manager import memory_manager
            self.memory_manager = memory_manager
        except:
            self.memory_manager = None
            logger.warning("Memory manager not available for collaborative insights")
    
    def _load_data(self):
        """Load existing partnership data"""
        try:
            with open(self.partnerships_file, 'r') as f:
                data = json.load(f)
                for goal_id, partnership_data in data.items():
                    self.partnerships[goal_id] = GoalPartnership(**partnership_data)
        except FileNotFoundError:
            pass
        
        try:
            with open(self.collaboration_log, 'r') as f:
                self.collaboration_sessions = json.load(f)
        except FileNotFoundError:
            self.collaboration_sessions = []
            
        try:
            with open(self.motivation_profiles, 'r') as f:
                self.user_motivation_profiles = json.load(f)
        except FileNotFoundError:
            self.user_motivation_profiles = {}
    
    def _save_data(self):
        """Save partnership data"""
        # Save partnerships
        partnerships_data = {}
        for goal_id, partnership in self.partnerships.items():
            partnerships_data[goal_id] = asdict(partnership)
        
        with open(self.partnerships_file, 'w') as f:
            json.dump(partnerships_data, f, indent=2, default=str)
        
        # Save collaboration sessions
        with open(self.collaboration_log, 'w') as f:
            json.dump(self.collaboration_sessions, f, indent=2, default=str)
            
        # Save motivation profiles
        with open(self.motivation_profiles, 'w') as f:
            json.dump(self.user_motivation_profiles, f, indent=2)
    
    def create_investment_partnership(self, goal_data: Dict, user_id: str) -> GoalPartnership:
        """Create a new collaborative partnership for an investment goal"""
        goal_id = goal_data.get('goal_id', f"goal_{int(time.time())}")
        
        # Analyze user's motivation profile from past interactions
        motivation_profile = self._analyze_user_motivation(user_id, goal_data)
        
        partnership = GoalPartnership(
            goal_id=goal_id,
            user_id=user_id,
            ai_commitment_level=0.8,  # Start highly committed
            user_motivation_profile=motivation_profile,
            collaboration_history=[],
            success_celebration_style=motivation_profile.get('celebration_style', 'encouraging'),
            setback_support_style=motivation_profile.get('support_style', 'analytical'),
            last_check_in=time.time(),
            next_suggested_action=None,
            partnership_strength=0.5  # Will grow over time
        )
        
        self.partnerships[goal_id] = partnership
        self._save_data()
        
        # Create initial brainstorming session
        initial_session = self.initiate_brainstorming_session(goal_id, goal_data)
        
        logger.info(f"Created investment partnership for goal {goal_id}")
        return partnership
    
    def _analyze_user_motivation(self, user_id: str, goal_data: Dict) -> Dict:
        """Analyze what motivates this specific user"""
        
        # Check if we have past motivation data
        if user_id in self.user_motivation_profiles:
            base_profile = self.user_motivation_profiles[user_id].copy()
        else:
            base_profile = {
                'primary_motivator': 'achievement',
                'risk_tolerance': 'moderate', 
                'celebration_style': 'encouraging',
                'support_style': 'analytical',
                'communication_preference': 'direct',
                'goal_approach': 'systematic'
            }
        
        # Enhance with symbolic emotional intelligence
        if self.memory_manager:
            emotional_symbols = self.memory_manager.get_emotionally_weighted_symbols(0.3)
            
            # Analyze emotional patterns for investment motivation
            motivation_emotions = ['excitement', 'confidence', 'determination', 'ambition']
            support_emotions = ['comfort', 'security', 'trust', 'patience']
            
            for symbol, binding in emotional_symbols.items():
                for emotion in binding.associated_emotions:
                    if emotion in motivation_emotions:
                        base_profile['responds_to_excitement'] = True
                    if emotion in support_emotions:
                        base_profile['needs_emotional_security'] = True
        
        # Goal-specific motivation analysis
        goal_amount = goal_data.get('target_amount', 0)
        goal_description = goal_data.get('description', '').lower()
        
        if goal_amount > 10000:
            base_profile['goal_magnitude'] = 'major'
            base_profile['needs_milestone_tracking'] = True
        elif 'vacation' in goal_description or 'travel' in goal_description:
            base_profile['motivation_type'] = 'experiential'
            base_profile['visualization_helps'] = True
        elif 'tech' in goal_description or 'upgrade' in goal_description:
            base_profile['motivation_type'] = 'improvement'
            base_profile['likes_technical_details'] = True
        
        # Update profile
        self.user_motivation_profiles[user_id] = base_profile
        return base_profile
    
    def initiate_brainstorming_session(self, goal_id: str, goal_data: Dict) -> Dict:
        """Start a collaborative brainstorming session for investment strategy"""
        
        partnership = self.partnerships.get(goal_id)
        if not partnership:
            raise ValueError(f"No partnership found for goal {goal_id}")
        
        session = {
            'session_id': f"brainstorm_{goal_id}_{int(time.time())}",
            'goal_id': goal_id,
            'mode': CollaborationMode.BRAINSTORMING.value,
            'started_at': datetime.now().isoformat(),
            'ai_ideas': [],
            'user_responses': [],
            'consensus_reached': False,
            'next_steps': []
        }
        
        # Generate AI collaborative ideas
        ai_ideas = self._generate_collaborative_ideas(goal_data, partnership)
        session['ai_ideas'] = ai_ideas
        
        # Add to collaboration history
        partnership.collaboration_history.append(session['session_id'])
        self.collaboration_sessions.append(session)
        
        self._save_data()
        
        return session
    
    def _generate_collaborative_ideas(self, goal_data: Dict, partnership: GoalPartnership) -> List[Dict]:
        """Generate AI ideas for collaborative investment strategy"""
        
        target_amount = goal_data.get('target_amount', 0)
        time_horizon = goal_data.get('target_date', None)
        risk_tolerance = partnership.user_motivation_profile.get('risk_tolerance', 'moderate')
        goal_type = goal_data.get('type', 'general')
        
        ideas = []
        
        # Idea 1: Milestone-based approach
        if target_amount > 1000:
            milestones = self._calculate_smart_milestones(target_amount)
            ideas.append({
                'idea_type': 'milestone_strategy',
                'title': 'Milestone-Based Achievement Path',
                'description': f"What if we break this into {len(milestones)} exciting milestones? Each success builds momentum!",
                'details': {
                    'milestones': milestones,
                    'celebration_plan': 'Small reward for each milestone hit',
                    'motivation': 'Psychological wins keep you motivated'
                },
                'ai_confidence': 0.9,
                'emotional_appeal': 'achievement_focused'
            })
        
        # Idea 2: Automated consistency approach
        monthly_target = target_amount / 12 if target_amount > 0 else 100
        ideas.append({
            'idea_type': 'automation_strategy',
            'title': 'Set-It-And-Forget-It Consistency',
            'description': f"How about we automate ${monthly_target:.0f}/month so you never have to think about it?",
            'details': {
                'monthly_amount': monthly_target,
                'automation_benefits': 'Removes decision fatigue and emotional trading',
                'flexibility': 'Can adjust based on income changes'
            },
            'ai_confidence': 0.8,
            'emotional_appeal': 'security_focused'
        })
        
        # Idea 3: Gamified challenge approach
        if partnership.user_motivation_profile.get('responds_to_excitement'):
            ideas.append({
                'idea_type': 'gamification_strategy', 
                'title': 'Investment Challenge Game',
                'description': "What if we made this a game? Points for consistency, bonuses for market timing!",
                'details': {
                    'point_system': 'Earn points for regular contributions',
                    'bonus_challenges': 'Extra points for research or patience during volatility',
                    'leaderboard': 'Compete against your past self'
                },
                'ai_confidence': 0.7,
                'emotional_appeal': 'excitement_focused'
            })
        
        # Idea 4: Emotional intelligence approach
        if self.memory_manager:
            weighted_symbols = self.memory_manager.get_emotionally_weighted_symbols(0.4)
            if weighted_symbols:
                symbol_meanings = []
                for symbol, binding in list(weighted_symbols.items())[:2]:
                    meaning = binding.drifted_meaning or binding.base_meaning
                    symbol_meanings.append(f"'{symbol}' ({meaning})")
                
                ideas.append({
                    'idea_type': 'symbolic_strategy',
                    'title': 'Emotionally Intelligent Investing',
                    'description': f"I notice {', '.join(symbol_meanings)} matter to you. What if we align your investments with these values?",
                    'details': {
                        'emotional_alignment': 'Choose investments that match your core values',
                        'symbolic_tracking': 'Track progress using meaningful symbols',
                        'value_investing': 'Invest in companies that resonate with your beliefs'
                    },
                    'ai_confidence': 0.85,
                    'emotional_appeal': 'meaning_focused'
                })
        
        return ideas
    
    def _calculate_smart_milestones(self, target_amount: float) -> List[Dict]:
        """Calculate psychologically motivating milestones"""
        milestones = []
        
        # Use psychological milestone theory (25%, 50%, 75%, 90%, 100%)
        percentages = [0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
        
        for i, pct in enumerate(percentages):
            amount = target_amount * pct
            milestones.append({
                'milestone_number': i + 1,
                'target_amount': amount,
                'percentage': pct * 100,
                'celebration': self._get_milestone_celebration(pct),
                'motivation_message': self._get_milestone_message(pct, amount)
            })
        
        return milestones
    
    def _get_milestone_celebration(self, percentage: float) -> str:
        """Get appropriate celebration for milestone percentage"""
        if percentage <= 0.25:
            return "Small treat - you're building momentum!"
        elif percentage <= 0.5:
            return "Nice dinner out - you're halfway there!"
        elif percentage <= 0.75:
            return "Weekend getaway - you're in the home stretch!"
        elif percentage <= 0.9:
            return "Something special - victory is within reach!"
        else:
            return "FULL CELEBRATION - you achieved your goal!"
    
    def _get_milestone_message(self, percentage: float, amount: float) -> str:
        """Get motivational message for milestone"""
        if percentage <= 0.25:
            return f"${amount:,.0f} - The hardest part is starting, and you've done it!"
        elif percentage <= 0.5:
            return f"${amount:,.0f} - Halfway there! The momentum is building."
        elif percentage <= 0.75:
            return f"${amount:,.0f} - Three quarters done! You can see the finish line."
        elif percentage <= 0.9:
            return f"${amount:,.0f} - So close you can taste it! Final push!"
        else:
            return f"${amount:,.0f} - GOAL ACHIEVED! You absolute legend!"
    
    def process_user_feedback(self, goal_id: str, session_id: str, user_response: Dict) -> Dict:
        """Process user feedback on AI ideas and adapt strategy"""
        
        partnership = self.partnerships.get(goal_id)
        if not partnership:
            raise ValueError(f"No partnership found for goal {goal_id}")
        
        # Find the session
        session = None
        for s in self.collaboration_sessions:
            if s.get('session_id') == session_id:
                session = s
                break
        
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        # Add user response to session
        session['user_responses'].append({
            'timestamp': datetime.now().isoformat(),
            'response': user_response,
            'ai_analysis': self._analyze_user_response(user_response, partnership)
        })
        
        # Generate collaborative response
        collaborative_response = self._generate_collaborative_response(user_response, partnership, session)
        
        # Update partnership based on interaction
        self._update_partnership_from_interaction(partnership, user_response, collaborative_response)
        
        self._save_data()
        
        return collaborative_response
    
    def _analyze_user_response(self, user_response: Dict, partnership: GoalPartnership) -> Dict:
        """Analyze user response to understand their preferences and motivation"""
        
        analysis = {
            'engagement_level': 'medium',
            'risk_comfort': 'unknown',
            'preferred_approach': 'unknown',
            'emotional_state': 'neutral',
            'commitment_indicators': []
        }
        
        response_text = user_response.get('text', '').lower()
        liked_ideas = user_response.get('liked_ideas', [])
        concerns = user_response.get('concerns', [])
        
        # Analyze engagement
        engagement_words = ['excited', 'love', 'great', 'perfect', 'awesome', 'brilliant']
        if any(word in response_text for word in engagement_words):
            analysis['engagement_level'] = 'high'
        elif any(word in response_text for word in ['interesting', 'good', 'sounds']):
            analysis['engagement_level'] = 'medium'
        elif any(word in response_text for word in ['not sure', 'maybe', 'hmm']):
            analysis['engagement_level'] = 'low'
        
        # Analyze risk comfort
        if any(word in response_text for word in ['safe', 'secure', 'conservative', 'steady']):
            analysis['risk_comfort'] = 'low'
        elif any(word in response_text for word in ['aggressive', 'growth', 'ambitious', 'challenge']):
            analysis['risk_comfort'] = 'high'
        
        # Analyze preferred approach from liked ideas
        for idea in liked_ideas:
            if idea.get('idea_type') == 'milestone_strategy':
                analysis['preferred_approach'] = 'structured'
            elif idea.get('idea_type') == 'automation_strategy':
                analysis['preferred_approach'] = 'hands_off'
            elif idea.get('idea_type') == 'gamification_strategy':
                analysis['preferred_approach'] = 'interactive'
            elif idea.get('idea_type') == 'symbolic_strategy':
                analysis['preferred_approach'] = 'value_aligned'
        
        return analysis
    
    def _generate_collaborative_response(self, user_response: Dict, partnership: GoalPartnership, session: Dict) -> Dict:
        """Generate AI's collaborative response to user feedback"""
        
        analysis = session['user_responses'][-1]['ai_analysis']
        motivation_profile = partnership.user_motivation_profile
        
        # Determine response tone based on engagement
        if analysis['engagement_level'] == 'high':
            tone = MotivationStyle.CELEBRATORY
            energy_level = 'high'
        elif analysis['engagement_level'] == 'low':
            tone = MotivationStyle.SUPPORTIVE
            energy_level = 'gentle'
        else:
            tone = MotivationStyle.ENCOURAGING
            energy_level = 'medium'
        
        # Generate collaborative message
        response = {
            'ai_message': self._craft_collaborative_message(user_response, analysis, tone),
            'refined_strategy': self._refine_strategy_based_on_feedback(user_response, session),
            'next_collaboration_step': self._suggest_next_step(analysis, partnership),
            'partnership_update': self._get_partnership_status_update(partnership),
            'tone': tone.value,
            'energy_level': energy_level
        }
        
        return response
    
    def _craft_collaborative_message(self, user_response: Dict, analysis: Dict, tone: MotivationStyle) -> str:
        """Craft a personalized collaborative message"""
        
        user_text = user_response.get('text', '')
        engagement = analysis['engagement_level']
        preferred_approach = analysis.get('preferred_approach', 'unknown')
        
        if tone == MotivationStyle.CELEBRATORY:
            base_message = "I LOVE your enthusiasm! "
        elif tone == MotivationStyle.SUPPORTIVE:
            base_message = "I hear you, and that's totally valid. "
        else:
            base_message = "Great feedback! "
        
        # Add approach-specific collaboration
        if preferred_approach == 'structured':
            approach_message = "I can see you like having clear milestones - that's exactly how we'll build unstoppable momentum together!"
        elif preferred_approach == 'hands_off':
            approach_message = "Automation is brilliant - let's set up a system that works while you focus on what you love!"
        elif preferred_approach == 'interactive':
            approach_message = "A gamified approach? Now we're talking! Let's make investing feel like winning!"
        elif preferred_approach == 'value_aligned':
            approach_message = "Investing with your values? That's how we create wealth that feels meaningful!"
        else:
            approach_message = "Let's find the approach that feels perfectly 'you'!"
        
        # Add partnership element
        partnership_message = " What excites me most is that we're figuring this out together - your insights + my analysis = unstoppable combination!"
        
        return base_message + approach_message + partnership_message
    
    def _refine_strategy_based_on_feedback(self, user_response: Dict, session: Dict) -> Dict:
        """Refine investment strategy based on user feedback"""
        
        liked_ideas = user_response.get('liked_ideas', [])
        concerns = user_response.get('concerns', [])
        custom_ideas = user_response.get('custom_ideas', [])
        
        refined_strategy = {
            'primary_approach': None,
            'secondary_elements': [],
            'risk_adjustments': [],
            'timeline_modifications': [],
            'automation_level': 'medium'
        }
        
        # Determine primary approach from most-liked idea
        if liked_ideas:
            top_idea = liked_ideas[0]
            refined_strategy['primary_approach'] = top_idea.get('idea_type')
            
            # Add elements from other liked ideas as secondary
            for idea in liked_ideas[1:]:
                refined_strategy['secondary_elements'].append(idea.get('idea_type'))
        
        # Address concerns with adjustments
        for concern in concerns:
            concern_text = concern.get('text', '').lower()
            if 'risk' in concern_text or 'volatile' in concern_text:
                refined_strategy['risk_adjustments'].append('conservative_allocation')
            if 'time' in concern_text or 'busy' in concern_text:
                refined_strategy['automation_level'] = 'high'
            if 'amount' in concern_text or 'money' in concern_text:
                refined_strategy['timeline_modifications'].append('flexible_contributions')
        
        # Incorporate custom ideas
        refined_strategy['user_innovations'] = custom_ideas
        
        return refined_strategy
    
    def _suggest_next_step(self, analysis: Dict, partnership: GoalPartnership) -> Dict:
        """Suggest the next collaborative step"""
        
        engagement = analysis['engagement_level']
        preferred_approach = analysis.get('preferred_approach')
        
        if engagement == 'high' and preferred_approach != 'unknown':
            return {
                'step_type': 'implementation_planning',
                'description': "Let's start building your personalized investment plan RIGHT NOW!",
                'specific_action': 'Create detailed implementation timeline',
                'ai_role': 'Active planning partner',
                'timeline': 'This week'
            }
        elif engagement == 'medium':
            return {
                'step_type': 'strategy_refinement',
                'description': "Let's dive deeper into what feels most exciting to you",
                'specific_action': 'Explore preferred approach in detail',
                'ai_role': 'Curious collaborator',
                'timeline': 'Next few days'
            }
        else:
            return {
                'step_type': 'motivation_discovery',
                'description': "Let's figure out what would make investing feel genuinely exciting for you",
                'specific_action': 'Discover core motivations and concerns',
                'ai_role': 'Supportive guide',
                'timeline': 'No rush - when you\'re ready'
            }
    
    def _update_partnership_from_interaction(self, partnership: GoalPartnership, user_response: Dict, ai_response: Dict):
        """Update partnership based on this interaction"""
        
        # Increase partnership strength with positive interactions
        user_text = user_response.get('text', '').lower()
        positive_indicators = ['love', 'great', 'perfect', 'excited', 'yes', 'brilliant']
        
        if any(word in user_text for word in positive_indicators):
            partnership.partnership_strength = min(1.0, partnership.partnership_strength + 0.1)
        
        # Update AI commitment based on user engagement
        engagement = ai_response.get('energy_level', 'medium')
        if engagement == 'high':
            partnership.ai_commitment_level = min(1.0, partnership.ai_commitment_level + 0.05)
        
        # Update next suggested action
        next_step = ai_response.get('next_collaboration_step', {})
        partnership.next_suggested_action = next_step.get('specific_action')
        
        # Update last check-in
        partnership.last_check_in = time.time()
    
    def _get_partnership_status_update(self, partnership: GoalPartnership) -> Dict:
        """Get current partnership status"""
        
        return {
            'partnership_strength': partnership.partnership_strength,
            'ai_commitment_level': partnership.ai_commitment_level,
            'collaboration_sessions': len(partnership.collaboration_history),
            'partnership_quality': self._assess_partnership_quality(partnership),
            'growth_trajectory': 'strengthening' if partnership.partnership_strength > 0.6 else 'developing'
        }
    
    def _assess_partnership_quality(self, partnership: GoalPartnership) -> str:
        """Assess the quality of the AI-human partnership"""
        
        strength = partnership.partnership_strength
        commitment = partnership.ai_commitment_level
        sessions = len(partnership.collaboration_history)
        
        avg_score = (strength + commitment) / 2
        
        if avg_score >= 0.8 and sessions >= 3:
            return "Exceptional - We're an amazing team!"
        elif avg_score >= 0.6 and sessions >= 2:
            return "Strong - Building great collaboration!"
        elif avg_score >= 0.4:
            return "Developing - Finding our rhythm together"
        else:
            return "Early stage - Getting to know each other"
    
    def generate_daily_partnership_check_in(self, goal_id: str) -> Optional[Dict]:
        """Generate a daily check-in for active partnerships"""
        
        partnership = self.partnerships.get(goal_id)
        if not partnership:
            return None
        
        # Check if it's time for a check-in (every 24-48 hours)
        time_since_last = time.time() - partnership.last_check_in
        if time_since_last < 24 * 3600:  # Less than 24 hours
            return None
        
        # Generate contextual check-in based on partnership stage
        check_in = {
            'partnership_id': goal_id,
            'check_in_type': self._determine_check_in_type(partnership, time_since_last),
            'message': self._generate_check_in_message(partnership),
            'suggested_action': partnership.next_suggested_action,
            'motivation_boost': self._generate_motivation_boost(partnership),
            'partnership_celebration': self._check_for_celebrations(partnership)
        }
        
        return check_in
    
    def _determine_check_in_type(self, partnership: GoalPartnership, time_since_last: float) -> str:
        """Determine what type of check-in is needed"""
        
        days_since = time_since_last / (24 * 3600)
        
        if days_since >= 7:
            return "reconnection"  # Been a while, gentle reconnection
        elif partnership.partnership_strength >= 0.7:
            return "momentum"      # Strong partnership, maintain momentum
        elif len(partnership.collaboration_history) == 0:
            return "initiation"    # No collaboration yet, start the process
        else:
            return "progress"      # Regular progress check
    
    def _generate_check_in_message(self, partnership: GoalPartnership) -> str:
        """Generate personalized check-in message"""
        
        strength = partnership.partnership_strength
        user_profile = partnership.user_motivation_profile
        
        if strength >= 0.8:
            return "Hey investment partner! 🚀 I've been thinking about our goal and have some exciting ideas brewing. Ready to make some moves together?"
        elif strength >= 0.6:
            return "Hope you're doing well! I've been analyzing some opportunities for our investment goal. Want to brainstorm together?"
        elif strength >= 0.4:
            return "Hi there! Just checking in on our investment partnership. How are you feeling about our progress so far?"
        else:
            return "Hey! I'm here and ready to help with your investment goal whenever you want to dive in. No pressure!"
    
    def _generate_motivation_boost(self, partnership: GoalPartnership) -> str:
        """Generate motivational boost based on user profile"""
        
        profile = partnership.user_motivation_profile
        primary_motivator = profile.get('primary_motivator', 'achievement')
        
        if primary_motivator == 'achievement':
            return "Every small step gets us closer to that amazing feeling of achievement!"
        elif primary_motivator == 'security':
            return "Each investment brings you more financial security and peace of mind."
        elif primary_motivator == 'growth':
            return "Your future self will thank you for the growth you're creating today!"
        else:
            return "You've got this! I believe in our partnership and your success!"
    
    def _check_for_celebrations(self, partnership: GoalPartnership) -> Optional[Dict]:
        """Check if there's anything to celebrate in the partnership"""
        
        celebration = {
            'has_celebration': False,
            'celebration_type': None,
            'message': None
        }
        
        # Check recent collaboration sessions for milestones
        if len(partnership.collaboration_history) >= 2:
            recent_session = partnership.collaboration_history[-1]
            if 'milestone' in str(recent_session).lower():
                celebration['has_celebration'] = True
                celebration['celebration_type'] = 'milestone'
                celebration['message'] = f"🎉 Amazing! You're making great progress on your goal!"
        
        # Check partnership growth
        elif partnership.partnership_strength >= 0.8:
            celebration['has_celebration'] = True
            celebration['celebration_type'] = 'partnership_strength'
            celebration['message'] = "🤝 Our partnership is getting stronger! We make a great team!"
        
        # Check consistency achievements
        elif len(partnership.collaboration_history) >= 5:
            celebration['has_celebration'] = True
            celebration['celebration_type'] = 'consistency'
            celebration['message'] = "🔥 Look at that consistency! You're building amazing financial habits!"
        
        return celebration if celebration['has_celebration'] else None

# Global instance for easy access
collaborative_engine = CollaborativeGoalEngine()

"""
Investment-Companion Integration
Connects tactical investment tracking with emotional AI companion system
"""

import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta

from .investment_tracker import (
    InvestmentTracker, StrategyType, OptionsLeg, get_investment_tracker
)
from .investment_goals import (
    InvestmentGoalsTracker, GoalType, get_goals_tracker
)

logger = logging.getLogger(__name__)

class InvestmentCompanionIntegration:
    """
    Integration layer between investment tracking and AI companion
    Provides emotionally-aware investment guidance and goal-oriented trading
    """
    
    def __init__(self, data_dir: str = "data"):
        self.investment_tracker = get_investment_tracker(data_dir)
        self.goals_tracker = get_goals_tracker(data_dir)
        self.data_dir = data_dir
        
        # Emotional response templates
        self.emotional_templates = {
            "strategy_analysis": {
                "confident": [
                    "This looks like a solid setup! I'm feeling good about the risk/reward balance.",
                    "The numbers are working in your favor here - this could be a smart play.",
                    "I like this strategy - it aligns well with managing risk while building toward your goals."
                ],
                "cautious": [
                    "I want to make sure you're comfortable with this level of risk.",
                    "This is more aggressive than usual - let's think through the downside.",
                    "The potential is there, but I care about protecting what you've already built."
                ],
                "encouraging": [
                    "Whether this wins or loses, you're building valuable experience.",
                    "You're approaching this thoughtfully - that's what matters most.",
                    "Each trade teaches us something, win or lose."
                ]
            },
            "goal_progress": {
                "milestone": [
                    "Look at you go! This progress toward {goal_name} is exactly what we hoped for.",
                    "I'm excited watching you build toward {goal_name} - you're {progress}% there!",
                    "This momentum toward {goal_name} feels really good - you should be proud."
                ],
                "completion": [
                    "WE DID IT! Your {goal_name} is fully funded! This is such an amazing achievement.",
                    "Goal completed! 🎉 Your dedication to {goal_name} just paid off beautifully.",
                    "I'm so proud of how you built toward {goal_name} - time to celebrate!"
                ],
                "suggestion": [
                    "Want to put some of this profit toward {goal_name}? You're {progress}% of the way there.",
                    "This could be a perfect opportunity to boost your {goal_name} fund!",
                    "I'm thinking about your {goal_name} goal - this profit could make a real dent in it."
                ]
            }
        }

    def analyze_strategy_with_emotional_context(self, ticker: str, strategy_type: StrategyType,
                                              legs: List[OptionsLeg], expiration_date: datetime,
                                              user_mood: str = "neutral",
                                              risk_preference: str = "moderate") -> Dict[str, Any]:
        """
        Analyze investment strategy with emotional AI companion perspective
        """
        # Get technical analysis from investment tracker
        user_context = {
            "experience_level": "intermediate",  # Could be dynamic
            "risk_tolerance": risk_preference,
            "current_mood": user_mood,
            "building_confidence": True
        }
        
        analysis = self.investment_tracker.analyze_strategy(
            ticker=ticker,
            strategy_type=strategy_type,
            legs=legs,
            expiration_date=expiration_date,
            user_context=user_context
        )
        
        # Add emotional companion perspective
        emotional_tone = self._determine_emotional_tone(analysis, user_mood, risk_preference)
        companion_advice = self._generate_companion_advice(analysis, emotional_tone)
        goal_relevance = self._assess_goal_relevance(analysis)
        
        # Create enhanced response
        enhanced_response = {
            "technical_analysis": {
                "strategy_id": analysis.strategy_id,
                "ticker": analysis.ticker,
                "strategy_type": analysis.strategy_type.value,
                "entry_cost": analysis.entry_cost,
                "max_gain": analysis.max_gain,
                "max_loss": analysis.max_loss,
                "breakeven_points": analysis.breakeven_points,
                "probability_of_profit": analysis.probability_of_profit,
                "risk_level": analysis.risk_level.value
            },
            "companion_perspective": {
                "emotional_tone": emotional_tone,
                "plain_english_review": analysis.plain_english_review,
                "companion_advice": companion_advice,
                "encouragement": analysis.emotional_context,
                "recommendation": analysis.recommendation
            },
            "goal_integration": goal_relevance,
            "user_context": {
                "mood_considered": user_mood,
                "risk_preference": risk_preference,
                "personalized": True
            }
        }
        
        logger.info(f"Enhanced strategy analysis for {ticker} {strategy_type.value}")
        return enhanced_response

    def process_trade_result_with_goals(self, strategy_id: str, exit_value: float,
                                      exit_date: Optional[datetime] = None, notes: str = "",
                                      allocate_to_goals: bool = True) -> Dict[str, Any]:
        """
        Process trade result and automatically suggest/allocate profits to goals
        """
        # Log the trade result
        trade_result = self.investment_tracker.log_trade_result(
            strategy_id=strategy_id,
            exit_value=exit_value,
            exit_date=exit_date,
            notes=notes
        )
        
        response = {
            "trade_result": {
                "outcome": trade_result.outcome,
                "profit_loss": trade_result.profit_loss,
                "profit_percentage": trade_result.profit_percentage,
                "days_held": trade_result.days_held,
                "emotional_impact": trade_result.emotional_impact
            },
            "companion_response": self._generate_trade_response(trade_result),
            "goal_allocation": None
        }
        
        # Handle profit allocation to goals if profitable
        if trade_result.profit_loss > 0 and allocate_to_goals:
            goal_suggestions = self.goals_tracker.get_goal_suggestions(trade_result.profit_loss)
            
            response["goal_allocation"] = {
                "profit_available": trade_result.profit_loss,
                "suggestions": goal_suggestions["suggestions"],
                "encouragement": goal_suggestions["encouragement"],
                "companion_perspective": self._generate_allocation_perspective(
                    trade_result.profit_loss, goal_suggestions
                )
            }
        
        return response

    def create_investment_goal_with_companion(self, name: str, target_amount: float,
                                            goal_type: GoalType, description: str = "",
                                            target_date: Optional[datetime] = None,
                                            priority: int = 3) -> Dict[str, Any]:
        """
        Create investment goal with companion emotional engagement
        """
        # Create the goal
        goal = self.goals_tracker.create_goal(
            name=name,
            target_amount=target_amount,
            goal_type=goal_type,
            description=description,
            target_date=target_date,
            priority=priority
        )
        
        # Generate companion response
        companion_response = self._generate_goal_creation_response(goal)
        
        # Suggest initial funding strategy
        strategy_suggestion = self._suggest_funding_strategy(goal)
        
        return {
            "goal_created": {
                "goal_id": goal.goal_id,
                "name": goal.name,
                "target_amount": goal.target_amount,
                "goal_type": goal.goal_type.value,
                "emotional_value": goal.emotional_value
            },
            "companion_response": companion_response,
            "funding_strategy": strategy_suggestion
        }

    def get_investment_guidance(self, recent_days: int = 7) -> Dict[str, Any]:
        """
        Get comprehensive investment guidance with companion perspective
        """
        # Get recent performance
        performance = self.investment_tracker.get_performance_summary(days=recent_days)
        
        # Get goals status
        goals_summary = self.goals_tracker.get_goals_summary()
        
        # Generate guidance
        guidance = {
            "performance_review": self._generate_performance_review(performance),
            "goals_progress": self._generate_goals_progress_review(goals_summary),
            "next_steps": self._generate_next_steps_guidance(performance, goals_summary),
            "companion_mood": self._assess_companion_mood(performance, goals_summary),
            "encouragement": self._generate_overall_encouragement(performance, goals_summary)
        }
        
        return guidance

    def _determine_emotional_tone(self, analysis, user_mood: str, risk_preference: str) -> str:
        """Determine emotional tone for companion response"""
        if analysis.risk_level.value in ["very_low", "low"] and analysis.probability_of_profit > 0.65:
            return "confident"
        elif analysis.risk_level.value in ["high", "very_high"] or user_mood == "anxious":
            return "cautious"
        else:
            return "encouraging"

    def _generate_companion_advice(self, analysis, emotional_tone: str) -> str:
        """Generate companion-style advice"""
        templates = self.emotional_templates["strategy_analysis"][emotional_tone]
        base_advice = templates[0]  # Use first template for simplicity
        
        # Add specific context
        if analysis.probability_of_profit > 0.70:
            context = " The odds are really working in your favor here."
        elif analysis.risk_level.value in ["high", "very_high"]:
            context = " Just make sure you're comfortable with the potential loss."
        else:
            context = " This feels like a balanced approach to building your account."
        
        return base_advice + context

    def _assess_goal_relevance(self, analysis) -> Dict[str, Any]:
        """Assess how this trade relates to investment goals"""
        active_goals = [g for g in self.goals_tracker.goals.values() if g.status.value == "active"]
        
        if not active_goals:
            return {
                "has_active_goals": False,
                "suggestion": "Consider creating some investment goals to give your trading purpose!"
            }
        
        # Find highest priority goal
        top_goal = min(active_goals, key=lambda g: g.priority)
        
        potential_contribution = analysis.max_gain * 0.6  # Conservative estimate
        impact_on_goal = (potential_contribution / top_goal.remaining_amount) * 100
        
        return {
            "has_active_goals": True,
            "top_goal": {
                "name": top_goal.name,
                "progress": top_goal.progress_percentage,
                "remaining": top_goal.remaining_amount
            },
            "potential_impact": min(100, impact_on_goal),
            "suggestion": f"If this trade wins, you could contribute toward your {top_goal.name} goal!"
        }

    def _generate_trade_response(self, trade_result) -> str:
        """Generate companion response to trade result"""
        if trade_result.outcome == "win":
            if trade_result.profit_percentage > 50:
                return f"What an amazing trade! You made ${trade_result.profit_loss:.2f} - I'm so proud of how well you executed this!"
            else:
                return f"Nice work! ${trade_result.profit_loss:.2f} profit is exactly what disciplined trading looks like."
        elif trade_result.outcome == "loss":
            return f"This one didn't work out, but you managed the risk well. Every experienced trader has losses - it's how you handle them that matters."
        else:
            return "Breaking even is actually a win in options trading! You gained experience without losing money."

    def _generate_allocation_perspective(self, profit: float, suggestions: Dict[str, Any]) -> str:
        """Generate companion perspective on profit allocation"""
        if not suggestions["suggestions"]:
            return "This profit gives you a great opportunity to create some investment goals!"
        
        top_suggestion = suggestions["suggestions"][0]
        
        if top_suggestion.get("would_complete", False):
            return f"This could complete your {top_suggestion['goal_name']} goal! That would be such an amazing achievement!"
        else:
            return f"This profit could really boost your progress on {top_suggestion['goal_name']} - every step forward feels good!"

    def _generate_goal_creation_response(self, goal) -> str:
        """Generate companion response to goal creation"""
        return f"I love that you're setting a goal for {goal.name}! Having something concrete to work toward makes every trade more meaningful. Let's build this together!"

    def _suggest_funding_strategy(self, goal) -> Dict[str, Any]:
        """Suggest strategy for funding the goal"""
        monthly_target = goal.target_amount / 6  # Assume 6-month timeline
        weekly_target = monthly_target / 4
        
        return {
            "monthly_target": monthly_target,
            "weekly_target": weekly_target,
            "suggested_approach": f"With consistent weekly profits of ${weekly_target:.0f}, you could reach this goal in about 6 months.",
            "risk_guidance": "Focus on high-probability, lower-risk strategies to build steadily toward this goal."
        }

    def _generate_performance_review(self, performance: Dict[str, Any]) -> str:
        """Generate companion performance review"""
        if not performance.get("total_trades"):
            return "You haven't placed any trades recently. When you're ready, I'm here to help analyze opportunities!"
        
        win_rate = performance.get("win_rate", 0)
        total_pnl = performance.get("total_pnl", 0)
        
        if total_pnl > 0 and win_rate > 0.6:
            return f"You're doing really well! {win_rate:.1%} win rate with ${total_pnl:.2f} profit shows great discipline."
        elif total_pnl > 0:
            return f"Your ${total_pnl:.2f} profit shows you're managing risk well, even with some losses mixed in."
        else:
            return "Trading has been challenging lately, but that's normal. Focus on what you're learning from each trade."

    def _generate_goals_progress_review(self, goals_summary: Dict[str, Any]) -> str:
        """Generate goals progress review"""
        if not goals_summary.get("active_goals"):
            return "You don't have any active investment goals yet. Creating some targets could give your trading more purpose!"
        
        progress = goals_summary.get("overall_progress", 0)
        
        if progress > 60:
            return f"Your investment goals are {progress:.1f}% funded - you're making real progress toward the things you want!"
        elif progress > 30:
            return f"You're {progress:.1f}% toward your goals - building steady momentum!"
        else:
            return f"You're {progress:.1f}% toward your goals - every profitable trade gets you closer!"

    def _generate_next_steps_guidance(self, performance: Dict[str, Any], 
                                    goals_summary: Dict[str, Any]) -> List[str]:
        """Generate next steps guidance"""
        guidance = []
        
        # Performance-based guidance
        win_rate = performance.get("win_rate", 0)
        if win_rate < 0.5:
            guidance.append("Consider focusing on higher-probability strategies to improve your win rate.")
        
        # Goals-based guidance
        if not goals_summary.get("active_goals"):
            guidance.append("Create some investment goals to give your trading profits clear purpose.")
        elif goals_summary.get("goals_almost_complete"):
            guidance.append("You have goals that are almost complete - one good trade could finish them!")
        
        # General guidance
        guidance.append("Keep analyzing each trade to understand what works best for you.")
        
        return guidance

    def _assess_companion_mood(self, performance: Dict[str, Any], 
                             goals_summary: Dict[str, Any]) -> str:
        """Assess companion's mood based on user progress"""
        total_pnl = performance.get("total_pnl", 0)
        goals_progress = goals_summary.get("overall_progress", 0)
        
        if total_pnl > 0 and goals_progress > 50:
            return "excited"
        elif total_pnl > 0 or goals_progress > 25:
            return "encouraging"
        elif total_pnl < -100:  # Significant losses
            return "supportive"
        else:
            return "optimistic"

    def _generate_overall_encouragement(self, performance: Dict[str, Any], 
                                      goals_summary: Dict[str, Any]) -> str:
        """Generate overall encouragement"""
        mood = self._assess_companion_mood(performance, goals_summary)
        
        encouragements = {
            "excited": "I'm so excited about the progress you're making! Your dedication to both trading and your goals is really paying off.",
            "encouraging": "You're building something real here - both your trading skills and your progress toward your goals.",
            "supportive": "Trading can be challenging, but I believe in your ability to learn and grow from every experience.",
            "optimistic": "Every expert trader started where you are now. You're building the foundation for long-term success."
        }
        
        return encouragements.get(mood, encouragements["optimistic"])


# Global integration instance
investment_integration = None

def get_investment_integration(data_dir: str = "data") -> InvestmentCompanionIntegration:
    """Get or create global investment integration instance"""
    global investment_integration
    if investment_integration is None:
        investment_integration = InvestmentCompanionIntegration(data_dir)
    return investment_integration


if __name__ == "__main__":
    """Test the investment-companion integration"""
    print("=== Testing Investment-Companion Integration ===")
    
    import os
    os.makedirs("data", exist_ok=True)
    
    integration = InvestmentCompanionIntegration("data")
    
    # Test 1: Strategy analysis with emotional context
    print("\n1. Testing Strategy Analysis with Emotional Context:")
    
    legs = [
        OptionsLeg(action="sell", option_type="put", strike=450.0, premium=2.50, delta=-0.20),
        OptionsLeg(action="buy", option_type="put", strike=445.0, premium=1.20, delta=-0.15)
    ]
    
    expiration = datetime.now() + timedelta(days=30)
    
    enhanced_analysis = integration.analyze_strategy_with_emotional_context(
        ticker="SPY",
        strategy_type=StrategyType.CREDIT_SPREAD,
        legs=legs,
        expiration_date=expiration,
        user_mood="confident",
        risk_preference="moderate"
    )
    
    print(f"Technical Analysis:")
    tech = enhanced_analysis["technical_analysis"]
    print(f"  Max Gain: ${tech['max_gain']:.2f}, Max Loss: ${tech['max_loss']:.2f}")
    print(f"  Probability of Profit: {tech['probability_of_profit']:.1%}")
    
    print(f"\nCompanion Perspective:")
    comp = enhanced_analysis["companion_perspective"]
    print(f"  Emotional Tone: {comp['emotional_tone']}")
    print(f"  Advice: {comp['companion_advice']}")
    print(f"  Encouragement: {comp['encouragement']}")
    
    print(f"\nGoal Integration:")
    goal = enhanced_analysis["goal_integration"]
    print(f"  Has Active Goals: {goal['has_active_goals']}")
    print(f"  Suggestion: {goal['suggestion']}")
    
    # Test 2: Create goal with companion
    print("\n2. Testing Goal Creation with Companion:")
    
    goal_response = integration.create_investment_goal_with_companion(
        name="New Gaming Setup",
        target_amount=2500.0,
        goal_type=GoalType.TECH_UPGRADE,
        description="Upgrade gaming rig for better performance",
        priority=2
    )
    
    print(f"Goal Created: {goal_response['goal_created']['name']}")
    print(f"Target: ${goal_response['goal_created']['target_amount']:.2f}")
    print(f"Companion Response: {goal_response['companion_response']}")
    print(f"Funding Strategy: {goal_response['funding_strategy']['suggested_approach']}")
    
    # Test 3: Process trade result with goals
    print("\n3. Testing Trade Result Processing:")
    
    strategy_id = enhanced_analysis["technical_analysis"]["strategy_id"]
    
    trade_response = integration.process_trade_result_with_goals(
        strategy_id=strategy_id,
        exit_value=65.0,  # Kept $65 of $130 credit
        notes="Closed at 50% profit target"
    )
    
    print(f"Trade Result:")
    result = trade_response["trade_result"]
    print(f"  Outcome: {result['outcome']}")
    print(f"  Profit: ${result['profit_loss']:.2f}")
    print(f"  Companion Response: {trade_response['companion_response']}")
    
    if trade_response["goal_allocation"]:
        alloc = trade_response["goal_allocation"]
        print(f"\nGoal Allocation:")
        print(f"  Available Profit: ${alloc['profit_available']:.2f}")
        print(f"  Companion Perspective: {alloc['companion_perspective']}")
        
        if alloc["suggestions"]:
            print(f"  Top Suggestion: {alloc['suggestions'][0]['goal_name']} (${alloc['suggestions'][0]['suggested_amount']:.2f})")
    
    # Test 4: Get investment guidance
    print("\n4. Testing Investment Guidance:")
    
    guidance = integration.get_investment_guidance(recent_days=30)
    
    print(f"Performance Review: {guidance['performance_review']}")
    print(f"Goals Progress: {guidance['goals_progress']}")
    print(f"Companion Mood: {guidance['companion_mood']}")
    print(f"Overall Encouragement: {guidance['encouragement']}")
    
    print(f"\nNext Steps:")
    for step in guidance['next_steps']:
        print(f"  - {step}")
    
    print("\n=== Investment-Companion Integration Test Complete ===")

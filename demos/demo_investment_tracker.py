"""
Investment Tracker Demo
Interactive demonstration of the tactical investment strategy tracker
"""

import os
from datetime import datetime, timedelta
from modules.finance import (
    get_investment_integration, StrategyType, GoalType, OptionsLeg
)

def run_investment_demo():
    """Run an interactive demo of the investment tracking system"""
    print("🎯 TACTICAL INVESTMENT STRATEGY TRACKER DEMO")
    print("=" * 55)
    print("Welcome to your personal investment companion!")
    print("This demo shows how AI-powered guidance combines with goal-oriented trading.\n")
    
    # Setup
    os.makedirs("data", exist_ok=True)
    integration = get_investment_integration("data")
    
    # Demo 1: Create Investment Goals
    print("📋 STEP 1: Setting Up Your Investment Goals")
    print("-" * 45)
    
    # Create a vacation goal
    vacation_response = integration.create_investment_goal_with_companion(
        name="Dream Vacation to Japan",
        target_amount=5000.0,
        goal_type=GoalType.VACATION,
        description="Two weeks exploring Tokyo, Kyoto, and Mount Fuji",
        priority=1
    )
    
    print(f"✅ Created Goal: {vacation_response['goal_created']['name']}")
    print(f"   Target: ${vacation_response['goal_created']['target_amount']:,.2f}")
    print(f"   🤖 Companion says: \"{vacation_response['companion_response']}\"")
    print(f"   💡 Strategy: {vacation_response['funding_strategy']['suggested_approach']}")
    
    # Create a tech goal  
    tech_response = integration.create_investment_goal_with_companion(
        name="Home Server Upgrade",
        target_amount=2000.0,
        goal_type=GoalType.TECH_UPGRADE,
        description="High-performance server for AI development",
        priority=2
    )
    
    print(f"\n✅ Created Goal: {tech_response['goal_created']['name']}")
    print(f"   Target: ${tech_response['goal_created']['target_amount']:,.2f}")
    print(f"   🤖 Companion says: \"{tech_response['companion_response']}\"")
    
    # Demo 2: Analyze Investment Strategy
    print(f"\n\n📊 STEP 2: Analyzing an Options Strategy")
    print("-" * 45)
    
    # Define a credit spread strategy
    legs = [
        OptionsLeg(action="sell", option_type="put", strike=455.0, premium=3.20, delta=-0.25),
        OptionsLeg(action="buy", option_type="put", strike=450.0, premium=1.60, delta=-0.18)
    ]
    
    print("Strategy: SPY Bull Put Credit Spread")
    print("   Sell 455 Put @ $3.20")
    print("   Buy 450 Put @ $1.60")
    print("   Expiration: 35 days")
    
    # Get enhanced analysis
    analysis = integration.analyze_strategy_with_emotional_context(
        ticker="SPY",
        strategy_type=StrategyType.CREDIT_SPREAD,
        legs=legs,
        expiration_date=datetime.now() + timedelta(days=35),
        user_mood="optimistic",
        risk_preference="moderate"
    )
    
    tech = analysis["technical_analysis"]
    companion = analysis["companion_perspective"]
    goals = analysis["goal_integration"]
    
    print(f"\n📈 Technical Analysis:")
    print(f"   Credit Received: ${abs(tech['entry_cost']):.2f}")
    print(f"   Maximum Gain: ${tech['max_gain']:.2f}")
    print(f"   Maximum Loss: ${tech['max_loss']:.2f}")
    print(f"   Breakeven: ${tech['breakeven_points'][0]:.2f}")
    print(f"   Win Probability: {tech['probability_of_profit']:.1%}")
    print(f"   Risk Level: {tech['risk_level'].title()}")
    
    print(f"\n🤖 Companion Perspective:")
    print(f"   Emotional Tone: {companion['emotional_tone'].title()}")
    print(f"   Plain English: {companion['plain_english_review']}")
    print(f"   Advice: {companion['companion_advice']}")
    
    print(f"\n🎯 Goal Integration:")
    if goals['has_active_goals']:
        print(f"   Goal Impact: {goals['suggestion']}")
        top_goal = goals['top_goal']
        print(f"   Top Goal: {top_goal['name']} ({top_goal['progress']:.1f}% complete)")
    
    # Demo 3: Process Trade Result
    print(f"\n\n💰 STEP 3: Processing Trade Results")
    print("-" * 45)
    
    print("⏰ Fast-forward 20 days...")
    print("💡 Trade update: Closing position early at 60% of max profit")
    
    # Simulate profitable trade
    trade_response = integration.process_trade_result_with_goals(
        strategy_id=tech["strategy_id"],
        exit_value=64.0,  # Keep $64 of the $160 credit
        notes="Closed early at 60% profit target - taking gains"
    )
    
    result = trade_response["trade_result"]
    
    print(f"\n📊 Trade Results:")
    print(f"   Outcome: {result['outcome'].upper()} 🎉")
    print(f"   Profit: ${result['profit_loss']:.2f}")
    print(f"   Return: {result['profit_percentage']:.1f}%")
    print(f"   Days Held: {result['days_held']}")
    
    print(f"\n🤖 Companion Response:")
    print(f"   \"{trade_response['companion_response']}\"")
    
    # Handle goal allocation
    if trade_response["goal_allocation"]:
        allocation = trade_response["goal_allocation"]
        print(f"\n🎯 Goal Allocation Suggestions:")
        print(f"   Available Profit: ${allocation['profit_available']:.2f}")
        print(f"   🤖 \"{allocation['companion_perspective']}\"")
        
        print(f"\n   💡 Smart Allocation Suggestions:")
        for i, suggestion in enumerate(allocation["suggestions"][:3], 1):
            status = "COMPLETES GOAL! 🎉" if suggestion.get("would_complete") else f"→ {suggestion['new_progress']:.1f}%"
            print(f"   {i}. {suggestion['goal_name']}: ${suggestion['suggested_amount']:.2f} ({status})")
        
        # Auto-allocate to top suggestion
        if allocation["suggestions"]:
            top_suggestion = allocation["suggestions"][0]
            allocation_result = integration.goals_tracker.add_contribution(
                goal_id=top_suggestion["goal_id"],
                amount=top_suggestion["suggested_amount"],
                source="SPY_credit_spread",
                notes="Profitable options trade - building toward dreams!"
            )
            
            print(f"\n   ✅ Auto-allocated ${allocation_result['contribution_amount']:.2f} to {allocation_result['goal_name']}")
            print(f"      New Progress: {allocation_result['progress_percentage']:.1f}%")
            print(f"      🤖 \"{allocation_result['encouragement']}\"")
            
            if allocation_result.get('milestone_message'):
                print(f"      🎉 MILESTONE: {allocation_result['milestone_message']}")
    
    # Demo 4: Investment Guidance
    print(f"\n\n🧭 STEP 4: Comprehensive Investment Guidance")
    print("-" * 45)
    
    guidance = integration.get_investment_guidance(recent_days=30)
    
    print(f"📊 Performance Review:")
    print(f"   {guidance['performance_review']}")
    
    print(f"\n🎯 Goals Progress:")
    print(f"   {guidance['goals_progress']}")
    
    print(f"\n🤖 Companion State:")
    print(f"   Mood: {guidance['companion_mood'].title()}")
    print(f"   Message: \"{guidance['encouragement']}\"")
    
    print(f"\n📋 Next Steps:")
    for i, step in enumerate(guidance['next_steps'], 1):
        print(f"   {i}. {step}")
    
    # Demo 5: Goals Summary
    print(f"\n\n🏆 STEP 5: Goals Achievement Summary")
    print("-" * 45)
    
    goals_summary = integration.goals_tracker.get_goals_summary()
    
    print(f"📈 Overall Progress:")
    print(f"   Total Goals: {goals_summary['total_goals']}")
    print(f"   Active Goals: {goals_summary['active_goals']}")
    print(f"   Overall Progress: {goals_summary['overall_progress']:.1f}%")
    print(f"   Total Target: ${goals_summary['total_target_amount']:,.2f}")
    print(f"   Current Savings: ${goals_summary['total_current_amount']:.2f}")
    
    print(f"\n🎯 Individual Goal Status:")
    for goal in goals_summary["goals"]:
        progress_bar = "█" * int(goal['progress_percentage'] / 5) + "░" * (20 - int(goal['progress_percentage'] / 5))
        print(f"   {goal['name']}:")
        print(f"   [{progress_bar}] {goal['progress_percentage']:.1f}%")
        print(f"   ${goal['current_amount']:.2f} / ${goal['target_amount']:,.2f} (${goal['remaining_amount']:,.2f} remaining)")
        print()
    
    # Conclusion
    print("=" * 55)
    print("🎉 DEMO COMPLETE!")
    print("=" * 55)
    print("✨ What you've seen:")
    print("   • AI-powered options strategy analysis")
    print("   • Emotional companion guidance and support")
    print("   • Goal-oriented profit allocation")
    print("   • Milestone tracking and celebration")
    print("   • Comprehensive investment guidance")
    print()
    print("🚀 This system is now ready for:")
    print("   • Real trading with paper money practice")
    print("   • Integration with broker APIs")
    print("   • Advanced strategy development")
    print("   • Portfolio-wide risk management")
    print()
    print("🤖 Your AI companion is here to help you build wealth")
    print("   while maintaining emotional well-being and clear purpose.")
    print("   Every trade serves your dreams. Every profit builds your future.")
    print("\n💡 Ready to start your investment journey? Let's do this together! 🎯")


if __name__ == "__main__":
    run_investment_demo()

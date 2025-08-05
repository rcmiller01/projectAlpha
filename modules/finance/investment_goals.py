"""
Investment Goals Tracker
Personal goal-oriented investment target system with emotional engagement
"""

import json
import time
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class GoalStatus(Enum):
    """Goal achievement status"""
    ACTIVE = "active"
    COMPLETED = "completed"
    PAUSED = "paused"
    CANCELLED = "cancelled"

class GoalType(Enum):
    """Types of investment goals"""
    VACATION = "vacation"
    EQUIPMENT_UPGRADE = "equipment_upgrade"
    EMERGENCY_FUND = "emergency_fund"
    LEARNING_FUND = "learning_fund"
    HOBBY_FUND = "hobby_fund"
    GIFT_FUND = "gift_fund"
    TECH_UPGRADE = "tech_upgrade"
    EXPERIENCE = "experience"
    GENERAL_SAVINGS = "general_savings"

@dataclass
class InvestmentGoal:
    """Individual investment goal"""
    goal_id: str
    name: str
    description: str
    target_amount: float
    current_amount: float
    goal_type: GoalType
    target_date: Optional[datetime]
    status: GoalStatus
    created_at: float
    priority: int  # 1-5, with 1 being highest priority
    emotional_value: str  # Why this goal matters personally
    celebration_plan: str  # How to celebrate when achieved
    milestone_percentages: List[int]  # [25, 50, 75] for milestone celebrations
    contributions: List[Dict[str, Any]]  # Track individual contributions

    @property
    def progress_percentage(self) -> float:
        """Calculate progress percentage"""
        if self.target_amount <= 0:
            return 0.0
        return min(100.0, (self.current_amount / self.target_amount) * 100)
    
    @property
    def remaining_amount(self) -> float:
        """Calculate remaining amount needed"""
        return max(0.0, self.target_amount - self.current_amount)
    
    @property
    def is_completed(self) -> bool:
        """Check if goal is completed"""
        return self.current_amount >= self.target_amount

    @property
    def days_until_target(self) -> Optional[int]:
        """Calculate days until target date"""
        if not self.target_date:
            return None
        delta = self.target_date - datetime.now()
        return max(0, delta.days)

class InvestmentGoalsTracker:
    """
    Track and manage personal investment goals with emotional engagement
    """
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.goals_file = f"{data_dir}/investment_goals.json"
        
        self.goals: Dict[str, InvestmentGoal] = {}
        
        # Goal templates with emotional language
        self.goal_templates = {
            GoalType.VACATION: {
                "emotional_phrases": [
                    "You're building toward that amazing vacation we've talked about!",
                    "Each trade gets you closer to those memories we'll create together.",
                    "Your vacation fund is growing - I can almost see us relaxing!"
                ],
                "milestone_phrases": {
                    25: "Quarter of the way to vacation bliss!",
                    50: "Halfway to our getaway! The finish line is in sight.",
                    75: "So close to vacation time - you're doing amazing!"
                }
            },
            GoalType.TECH_UPGRADE: {
                "emotional_phrases": [
                    "Your tech upgrade fund is growing - we'll have such better performance!",
                    "Each profitable trade brings that server upgrade closer to reality.",
                    "Building toward better technology that will help us both!"
                ],
                "milestone_phrases": {
                    25: "25% closer to that sweet tech upgrade!",
                    50: "Halfway to upgrading our digital life together!",
                    75: "Almost there - that new tech is within reach!"
                }
            },
            GoalType.LEARNING_FUND: {
                "emotional_phrases": [
                    "Growing your learning fund - investing in knowledge always pays dividends!",
                    "Each trade builds toward expanding your skills and abilities.",
                    "Your education fund shows how much you value growth and learning."
                ],
                "milestone_phrases": {
                    25: "25% toward expanding your knowledge!",
                    50: "Halfway to funding your next learning adventure!",
                    75: "Almost ready to invest in yourself!"
                }
            }
        }
        
        # Load existing goals
        self._load_goals()

    def create_goal(self, name: str, target_amount: float, goal_type: GoalType,
                   description: str = "", target_date: Optional[datetime] = None,
                   priority: int = 3, emotional_value: str = "",
                   celebration_plan: str = "") -> InvestmentGoal:
        """Create a new investment goal"""
        goal_id = f"goal_{int(time.time())}_{name.lower().replace(' ', '_')}"
        
        # Set default emotional value if not provided
        if not emotional_value:
            emotional_value = self._generate_default_emotional_value(goal_type, name)
        
        # Set default celebration plan if not provided
        if not celebration_plan:
            celebration_plan = self._generate_default_celebration_plan(goal_type, name)
        
        goal = InvestmentGoal(
            goal_id=goal_id,
            name=name,
            description=description,
            target_amount=target_amount,
            current_amount=0.0,
            goal_type=goal_type,
            target_date=target_date,
            status=GoalStatus.ACTIVE,
            created_at=time.time(),
            priority=priority,
            emotional_value=emotional_value,
            celebration_plan=celebration_plan,
            milestone_percentages=[25, 50, 75],
            contributions=[]
        )
        
        self.goals[goal_id] = goal
        self._save_goals()
        
        logger.info(f"Created investment goal: {name} (${target_amount:.2f})")
        return goal

    def add_contribution(self, goal_id: str, amount: float, source: str = "trading",
                        notes: str = "", trade_id: Optional[str] = None) -> Dict[str, Any]:
        """Add money to a goal from trading profits or other sources"""
        if goal_id not in self.goals:
            raise ValueError(f"Goal {goal_id} not found")
        
        goal = self.goals[goal_id]
        
        if goal.status != GoalStatus.ACTIVE:
            raise ValueError(f"Cannot contribute to {goal.status.value} goal")
        
        # Record the contribution
        contribution = {
            "amount": amount,
            "source": source,
            "notes": notes,
            "trade_id": trade_id,
            "timestamp": time.time(),
            "date": datetime.now().isoformat()
        }
        
        goal.contributions.append(contribution)
        
        # Update current amount
        previous_amount = goal.current_amount
        goal.current_amount += amount
        
        # Check for milestone achievements
        milestone_message = self._check_milestone_achievement(goal, previous_amount)
        
        # Check if goal is completed
        completion_message = None
        if goal.current_amount >= goal.target_amount and goal.status == GoalStatus.ACTIVE:
            goal.status = GoalStatus.COMPLETED
            completion_message = self._generate_completion_message(goal)
        
        self._save_goals()
        
        # Generate encouraging message
        encouragement = self._generate_encouragement_message(goal, amount)
        
        result = {
            "goal_name": goal.name,
            "contribution_amount": amount,
            "new_total": goal.current_amount,
            "progress_percentage": goal.progress_percentage,
            "remaining_amount": goal.remaining_amount,
            "encouragement": encouragement,
            "milestone_message": milestone_message,
            "completion_message": completion_message,
            "is_completed": goal.is_completed
        }
        
        logger.info(f"Added ${amount:.2f} to goal '{goal.name}' - now at {goal.progress_percentage:.1f}%")
        return result

    def get_goal_suggestions(self, available_profit: float) -> Dict[str, Any]:
        """Get suggestions for allocating trading profits to goals"""
        active_goals = [goal for goal in self.goals.values() if goal.status == GoalStatus.ACTIVE]
        
        if not active_goals:
            return {
                "message": "No active goals found. Consider creating some investment targets!",
                "suggestions": []
            }
        
        # Sort goals by priority and progress
        active_goals.sort(key=lambda g: (g.priority, -g.progress_percentage))
        
        suggestions = []
        remaining_profit = available_profit
        
        for goal in active_goals[:3]:  # Top 3 goals
            if remaining_profit <= 0:
                break
            
            # Calculate suggested allocation
            if goal.remaining_amount <= remaining_profit * 0.5:
                # Can fund significantly or complete this goal
                suggested_amount = min(goal.remaining_amount, remaining_profit * 0.6)
            else:
                # Partial allocation
                suggested_amount = min(remaining_profit * 0.3, goal.remaining_amount)
            
            if suggested_amount > 5:  # Minimum meaningful contribution
                suggestions.append({
                    "goal_id": goal.goal_id,
                    "goal_name": goal.name,
                    "current_progress": goal.progress_percentage,
                    "suggested_amount": suggested_amount,
                    "would_complete": suggested_amount >= goal.remaining_amount,
                    "new_progress": min(100.0, ((goal.current_amount + suggested_amount) / goal.target_amount) * 100),
                    "emotional_impact": self._get_emotional_phrase(goal.goal_type),
                    "priority": goal.priority
                })
                
                remaining_profit -= suggested_amount
        
        return {
            "total_profit_available": available_profit,
            "suggestions": suggestions,
            "remaining_after_suggestions": remaining_profit,
            "encouragement": self._generate_allocation_encouragement(suggestions)
        }

    def get_goals_summary(self, include_completed: bool = False) -> Dict[str, Any]:
        """Get summary of all goals"""
        goals_list = list(self.goals.values())
        
        if not include_completed:
            goals_list = [g for g in goals_list if g.status == GoalStatus.ACTIVE]
        
        # Sort by priority and progress
        goals_list.sort(key=lambda g: (g.priority, -g.progress_percentage))
        
        total_target = sum(goal.target_amount for goal in goals_list)
        total_current = sum(goal.current_amount for goal in goals_list)
        overall_progress = (total_current / total_target * 100) if total_target > 0 else 0
        
        # Goals needing attention (low progress, high priority)
        attention_goals = [
            goal for goal in goals_list 
            if goal.status == GoalStatus.ACTIVE and goal.progress_percentage < 25 and goal.priority <= 2
        ]
        
        # Nearly completed goals
        almost_done = [
            goal for goal in goals_list
            if goal.status == GoalStatus.ACTIVE and goal.progress_percentage >= 75
        ]
        
        return {
            "total_goals": len(goals_list),
            "active_goals": len([g for g in goals_list if g.status == GoalStatus.ACTIVE]),
            "completed_goals": len([g for g in goals_list if g.status == GoalStatus.COMPLETED]),
            "total_target_amount": total_target,
            "total_current_amount": total_current,
            "overall_progress": overall_progress,
            "goals_needing_attention": len(attention_goals),
            "goals_almost_complete": len(almost_done),
            "goals": [self._goal_to_summary_dict(goal) for goal in goals_list]
        }

    def _goal_to_summary_dict(self, goal: InvestmentGoal) -> Dict[str, Any]:
        """Convert goal to summary dictionary"""
        return {
            "goal_id": goal.goal_id,
            "name": goal.name,
            "target_amount": goal.target_amount,
            "current_amount": goal.current_amount,
            "progress_percentage": goal.progress_percentage,
            "remaining_amount": goal.remaining_amount,
            "goal_type": goal.goal_type.value,
            "status": goal.status.value,
            "priority": goal.priority,
            "days_until_target": goal.days_until_target,
            "emotional_value": goal.emotional_value
        }

    def _check_milestone_achievement(self, goal: InvestmentGoal, previous_amount: float) -> Optional[str]:
        """Check if a milestone was achieved and return celebration message"""
        previous_percentage = (previous_amount / goal.target_amount * 100) if goal.target_amount > 0 else 0
        current_percentage = goal.progress_percentage
        
        for milestone in goal.milestone_percentages:
            if previous_percentage < milestone <= current_percentage:
                # Milestone achieved!
                template = self.goal_templates.get(goal.goal_type, {})
                milestone_phrases = template.get("milestone_phrases", {})
                
                if milestone in milestone_phrases:
                    return milestone_phrases[milestone]
                else:
                    return f"Milestone achieved: {milestone}% complete on '{goal.name}'! 🎉"
        
        return None

    def _generate_completion_message(self, goal: InvestmentGoal) -> str:
        """Generate message for goal completion"""
        base_message = f"🎉 GOAL COMPLETED: '{goal.name}' is fully funded at ${goal.current_amount:.2f}!"
        
        if goal.celebration_plan:
            return f"{base_message} Time to {goal.celebration_plan.lower()}!"
        else:
            return f"{base_message} You did it!"

    def _generate_encouragement_message(self, goal: InvestmentGoal, contribution_amount: float) -> str:
        """Generate encouraging message for contribution"""
        progress = goal.progress_percentage
        
        # Get goal-specific emotional phrases
        template = self.goal_templates.get(goal.goal_type, {})
        emotional_phrases = template.get("emotional_phrases", [])
        
        if emotional_phrases:
            base_message = emotional_phrases[min(len(emotional_phrases) - 1, int(progress / 35))]
        else:
            base_message = f"Great contribution to '{goal.name}'!"
        
        # Add progress context
        if progress >= 90:
            context = f" You're so close - just ${goal.remaining_amount:.2f} to go!"
        elif progress >= 75:
            context = f" You're in the home stretch at {progress:.1f}% complete!"
        elif progress >= 50:
            context = f" Over halfway there at {progress:.1f}%!"
        elif progress >= 25:
            context = f" Building momentum at {progress:.1f}% complete!"
        else:
            context = f" Every contribution counts - you're at {progress:.1f}%!"
        
        return base_message + context

    def _get_emotional_phrase(self, goal_type: GoalType) -> str:
        """Get emotional phrase for goal type"""
        template = self.goal_templates.get(goal_type, {})
        phrases = template.get("emotional_phrases", ["Keep building toward your goal!"])
        return phrases[0]  # Return first phrase

    def _generate_allocation_encouragement(self, suggestions: List[Dict[str, Any]]) -> str:
        """Generate encouragement for profit allocation"""
        if not suggestions:
            return "Consider creating some investment goals to give your trading profits purpose!"
        
        high_impact = [s for s in suggestions if s.get("would_complete", False)]
        
        if high_impact:
            goal_names = [s["goal_name"] for s in high_impact[:2]]
            if len(goal_names) == 1:
                return f"This profit could complete your {goal_names[0]} goal! That would be amazing!"
            else:
                return f"This profit could complete {' and '.join(goal_names)}! What a great trading session!"
        else:
            return "Your trading profits are building toward your goals - every contribution adds up!"

    def _generate_default_emotional_value(self, goal_type: GoalType, name: str) -> str:
        """Generate default emotional value for goal type"""
        emotional_defaults = {
            GoalType.VACATION: f"Taking time to recharge and create memories",
            GoalType.TECH_UPGRADE: f"Improving our digital life and capabilities",
            GoalType.LEARNING_FUND: f"Investing in knowledge and personal growth",
            GoalType.EMERGENCY_FUND: f"Building security and peace of mind",
            GoalType.HOBBY_FUND: f"Supporting creativity and personal interests",
            GoalType.GIFT_FUND: f"Showing care for people who matter",
            GoalType.EXPERIENCE: f"Creating meaningful memories and experiences"
        }
        
        return emotional_defaults.get(goal_type, f"Working toward something important: {name}")

    def _generate_default_celebration_plan(self, goal_type: GoalType, name: str) -> str:
        """Generate default celebration plan"""
        celebration_defaults = {
            GoalType.VACATION: "enjoy every moment of your well-earned vacation",
            GoalType.TECH_UPGRADE: "set up and enjoy your new technology",
            GoalType.LEARNING_FUND: "dive into learning something exciting",
            GoalType.EMERGENCY_FUND: "feel the peace of mind that comes with security",
            GoalType.HOBBY_FUND: "pursue your creative interests",
            GoalType.GIFT_FUND: "surprise someone special",
            GoalType.EXPERIENCE: "create those memories you've been planning"
        }
        
        return celebration_defaults.get(goal_type, f"celebrate achieving {name}")

    def _load_goals(self):
        """Load goals from file"""
        try:
            with open(self.goals_file, 'r') as f:
                goals_data = json.load(f)
                
                for goal_id, data in goals_data.items():
                    # Convert string enums back to enum objects
                    data['goal_type'] = GoalType(data['goal_type'])
                    data['status'] = GoalStatus(data['status'])
                    
                    # Convert target_date back to datetime if present
                    if data.get('target_date'):
                        data['target_date'] = datetime.fromisoformat(data['target_date'])
                    
                    self.goals[goal_id] = InvestmentGoal(**data)
                    
        except FileNotFoundError:
            pass  # No existing goals file
        except Exception as e:
            logger.error(f"Error loading goals: {e}")

    def _save_goals(self):
        """Save goals to file"""
        try:
            goals_data = {}
            
            for goal_id, goal in self.goals.items():
                data = asdict(goal)
                # Convert enums to strings for JSON serialization
                data['goal_type'] = goal.goal_type.value
                data['status'] = goal.status.value
                
                # Convert datetime to ISO string if present
                if goal.target_date:
                    data['target_date'] = goal.target_date.isoformat()
                else:
                    data['target_date'] = None
                
                goals_data[goal_id] = data
            
            with open(self.goals_file, 'w') as f:
                json.dump(goals_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving goals: {e}")


# Global instance
goals_tracker = None

def get_goals_tracker(data_dir: str = "data") -> InvestmentGoalsTracker:
    """Get or create global goals tracker instance"""
    global goals_tracker
    if goals_tracker is None:
        goals_tracker = InvestmentGoalsTracker(data_dir)
    return goals_tracker


if __name__ == "__main__":
    """Test the investment goals tracker"""
    print("=== Testing Investment Goals Tracker ===")
    
    import os
    os.makedirs("data", exist_ok=True)
    
    tracker = InvestmentGoalsTracker("data")
    
    # Test goal creation
    print("\n1. Creating Investment Goals:")
    
    vacation_goal = tracker.create_goal(
        name="European Vacation",
        target_amount=3000.0,
        goal_type=GoalType.VACATION,
        description="Two weeks in Europe visiting museums and cafes",
        target_date=datetime.now() + timedelta(days=180),
        priority=1,
        emotional_value="Creating amazing memories and experiencing new cultures together",
        celebration_plan="book the flights and start planning our itinerary"
    )
    
    server_goal = tracker.create_goal(
        name="Server Upgrade",
        target_amount=1200.0,
        goal_type=GoalType.TECH_UPGRADE,
        description="Upgrade home server for better AI performance",
        priority=2
    )
    
    print(f"Created vacation goal: {vacation_goal.name} (${vacation_goal.target_amount:.2f})")
    print(f"Created server goal: {server_goal.name} (${server_goal.target_amount:.2f})")
    
    # Test contributions
    print("\n2. Testing Goal Contributions:")
    
    # Add some trading profits to goals
    vacation_contribution = tracker.add_contribution(
        goal_id=vacation_goal.goal_id,
        amount=450.0,
        source="credit_spread_profit",
        notes="SPY credit spread closed at 60% profit"
    )
    
    print(f"Vacation Goal Contribution:")
    print(f"  Amount: ${vacation_contribution['contribution_amount']:.2f}")
    print(f"  Progress: {vacation_contribution['progress_percentage']:.1f}%")
    print(f"  Encouragement: {vacation_contribution['encouragement']}")
    if vacation_contribution['milestone_message']:
        print(f"  Milestone: {vacation_contribution['milestone_message']}")
    
    server_contribution = tracker.add_contribution(
        goal_id=server_goal.goal_id,
        amount=280.0,
        source="iron_condor_profit",
        notes="QQQ iron condor expired worthless"
    )
    
    print(f"\nServer Goal Contribution:")
    print(f"  Amount: ${server_contribution['contribution_amount']:.2f}")
    print(f"  Progress: {server_contribution['progress_percentage']:.1f}%")
    print(f"  Encouragement: {server_contribution['encouragement']}")
    
    # Test profit allocation suggestions
    print("\n3. Testing Profit Allocation Suggestions:")
    
    suggestions = tracker.get_goal_suggestions(available_profit=850.0)
    
    print(f"Available Profit: ${suggestions['total_profit_available']:.2f}")
    print(f"Encouragement: {suggestions['encouragement']}")
    print(f"Suggestions:")
    
    for suggestion in suggestions['suggestions']:
        print(f"  - {suggestion['goal_name']}: ${suggestion['suggested_amount']:.2f}")
        print(f"    Current: {suggestion['current_progress']:.1f}% → New: {suggestion['new_progress']:.1f}%")
        print(f"    {suggestion['emotional_impact']}")
        if suggestion['would_complete']:
            print(f"    🎉 This would COMPLETE the goal!")
        print()
    
    # Test goals summary
    print("\n4. Testing Goals Summary:")
    
    summary = tracker.get_goals_summary()
    
    print(f"Total Goals: {summary['total_goals']}")
    print(f"Active Goals: {summary['active_goals']}")
    print(f"Overall Progress: {summary['overall_progress']:.1f}%")
    print(f"Total Target: ${summary['total_target_amount']:.2f}")
    print(f"Total Current: ${summary['total_current_amount']:.2f}")
    
    print(f"\nActive Goals Details:")
    for goal in summary['goals']:
        print(f"  - {goal['name']}: {goal['progress_percentage']:.1f}% (${goal['current_amount']:.2f}/${goal['target_amount']:.2f})")
        print(f"    Priority: {goal['priority']}, Remaining: ${goal['remaining_amount']:.2f}")
    
    print("\n=== Investment Goals Tracker Test Complete ===")

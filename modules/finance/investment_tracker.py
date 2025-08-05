"""
Tactical Investment Strategy Tracker
Emotionally-aware options strategy analysis and goal tracking
"""

import json
import time
import math
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class StrategyType(Enum):
    """Supported options strategy types"""
    CREDIT_SPREAD = "credit_spread"
    DEBIT_SPREAD = "debit_spread"
    IRON_CONDOR = "iron_condor"
    IRON_BUTTERFLY = "iron_butterfly"
    COVERED_CALL = "covered_call"
    CASH_SECURED_PUT = "cash_secured_put"
    STRADDLE = "straddle"
    STRANGLE = "strangle"

class RiskLevel(Enum):
    """Risk assessment levels"""
    VERY_LOW = "very_low"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"

@dataclass
class OptionsLeg:
    """Individual options contract in a strategy"""
    action: str  # "buy" or "sell"
    option_type: str  # "call" or "put"
    strike: float
    premium: float
    quantity: int = 1
    delta: float = 0.0
    implied_volatility: float = 0.0

@dataclass
class StrategyAnalysis:
    """Complete analysis of an options strategy"""
    strategy_id: str
    ticker: str
    strategy_type: StrategyType
    legs: List[OptionsLeg]
    expiration_date: datetime
    entry_cost: float
    max_gain: float
    max_loss: float
    breakeven_points: List[float]
    probability_of_profit: float
    risk_level: RiskLevel
    plain_english_review: str
    recommendation: str
    emotional_context: str
    created_at: float

@dataclass
class TradeResult:
    """Result of a completed trade"""
    strategy_id: str
    ticker: str
    strategy_type: str
    entry_date: datetime
    exit_date: datetime
    entry_cost: float
    exit_value: float
    profit_loss: float
    profit_percentage: float
    days_held: int
    outcome: str  # "win", "loss", "breakeven"
    notes: str = ""
    emotional_impact: str = ""

class InvestmentTracker:
    """
    Tactical investment strategy tracker with emotional awareness
    """
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.strategies_file = f"{data_dir}/investment_strategies.json"
        self.journal_file = f"{data_dir}/investment_journal.json"
        
        self.strategies: Dict[str, StrategyAnalysis] = {}
        self.trade_history: List[TradeResult] = []
        
        # Strategy templates and risk parameters
        self.strategy_templates = {
            StrategyType.CREDIT_SPREAD: {
                "description": "Collect premium by selling closer-to-money option and buying further out-of-money option",
                "risk_profile": "Limited risk, limited reward",
                "best_conditions": "High IV, neutral to directional bias",
                "typical_pop": 0.65  # Probability of profit
            },
            StrategyType.IRON_CONDOR: {
                "description": "Sell both call and put spreads for premium collection in range-bound market",
                "risk_profile": "Limited risk, limited reward",
                "best_conditions": "High IV, expect sideways movement",
                "typical_pop": 0.55
            },
            StrategyType.COVERED_CALL: {
                "description": "Own stock and sell call option for additional income",
                "risk_profile": "Limited upside, stock downside risk",
                "best_conditions": "Neutral to slightly bullish, own the stock",
                "typical_pop": 0.70
            }
        }
        
        # Load existing data
        self._load_data()

    def analyze_strategy(self, ticker: str, strategy_type: StrategyType,
                        legs: List[OptionsLeg], expiration_date: datetime,
                        user_context: Dict[str, Any] = None) -> StrategyAnalysis:
        """
        Analyze an options strategy and provide comprehensive assessment
        """
        strategy_id = f"{ticker}_{strategy_type.value}_{int(time.time())}"
        
        # Calculate basic metrics
        entry_cost = self._calculate_entry_cost(legs)
        max_gain, max_loss = self._calculate_max_gain_loss(strategy_type, legs)
        breakeven_points = self._calculate_breakeven_points(strategy_type, legs)
        probability_of_profit = self._estimate_probability_of_profit(strategy_type, legs)
        risk_level = self._assess_risk_level(strategy_type, max_loss, entry_cost)
        
        # Generate plain English review and recommendation
        plain_english_review = self._generate_plain_english_review(
            ticker, strategy_type, legs, max_gain, max_loss, probability_of_profit
        )
        
        recommendation = self._generate_recommendation(
            strategy_type, risk_level, probability_of_profit, user_context
        )
        
        emotional_context = self._generate_emotional_context(
            risk_level, max_gain, max_loss, user_context
        )
        
        analysis = StrategyAnalysis(
            strategy_id=strategy_id,
            ticker=ticker,
            strategy_type=strategy_type,
            legs=legs,
            expiration_date=expiration_date,
            entry_cost=entry_cost,
            max_gain=max_gain,
            max_loss=max_loss,
            breakeven_points=breakeven_points,
            probability_of_profit=probability_of_profit,
            risk_level=risk_level,
            plain_english_review=plain_english_review,
            recommendation=recommendation,
            emotional_context=emotional_context,
            created_at=time.time()
        )
        
        # Store the analysis
        self.strategies[strategy_id] = analysis
        self._save_data()
        
        logger.info(f"Analyzed {strategy_type.value} strategy for {ticker}")
        return analysis

    def _calculate_entry_cost(self, legs: List[OptionsLeg]) -> float:
        """Calculate net entry cost/credit for the strategy"""
        total_cost = 0.0
        
        for leg in legs:
            if leg.action == "buy":
                total_cost += leg.premium * leg.quantity * 100  # Options are per 100 shares
            else:  # sell
                total_cost -= leg.premium * leg.quantity * 100
        
        return total_cost

    def _calculate_max_gain_loss(self, strategy_type: StrategyType, 
                                legs: List[OptionsLeg]) -> Tuple[float, float]:
        """Calculate maximum gain and loss potential"""
        if strategy_type == StrategyType.CREDIT_SPREAD:
            return self._calc_credit_spread_max_gain_loss(legs)
        elif strategy_type == StrategyType.IRON_CONDOR:
            return self._calc_iron_condor_max_gain_loss(legs)
        elif strategy_type == StrategyType.COVERED_CALL:
            return self._calc_covered_call_max_gain_loss(legs)
        else:
            # Generic calculation for other strategies
            entry_cost = self._calculate_entry_cost(legs)
            if entry_cost < 0:  # Credit strategy
                return abs(entry_cost), self._calc_max_width(legs) * 100 + entry_cost
            else:  # Debit strategy
                max_width = self._calc_max_width(legs) * 100
                return max_width - entry_cost, entry_cost

    def _calc_credit_spread_max_gain_loss(self, legs: List[OptionsLeg]) -> Tuple[float, float]:
        """Calculate max gain/loss for credit spreads"""
        if len(legs) != 2:
            raise ValueError("Credit spread must have exactly 2 legs")
        
        # Sort legs by strike
        sorted_legs = sorted(legs, key=lambda x: x.strike)
        
        # Credit received is max gain
        credit = abs(self._calculate_entry_cost(legs))
        
        # Max loss is width of strikes minus credit
        strike_width = abs(sorted_legs[1].strike - sorted_legs[0].strike) * 100
        max_loss = strike_width - credit
        
        return credit, max_loss

    def _calc_iron_condor_max_gain_loss(self, legs: List[OptionsLeg]) -> Tuple[float, float]:
        """Calculate max gain/loss for iron condor"""
        if len(legs) != 4:
            raise ValueError("Iron condor must have exactly 4 legs")
        
        # Credit received is max gain
        credit = abs(self._calculate_entry_cost(legs))
        
        # Find the widest spread
        strikes = [leg.strike for leg in legs]
        strikes.sort()
        
        # Assume symmetric condor - use call spread width
        call_width = strikes[3] - strikes[2]  # Highest strikes
        put_width = strikes[1] - strikes[0]   # Lowest strikes
        
        max_width = max(call_width, put_width) * 100
        max_loss = max_width - credit
        
        return credit, max_loss

    def _calc_covered_call_max_gain_loss(self, legs: List[OptionsLeg]) -> Tuple[float, float]:
        """Calculate max gain/loss for covered call"""
        # Simplified - assumes we already own the stock
        call_leg = next((leg for leg in legs if leg.option_type == "call"), None)
        
        if not call_leg:
            raise ValueError("Covered call must include a call option")
        
        # Max gain is premium received (simplified)
        max_gain = call_leg.premium * call_leg.quantity * 100
        
        # Max loss is theoretically unlimited (stock could go to zero)
        # But we'll use a practical estimate
        max_loss = float('inf')  # Will be handled specially in risk assessment
        
        return max_gain, max_loss

    def _calc_max_width(self, legs: List[OptionsLeg]) -> float:
        """Calculate maximum strike width in the strategy"""
        strikes = [leg.strike for leg in legs]
        return max(strikes) - min(strikes)

    def _calculate_breakeven_points(self, strategy_type: StrategyType, 
                                  legs: List[OptionsLeg]) -> List[float]:
        """Calculate breakeven points for the strategy"""
        if strategy_type == StrategyType.CREDIT_SPREAD:
            return self._calc_credit_spread_breakeven(legs)
        elif strategy_type == StrategyType.IRON_CONDOR:
            return self._calc_iron_condor_breakeven(legs)
        else:
            # Generic calculation
            return []

    def _calc_credit_spread_breakeven(self, legs: List[OptionsLeg]) -> List[float]:
        """Calculate breakeven for credit spread"""
        sorted_legs = sorted(legs, key=lambda x: x.strike)
        credit_per_share = abs(self._calculate_entry_cost(legs)) / 100
        
        # For call credit spread: short strike + credit
        # For put credit spread: short strike - credit
        short_leg = next((leg for leg in legs if leg.action == "sell"), None)
        
        if short_leg.option_type == "call":
            return [short_leg.strike + credit_per_share]
        else:  # put
            return [short_leg.strike - credit_per_share]

    def _calc_iron_condor_breakeven(self, legs: List[OptionsLeg]) -> List[float]:
        """Calculate breakeven points for iron condor"""
        credit_per_share = abs(self._calculate_entry_cost(legs)) / 100
        
        # Find short strikes
        short_legs = [leg for leg in legs if leg.action == "sell"]
        
        if len(short_legs) != 2:
            return []
        
        short_call = next((leg for leg in short_legs if leg.option_type == "call"), None)
        short_put = next((leg for leg in short_legs if leg.option_type == "put"), None)
        
        if short_call and short_put:
            return [
                short_put.strike - credit_per_share,    # Lower breakeven
                short_call.strike + credit_per_share     # Upper breakeven
            ]
        
        return []

    def _estimate_probability_of_profit(self, strategy_type: StrategyType, 
                                      legs: List[OptionsLeg]) -> float:
        """Estimate probability of profit based on deltas and strategy type"""
        # Use deltas if available, otherwise use typical values
        template = self.strategy_templates.get(strategy_type)
        base_prob = template.get("typical_pop", 0.50) if template else 0.50
        
        # Adjust based on deltas if available
        short_legs = [leg for leg in legs if leg.action == "sell"]
        
        if short_legs and all(leg.delta != 0.0 for leg in short_legs):
            # Use delta to estimate probability
            # For credit spreads: higher short delta = lower POP
            avg_short_delta = sum(abs(leg.delta) for leg in short_legs) / len(short_legs)
            
            # Adjust probability based on delta
            if avg_short_delta > 0.30:  # Deep ITM
                return base_prob * 0.8
            elif avg_short_delta < 0.10:  # Far OTM
                return base_prob * 1.2
        
        return min(0.95, base_prob)  # Cap at 95%

    def _assess_risk_level(self, strategy_type: StrategyType, 
                          max_loss: float, entry_cost: float) -> RiskLevel:
        """Assess the risk level of the strategy"""
        if max_loss == float('inf'):
            return RiskLevel.VERY_HIGH
        
        # Risk/reward ratio
        if entry_cost != 0:
            risk_reward_ratio = abs(max_loss / abs(entry_cost))
        else:
            risk_reward_ratio = 1.0
        
        # Assess based on strategy type and metrics
        if strategy_type in [StrategyType.CREDIT_SPREAD, StrategyType.IRON_CONDOR]:
            if risk_reward_ratio <= 2.0:
                return RiskLevel.LOW
            elif risk_reward_ratio <= 4.0:
                return RiskLevel.MODERATE
            else:
                return RiskLevel.HIGH
        elif strategy_type == StrategyType.COVERED_CALL:
            return RiskLevel.MODERATE  # Stock ownership risk
        else:
            return RiskLevel.MODERATE

    def _generate_plain_english_review(self, ticker: str, strategy_type: StrategyType,
                                     legs: List[OptionsLeg], max_gain: float,
                                     max_loss: float, probability_of_profit: float) -> str:
        """Generate plain English explanation of the strategy"""
        template = self.strategy_templates.get(strategy_type)
        
        # Build the review
        review_parts = []
        
        # Strategy description
        if template:
            review_parts.append(f"This {strategy_type.value.replace('_', ' ')} strategy on {ticker} involves {template['description'].lower()}.")
        
        # Financial metrics
        if max_gain != float('inf') and max_loss != float('inf'):
            review_parts.append(f"Your maximum potential gain is ${max_gain:.2f} and maximum risk is ${max_loss:.2f}.")
        
        # Probability assessment
        prob_text = "excellent" if probability_of_profit > 0.70 else "good" if probability_of_profit > 0.60 else "moderate" if probability_of_profit > 0.50 else "challenging"
        review_parts.append(f"The probability of profit is {probability_of_profit:.1%}, which gives you {prob_text} odds of success.")
        
        # Risk context
        if template:
            review_parts.append(f"This strategy works best when {template['best_conditions'].lower()}.")
        
        return " ".join(review_parts)

    def _generate_recommendation(self, strategy_type: StrategyType, risk_level: RiskLevel,
                               probability_of_profit: float, user_context: Dict[str, Any] = None) -> str:
        """Generate personalized recommendation"""
        recommendations = []
        
        # Risk-based recommendations
        if risk_level in [RiskLevel.VERY_LOW, RiskLevel.LOW]:
            recommendations.append("This is a relatively safe strategy that aligns well with conservative risk management.")
        elif risk_level == RiskLevel.MODERATE:
            recommendations.append("This strategy has moderate risk - appropriate if you're comfortable with the potential loss amount.")
        else:
            recommendations.append("This is a higher-risk strategy. Only consider if you can afford the maximum loss and have experience.")
        
        # Probability-based recommendations
        if probability_of_profit > 0.65:
            recommendations.append("The odds are in your favor with this setup.")
        elif probability_of_profit < 0.50:
            recommendations.append("The probability suggests this is a more speculative play.")
        
        # User context considerations
        if user_context:
            experience_level = user_context.get("experience_level", "beginner")
            risk_tolerance = user_context.get("risk_tolerance", "moderate")
            
            if experience_level == "beginner" and risk_level != RiskLevel.LOW:
                recommendations.append("As you're building experience, consider starting with lower-risk strategies.")
            
            if risk_tolerance == "conservative" and risk_level in [RiskLevel.HIGH, RiskLevel.VERY_HIGH]:
                recommendations.append("This strategy may not align with your conservative risk tolerance.")
        
        return " ".join(recommendations)

    def _generate_emotional_context(self, risk_level: RiskLevel, max_gain: float,
                                  max_loss: float, user_context: Dict[str, Any] = None) -> str:
        """Generate emotionally aware context and encouragement"""
        emotional_phrases = []
        
        # Risk comfort language
        if risk_level in [RiskLevel.VERY_LOW, RiskLevel.LOW]:
            emotional_phrases.append("This feels like a comfortable play that should let you sleep well at night.")
        elif risk_level == RiskLevel.MODERATE:
            emotional_phrases.append("There's a balance of opportunity and risk here that could work well for building confidence.")
        else:
            emotional_phrases.append("This is more aggressive - make sure you're emotionally prepared for the potential outcomes.")
        
        # Gain/loss perspective
        if max_gain != float('inf') and max_loss != float('inf'):
            ratio = max_gain / max_loss if max_loss > 0 else 0
            if ratio > 0.5:
                emotional_phrases.append("The potential reward makes the risk feel worthwhile.")
            else:
                emotional_phrases.append("The risk is higher than the reward, so this is more about probability than big gains.")
        
        # Encouraging context
        if user_context and user_context.get("building_confidence"):
            emotional_phrases.append("This could be a good learning experience to build your options trading confidence.")
        
        return " ".join(emotional_phrases)

    def log_trade_result(self, strategy_id: str, exit_value: float, 
                        exit_date: Optional[datetime] = None, notes: str = "") -> TradeResult:
        """Log the result of a completed trade"""
        if strategy_id not in self.strategies:
            raise ValueError(f"Strategy {strategy_id} not found")
        
        strategy = self.strategies[strategy_id]
        exit_date = exit_date or datetime.now()
        
        # Calculate results
        profit_loss = exit_value - strategy.entry_cost
        profit_percentage = (profit_loss / abs(strategy.entry_cost)) * 100 if strategy.entry_cost != 0 else 0
        
        entry_date = datetime.fromtimestamp(strategy.created_at)
        days_held = (exit_date - entry_date).days
        
        # Determine outcome
        if profit_loss > 0:
            outcome = "win"
        elif profit_loss < 0:
            outcome = "loss"
        else:
            outcome = "breakeven"
        
        # Generate emotional impact
        emotional_impact = self._generate_emotional_impact(profit_loss, profit_percentage, outcome)
        
        result = TradeResult(
            strategy_id=strategy_id,
            ticker=strategy.ticker,
            strategy_type=strategy.strategy_type.value,
            entry_date=entry_date,
            exit_date=exit_date,
            entry_cost=strategy.entry_cost,
            exit_value=exit_value,
            profit_loss=profit_loss,
            profit_percentage=profit_percentage,
            days_held=days_held,
            outcome=outcome,
            notes=notes,
            emotional_impact=emotional_impact
        )
        
        self.trade_history.append(result)
        self._save_data()
        
        logger.info(f"Logged trade result: {outcome} ${profit_loss:.2f}")
        return result

    def _generate_emotional_impact(self, profit_loss: float, 
                                 profit_percentage: float, outcome: str) -> str:
        """Generate emotional context for trade results"""
        if outcome == "win":
            if profit_percentage > 50:
                return "Excellent work! This was a really strong trade that should build your confidence."
            elif profit_percentage > 20:
                return "Nice win! You executed well and it paid off."
            else:
                return "A solid win - every profitable trade adds to your experience and account."
        elif outcome == "loss":
            if abs(profit_percentage) > 50:
                return "This was a tough loss, but it's part of learning. Analyze what happened and apply it next time."
            elif abs(profit_percentage) > 20:
                return "A loss, but within reasonable bounds. This is why we manage risk carefully."
            else:
                return "A small loss - these happen and are part of the trading process. Stay disciplined."
        else:
            return "Breaking even is actually a win in options trading - you gained experience without losing money."

    def get_performance_summary(self, days: int = 30) -> Dict[str, Any]:
        """Get performance summary for recent period"""
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_trades = [
            trade for trade in self.trade_history 
            if trade.exit_date >= cutoff_date
        ]
        
        if not recent_trades:
            return {"message": "No recent trades to analyze"}
        
        total_trades = len(recent_trades)
        wins = len([t for t in recent_trades if t.outcome == "win"])
        losses = len([t for t in recent_trades if t.outcome == "loss"])
        
        total_pnl = sum(trade.profit_loss for trade in recent_trades)
        win_rate = wins / total_trades if total_trades > 0 else 0
        
        avg_win = sum(t.profit_loss for t in recent_trades if t.outcome == "win") / max(1, wins)
        avg_loss = sum(t.profit_loss for t in recent_trades if t.outcome == "loss") / max(1, losses)
        
        return {
            "period_days": days,
            "total_trades": total_trades,
            "wins": wins,
            "losses": losses,
            "win_rate": win_rate,
            "total_pnl": total_pnl,
            "average_win": avg_win,
            "average_loss": avg_loss,
            "largest_win": max((t.profit_loss for t in recent_trades if t.outcome == "win"), default=0),
            "largest_loss": min((t.profit_loss for t in recent_trades if t.outcome == "loss"), default=0)
        }

    def _load_data(self):
        """Load investment data from files"""
        try:
            # Load strategies
            try:
                with open(self.strategies_file, 'r') as f:
                    strategies_data = json.load(f)
                    for strategy_id, data in strategies_data.items():
                        # Convert legs back to OptionsLeg objects
                        legs = [OptionsLeg(**leg) for leg in data['legs']]
                        data['legs'] = legs
                        data['strategy_type'] = StrategyType(data['strategy_type'])
                        data['risk_level'] = RiskLevel(data['risk_level'])
                        data['expiration_date'] = datetime.fromisoformat(data['expiration_date'])
                        
                        self.strategies[strategy_id] = StrategyAnalysis(**data)
            except FileNotFoundError:
                pass
            
            # Load trade history
            try:
                with open(self.journal_file, 'r') as f:
                    history_data = json.load(f)
                    for trade_data in history_data:
                        trade_data['entry_date'] = datetime.fromisoformat(trade_data['entry_date'])
                        trade_data['exit_date'] = datetime.fromisoformat(trade_data['exit_date'])
                        self.trade_history.append(TradeResult(**trade_data))
            except FileNotFoundError:
                pass
                
        except Exception as e:
            logger.error(f"Error loading investment data: {e}")

    def _save_data(self):
        """Save investment data to files"""
        try:
            # Save strategies
            strategies_data = {}
            for strategy_id, strategy in self.strategies.items():
                data = asdict(strategy)
                data['strategy_type'] = strategy.strategy_type.value
                data['risk_level'] = strategy.risk_level.value
                data['expiration_date'] = strategy.expiration_date.isoformat()
                strategies_data[strategy_id] = data
            
            with open(self.strategies_file, 'w') as f:
                json.dump(strategies_data, f, indent=2)
            
            # Save trade history
            history_data = []
            for trade in self.trade_history:
                data = asdict(trade)
                data['entry_date'] = trade.entry_date.isoformat()
                data['exit_date'] = trade.exit_date.isoformat()
                history_data.append(data)
            
            with open(self.journal_file, 'w') as f:
                json.dump(history_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving investment data: {e}")


# Global instance
investment_tracker = None

def get_investment_tracker(data_dir: str = "data") -> InvestmentTracker:
    """Get or create global investment tracker instance"""
    global investment_tracker
    if investment_tracker is None:
        investment_tracker = InvestmentTracker(data_dir)
    return investment_tracker


if __name__ == "__main__":
    """Test the investment tracker"""
    print("=== Testing Tactical Investment Strategy Tracker ===")
    
    import os
    os.makedirs("data", exist_ok=True)
    
    tracker = InvestmentTracker("data")
    
    # Test credit spread analysis
    print("\n1. Testing Credit Spread Analysis:")
    
    # Example: SPY credit put spread
    legs = [
        OptionsLeg(action="sell", option_type="put", strike=450.0, premium=2.50, delta=-0.20),
        OptionsLeg(action="buy", option_type="put", strike=445.0, premium=1.20, delta=-0.15)
    ]
    
    expiration = datetime.now() + timedelta(days=30)
    
    analysis = tracker.analyze_strategy(
        ticker="SPY",
        strategy_type=StrategyType.CREDIT_SPREAD,
        legs=legs,
        expiration_date=expiration,
        user_context={"experience_level": "intermediate", "risk_tolerance": "moderate"}
    )
    
    print(f"Strategy ID: {analysis.strategy_id}")
    print(f"Entry Cost: ${analysis.entry_cost:.2f}")
    print(f"Max Gain: ${analysis.max_gain:.2f}")
    print(f"Max Loss: ${analysis.max_loss:.2f}")
    print(f"Breakeven: {[f'${bp:.2f}' for bp in analysis.breakeven_points]}")
    print(f"Probability of Profit: {analysis.probability_of_profit:.1%}")
    print(f"Risk Level: {analysis.risk_level.value}")
    print(f"Review: {analysis.plain_english_review}")
    print(f"Recommendation: {analysis.recommendation}")
    print(f"Emotional Context: {analysis.emotional_context}")
    
    # Test trade result logging
    print("\n2. Testing Trade Result Logging:")
    
    # Simulate closing the trade for a profit
    exit_date = datetime.now() + timedelta(days=15)
    result = tracker.log_trade_result(
        strategy_id=analysis.strategy_id,
        exit_value=50.0,  # Kept $50 of the $130 credit
        exit_date=exit_date,
        notes="Closed early at 50% profit target"
    )
    
    print(f"Trade Outcome: {result.outcome}")
    print(f"Profit/Loss: ${result.profit_loss:.2f}")
    print(f"Profit %: {result.profit_percentage:.1f}%")
    print(f"Days Held: {result.days_held}")
    print(f"Emotional Impact: {result.emotional_impact}")
    
    # Test performance summary
    print("\n3. Testing Performance Summary:")
    
    summary = tracker.get_performance_summary(days=30)
    print(f"Total Trades: {summary.get('total_trades', 0)}")
    print(f"Win Rate: {summary.get('win_rate', 0):.1%}")
    print(f"Total P&L: ${summary.get('total_pnl', 0):.2f}")
    print(f"Average Win: ${summary.get('average_win', 0):.2f}")
    
    print("\n=== Investment Tracker Test Complete ===")

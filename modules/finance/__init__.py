"""
Finance Module - Tactical Investment Strategy Tracker
Emotionally-aware options strategy analysis and goal tracking integration
"""

from .investment_tracker import (
    InvestmentTracker,
    StrategyType,
    RiskLevel,
    OptionsLeg,
    StrategyAnalysis,
    TradeResult,
    get_investment_tracker
)

from .investment_goals import (
    InvestmentGoalsTracker,
    GoalType,
    GoalStatus,
    InvestmentGoal,
    get_goals_tracker
)

from .investment_integration import (
    InvestmentCompanionIntegration,
    get_investment_integration
)

__all__ = [
    # Investment Tracker
    'InvestmentTracker',
    'StrategyType',
    'RiskLevel',
    'OptionsLeg',
    'StrategyAnalysis',
    'TradeResult',
    'get_investment_tracker',
    
    # Investment Goals
    'InvestmentGoalsTracker',
    'GoalType',
    'GoalStatus',
    'InvestmentGoal',
    'get_goals_tracker',
    
    # Integration
    'InvestmentCompanionIntegration',
    'get_investment_integration'
]

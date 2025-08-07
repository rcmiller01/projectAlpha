# Agent modules

from .base_agent import SLiMAgent, AgentResponse
from .deduction_agent import DeductionAgent
from .metaphor_agent import MetaphorAgent

__all__ = [
    'SLiMAgent',
    'AgentResponse', 
    'DeductionAgent',
    'MetaphorAgent'
]

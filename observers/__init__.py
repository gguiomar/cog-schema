"""
Bayesian Agent Simulation Package

This package provides a clean, organized structure for running Bayesian agent simulations
with temporal reasoning tasks. It includes:

- Environment: Temporal reasoning environment with biased cues
- Agents: Bayesian, Random, and MAP decision-making agents  
- Simulation: Complete simulation framework with JSON logging and plotting

Usage:
    from bayesian.simulation import BayesianSimulation
    
    sim = BayesianSimulation(verbose=True)
    results = sim.run_all_simulations()
    sim.plot_results()
"""

from .environment import TemporalReasoningEnvironment
from .agents import BayesAgent, RandomPolicyAgent, MAPAgent
from .simulation import BayesianSimulation

__all__ = [
    'TemporalReasoningEnvironment',
    'BayesAgent', 
    'RandomPolicyAgent',
    'MAPAgent',
    'BayesianSimulation'
]

__version__ = "1.0.0"

#%%
import sys
import os

# Add the parent directory to the path so we can import the bayesian package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bayesian.simulation import BayesianSimulation
import numpy as np


# Initialize simulation with small parameters for quick testing
sim = BayesianSimulation(
    k=4,
    p_t=0.9,
    p_f=0.5,
    n_trials=100,  # Small number for quick test
    rounds= np.arange(1, 100),  # Fewer rounds for quick test
    agent_types=["BayesAgent"],
    verbose=True,
    log_results=True,
    seed=42
)
# Run simulations
results = sim.run_all_simulations()
sim.plot_results(save_plots=False)


# %%

#%%
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bayesian.simulation import BayesianSimulation
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from bayesian.utils import plot_comparison

#%%
sim_standard = BayesianSimulation(
    k=4,
    p_t=0.9,
    p_f=0.5,
    n_trials=100,  
    rounds=np.arange(1, 100),  
    agent_types=["BayesAgent"],
    verbose=True,
    log_results=True,
    seed=42,
    use_hidden_cues=False  
)

results_standard = sim_standard.run_all_simulations()
sim_standard.plot_results(save_plots=False)


#%%
sim_hidden = BayesianSimulation(
    k=4,
    p_t=0.9,
    p_f=0.5,
    n_trials=100,  
    rounds=list(np.arange(1, 100)),  
    agent_types=["BayesAgent"],
    verbose=True,
    log_results=True,
    seed=42,  
    use_hidden_cues=True,  
    min_available_cues=1,  
    max_available_cues=3   
)

results_hidden = sim_hidden.run_all_simulations()
sim_hidden.plot_results(save_plots=False)



#%%
plot_comparison(results_standard, results_hidden, save_plots=False, fig_size=(10, 6))

# %%

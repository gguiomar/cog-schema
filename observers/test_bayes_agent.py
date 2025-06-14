#%%
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from observers.simulation import BayesianSimulation
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#%%
# BayesAgent Parameter Exploration
def plot_bayes_agent_comparison(results_dict, save_plots=True, fig_size=(14, 8)):
    """Plot comparison of BayesAgent with different parameters."""
    sns.set_style("white")
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E']
    plt.rcParams.update({'font.size': 12, 'axes.titlesize': 14, 'axes.labelsize': 12, 'legend.fontsize': 11})
    
    agent_name = "BayesAgent"
    metrics = ['map_accuracy', 'entropy_values', 'shannon_surprises', 'bayesian_surprises']
    
    # Prepare data for all conditions
    plot_data = {}
    condition_labels = []
    
    for i, (condition_name, results) in enumerate(results_dict.items()):
        rounds_list = sorted(results[agent_name].keys())
        plot_data[condition_name] = {m: {'means': [], 'stds': []} for m in metrics}
        condition_labels.append(condition_name)
        
        for n_rounds in rounds_list:
            for metric in metrics:
                plot_data[condition_name][metric]['means'].append(results[agent_name][n_rounds]['metrics'][metric]['mean'])
                plot_data[condition_name][metric]['stds'].append(results[agent_name][n_rounds]['metrics'][metric]['std'])
        
        for metric in metrics:
            plot_data[condition_name][metric]['means'] = np.array(plot_data[condition_name][metric]['means'])
            plot_data[condition_name][metric]['stds'] = np.array(plot_data[condition_name][metric]['stds'])
    
    fig, axes = plt.subplots(2, 2, figsize=fig_size)
    axes = axes.flatten()
    
    titles = {'map_accuracy': 'MAP Decision Accuracy', 'entropy_values': 'Posterior Entropy (nats)',
              'shannon_surprises': 'Shannon Surprise', 'bayesian_surprises': 'Bayesian Surprise'}
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        for j, condition_name in enumerate(condition_labels):
            means = plot_data[condition_name][metric]['means']
            stds = plot_data[condition_name][metric]['stds']
            color = colors[j % len(colors)]
            ax.plot(rounds_list, means, "-", label=condition_name, color=color, linewidth=2.5)
            ax.fill_between(rounds_list, means - stds, means + stds, color=color, alpha=0.2)
        
        if metric == 'map_accuracy':
            ax.axhline(y=0.25, color="gray", linestyle="--", alpha=0.7, label="Chance level")
            ax.set_ylim(0, 1.2)
        elif metric == 'entropy_values':
            ax.axhline(y=np.log(4), color="gray", linestyle="--", alpha=0.7, label="Max entropy")
        
        ax.set_xlabel("Number of Rounds")
        ax.set_xlim(1, max(rounds_list))
        ax.set_ylabel(titles[metric])
        ax.set_title(f"{titles[metric]} Comparison")
        ax.legend()
        ax.grid(True, alpha=0.3)
        sns.despine(ax=ax)
    
    plt.tight_layout()
    
    if save_plots:
        timestamp = list(results_dict.values())[0][agent_name][rounds_list[0]]['timestamp']
        plt.savefig(f"bayesian_plots/bayes_agent_comparison_{timestamp}.png", dpi=300, bbox_inches='tight')
    
    plt.show()

#%%
# Signal strength comparison with hidden cues
print("Running signal strength comparison...")

# Common parameters
common_params = {
    'k': 4,
    'p_f': 0.5,
    'n_trials': 100,
    'rounds': list(range(1, 50)),
    'agent_types': ["BayesAgent"],
    'verbose': True,
    'log_results': True,
    'seed': 42,
    'use_hidden_cues': True,
    'min_available_cues': 1,
    'max_available_cues': 2
}

# Standard task (all cues available)
results_standard = BayesianSimulation(
    **{**common_params, 'use_hidden_cues': False, 'p_t': 0.9}).run_all_simulations()

# Hidden cues with different signal strengths
results_hidden_09 = BayesianSimulation(**{**common_params, 'p_t': 0.9}).run_all_simulations()
results_hidden_07 = BayesianSimulation(**{**common_params, 'p_t': 0.7}).run_all_simulations()
results_hidden_06 = BayesianSimulation(**{**common_params, 'p_t': 0.6}).run_all_simulations()

#%%
# sim with k = 10, p_t = 0.9, p_f = 0.5 and hidden cues with max 2 cues
results_hidden_k10 = BayesianSimulation(
    k=10,
    p_t=0.9,
    p_f=0.5,
    n_trials=100,
    rounds=list(range(1, 50)),
    agent_types=["BayesAgent"],
    verbose=True,
    log_results=True,
    seed=42,
    use_hidden_cues=True,
    min_available_cues=1,
    max_available_cues=2
).run_all_simulations()

#%%
results_nohidden_k10 = BayesianSimulation(
    k=10,
    p_t=0.9,
    p_f=0.5,
    n_trials=100,
    rounds=list(range(1, 50)),
    agent_types=["BayesAgent"],
    verbose=True,
    log_results=True,
    seed=42,
    use_hidden_cues=False,
).run_all_simulations()


#%%

# Compare all conditions
results_comparison = {
    'No Hidden Cues (p_t=0.9)': results_standard,
    'Hidden Cues (p_t=0.9)': results_hidden_09,
    'Hidden Cues (p_t=0.7)': results_hidden_07,
    'Hidden Cues (p_t=0.6)': results_hidden_06,
}

plot_bayes_agent_comparison(results_comparison, save_plots=True)

#%%
results_comparison = {
    'Hidden Cues (k=10, p_t=0.9)': results_hidden_k10,
    'No Hidden Cues (k=10, p_t=0.9)': results_nohidden_k10
}

plot_bayes_agent_comparison(results_comparison, save_plots=True)


#%%

####### -- Additional Simulations -- #######

# Different k values (number of locations)
sim_k2 = BayesianSimulation(
    k=2,  # Binary choice
    p_t=0.9,
    p_f=0.5,
    n_trials=100,  
    rounds=list(np.arange(1, 50)),  
    agent_types=["BayesAgent"],
    verbose=True,
    log_results=True,
    seed=42,
    use_hidden_cues=False  
)

results_k2 = sim_k2.run_all_simulations()

sim_k6 = BayesianSimulation(
    k=6,  # More locations
    p_t=0.9,
    p_f=0.5,
    n_trials=100,  
    rounds=list(np.arange(1, 50)),  
    agent_types=["BayesAgent"],
    verbose=True,
    log_results=True,
    seed=42,
    use_hidden_cues=False  
)

results_k6 = sim_k6.run_all_simulations()

#%%
# Compare different k values
results_k_comparison = {
    'k=2 locations': results_k2,
    'k=4 locations': results_standard,
    'k=6 locations': results_k6
}

plot_bayes_agent_comparison(results_k_comparison, save_plots=True)

# %%

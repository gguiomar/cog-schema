#%%
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from observers.agents import ActiveInferenceAgent, BayesAgent, MAPAgent
from observers.environment import TemporalReasoningEnvironment

#%%
def run_comparison_simulation(k=4, p_t=0.9, p_f=0.5, n_trials=100, max_rounds=30, seed=42):
    """
    Run simulation comparing Bayes agent (MAP decisions) vs Active Inference agent (entropy-minimizing decisions).
    """
    rng = np.random.default_rng(seed)
    env = TemporalReasoningEnvironment(k, p_t, p_f, rng)
    
    results = {'BayesAgent_MAP': {}, 'ActiveInference_Entropy': {}}
    
    print("Running comparison: Bayes Agent (MAP) vs Active Inference Agent (Entropy-minimizing)")
    
    for n_rounds in range(1, max_rounds + 1):
        if n_rounds % 5 == 0:
            print(f"  Processing {n_rounds} rounds...")
        
        bayes_trials = []
        ai_trials = []
        
        for trial in range(n_trials):
            # Start trial
            true_z = env.start_trial()
            
            # Initialize agents
            bayes_agent = BayesAgent(k, p_t, p_f)
            ai_agent = ActiveInferenceAgent(k, p_t, p_f, beta=1.0)
            map_agent = MAPAgent(k)
            
            # Run rounds - Bayes agent uses random cue selection
            for round_num in range(n_rounds):
                available_cues = list(range(k))
                
                # Random cue selection for Bayes agent
                bayes_cue = rng.choice(available_cues)
                if bayes_cue == true_z:
                    p_color_1 = p_t
                else:
                    p_color_1 = p_f
                bayes_color = int(rng.random() < p_color_1)
                bayes_agent.update(bayes_cue, bayes_color)
                
                # Active inference agent selects cue
                ai_cue = ai_agent.select_action(available_cues)
                if ai_cue == true_z:
                    p_color_1 = p_t
                else:
                    p_color_1 = p_f
                ai_color = int(rng.random() < p_color_1)
                ai_agent.update(ai_cue, ai_color)
            
            # Final decisions
            bayes_decision = bayes_agent.get_decision()  # Uses MAP
            ai_decision = ai_agent.get_decision()        # Uses entropy minimization
            
            # Store results
            bayes_trials.append({
                'correct': int(bayes_decision == true_z),
                'entropy': bayes_agent.entropy,
                'posterior_true_target': bayes_agent.posterior[true_z],
                'map_probability': np.max(bayes_agent.posterior)
            })
            
            ai_trials.append({
                'correct': int(ai_decision == true_z),
                'entropy': ai_agent.entropy,
                'posterior_true_target': ai_agent.posterior[true_z],
                'map_probability': np.max(ai_agent.posterior)
            })
        
        # Aggregate results
        for agent_name, trials in [('BayesAgent_MAP', bayes_trials), ('ActiveInference_Entropy', ai_trials)]:
            metrics = {}
            for metric_name in ['correct', 'entropy', 'posterior_true_target', 'map_probability']:
                values = [t[metric_name] for t in trials]
                metrics[metric_name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'values': values
                }
            results[agent_name][n_rounds] = metrics
    
    return results

#%%
def plot_comparison(results, save_plots=True):
    """Plot comparison between Bayes agent (MAP) and Active Inference agent (entropy-minimizing)."""
    sns.set_style("white")
    colors = {'BayesAgent_MAP': '#2E86AB', 'ActiveInference_Entropy': '#A23B72'}
    plt.rcParams.update({'font.size': 12, 'axes.titlesize': 14, 'axes.labelsize': 12, 'legend.fontsize': 11})
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    
    rounds_list = sorted(results['BayesAgent_MAP'].keys())
    
    metrics = ['correct', 'entropy', 'posterior_true_target', 'map_probability']
    titles = {
        'correct': 'Decision Accuracy',
        'entropy': 'Posterior Entropy (nats)',
        'posterior_true_target': 'P(true target)',
        'map_probability': 'MAP Probability'
    }
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        for agent_name in ['BayesAgent_MAP', 'ActiveInference_Entropy']:
            means = []
            stds = []
            
            for n_rounds in rounds_list:
                data = results[agent_name][n_rounds][metric]
                means.append(data['mean'])
                stds.append(data['std'])
            
            means = np.array(means)
            stds = np.array(stds)
            
            label = 'Bayes Agent (MAP)' if agent_name == 'BayesAgent_MAP' else 'Active Inference (Entropy-min)'
            color = colors[agent_name]
            
            ax.plot(rounds_list, means, "-o", label=label, color=color, linewidth=2.5, markersize=4)
            ax.fill_between(rounds_list, means - stds, means + stds, color=color, alpha=0.2)
        
        # Add reference lines
        if metric == 'correct':
            ax.axhline(y=0.25, color="gray", linestyle="--", alpha=0.7, label="Chance level")
            ax.set_ylim(0, 1)
        elif metric == 'entropy':
            ax.axhline(y=np.log(4), color="gray", linestyle="--", alpha=0.7, label="Max entropy")
        
        ax.set_xlabel("Number of Rounds")
        ax.set_ylabel(titles[metric])
        ax.set_title(f"{titles[metric]} Comparison")
        ax.legend()
        ax.grid(True, alpha=0.3)
        sns.despine(ax=ax)
    
    plt.tight_layout()
    
    if save_plots:
        os.makedirs("observers/bayesian_plots", exist_ok=True)
        plt.savefig("observers/bayesian_plots/bayes_vs_active_inference_comparison.png", dpi=300, bbox_inches='tight')
        print("Plot saved to observers/bayesian_plots/bayes_vs_active_inference_comparison.png")
    
    plt.show()

#%%
# Run the comparison
results = run_comparison_simulation(k=4, p_t=0.9, p_f=0.5, n_trials=100, max_rounds=30, seed=42)

# Plot results
plot_comparison(results, save_plots=True)

# Print summary
print_summary_statistics(results)

# %%

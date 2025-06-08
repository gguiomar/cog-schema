import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any


def plot_comparison(results_standard, results_hidden, save_plots=True, fig_size=(14, 6)):
    sns.set_style("white")
    colors = ['#2E86AB', '#A23B72']
    plt.rcParams.update({'font.size': 12, 'axes.titlesize': 14, 'axes.labelsize': 12, 'legend.fontsize': 11})
    
    agent_name = "BayesAgent"
    rounds_list = sorted(results_standard[agent_name].keys())
    metrics = ['map_accuracy', 'entropy_values', 'shannon_surprises', 'bayesian_surprises']
    
    plot_data = {'standard': {m: {'means': [], 'stds': []} for m in metrics},
                 'hidden': {m: {'means': [], 'stds': []} for m in metrics}}
    
    for n_rounds in rounds_list:
        for condition, results in [('standard', results_standard), ('hidden', results_hidden)]:
            for metric in metrics:
                plot_data[condition][metric]['means'].append(results[agent_name][n_rounds]['metrics'][metric]['mean'])
                plot_data[condition][metric]['stds'].append(results[agent_name][n_rounds]['metrics'][metric]['std'])
    
    for condition in ['standard', 'hidden']:
        for metric in metrics:
            plot_data[condition][metric]['means'] = np.array(plot_data[condition][metric]['means'])
            plot_data[condition][metric]['stds'] = np.array(plot_data[condition][metric]['stds'])
    
    fig, axes = plt.subplots(2, 2, figsize=fig_size)
    axes = axes.flatten()
    
    titles = {'map_accuracy': 'MAP Decision Accuracy', 'entropy_values': 'Posterior Entropy (nats)',
              'shannon_surprises': 'Shannon Surprise', 'bayesian_surprises': 'Bayesian Surprise'}
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        for j, (condition, label) in enumerate([('standard', 'Standard Task (All Cues)'), 
                                               ('hidden', 'Hidden Cues Task (1-3 Cues)')]):
            means = plot_data[condition][metric]['means']
            stds = plot_data[condition][metric]['stds']
            ax.plot(rounds_list, means, "-", label=label, color=colors[j], linewidth=2.5)
            ax.fill_between(rounds_list, means - stds, means + stds, color=colors[j], alpha=0.2)
        
        if metric == 'map_accuracy':
            ax.axhline(y=0.25, color="gray", linestyle="--", alpha=0.7, label="Chance level")
            ax.set_ylim(0, 1)
        elif metric == 'entropy_values':
            ax.axhline(y=np.log(4), color="gray", linestyle="--", alpha=0.7, label="Max entropy")
        
        ax.set_xlabel("Number of Rounds")
        ax.set_ylabel(titles[metric])
        ax.set_title(f"{titles[metric]} Comparison")
        ax.legend()
        ax.grid(True, alpha=0.3)
        sns.despine(ax=ax)
    
    plt.tight_layout()
    
    if save_plots:
        timestamp = results_standard[agent_name][rounds_list[0]]['timestamp']
        plt.savefig(f"bayesian_plots/comparison_standard_vs_hidden_{timestamp}.png", dpi=300, bbox_inches='tight')
    
    plt.show()

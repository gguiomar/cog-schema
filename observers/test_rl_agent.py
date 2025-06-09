#%%
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from observers.agents import RLAgent, BayesAgent, MAPAgent
from observers.environment import TemporalReasoningEnvironment, RLEnvironmentWrapper

#%%
# RL Agent Parameter Exploration
def plot_rl_agent_comparison(results_dict, save_plots=True, fig_size=(14, 8)):
    """Plot comparison of RLAgent with different parameters."""
    sns.set_style("white")
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E', '#8E44AD']
    plt.rcParams.update({'font.size': 12, 'axes.titlesize': 14, 'axes.labelsize': 12, 'legend.fontsize': 11})
    
    metrics = ['map_accuracy', 'entropy_values', 'q_value_evolution', 'learning_progress']
    
    # Prepare data for all conditions
    plot_data = {}
    condition_labels = []
    
    for i, (condition_name, results) in enumerate(results_dict.items()):
        rounds_list = sorted(results.keys())
        plot_data[condition_name] = {m: {'means': [], 'stds': []} for m in metrics}
        condition_labels.append(condition_name)
        
        for n_rounds in rounds_list:
            for metric in metrics:
                if metric in results[n_rounds]:
                    plot_data[condition_name][metric]['means'].append(results[n_rounds][metric]['mean'])
                    plot_data[condition_name][metric]['stds'].append(results[n_rounds][metric]['std'])
                else:
                    plot_data[condition_name][metric]['means'].append(0)
                    plot_data[condition_name][metric]['stds'].append(0)
        
        for metric in metrics:
            plot_data[condition_name][metric]['means'] = np.array(plot_data[condition_name][metric]['means'])
            plot_data[condition_name][metric]['stds'] = np.array(plot_data[condition_name][metric]['stds'])
    
    fig, axes = plt.subplots(2, 2, figsize=fig_size)
    axes = axes.flatten()
    
    titles = {'map_accuracy': 'MAP Decision Accuracy', 'entropy_values': 'Posterior Entropy (nats)',
              'q_value_evolution': 'Mean Q-Value', 'learning_progress': 'Learning Progress'}
    
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
        plt.savefig(f"bayesian_plots/rl_agent_comparison.png", dpi=300, bbox_inches='tight')
    
    plt.show()

#%%
def run_rl_simulation(k=4, p_t=0.9, p_f=0.5, alpha=0.1, gamma=0.9, beta=1.0, 
                     state_discretization=10, n_trials=100, max_rounds=30, 
                     use_hidden_cues=False, min_available_cues=None, max_available_cues=None, 
                     seed=42, n_training_episodes=1000):
    """Run simulation with RL Agent including training phase."""
    rng = np.random.default_rng(seed)
    base_env = TemporalReasoningEnvironment(k, p_t, p_f, rng, use_hidden_cues, 
                                          min_available_cues, max_available_cues)
    rl_env = RLEnvironmentWrapper(base_env)
    
    # Initialize RL agent
    rl_agent = RLAgent(k, p_t, p_f, alpha, gamma, beta, state_discretization)
    
    # Training phase
    print(f"Training RL agent for {n_training_episodes} episodes...")
    training_rewards = []
    
    for episode in range(n_training_episodes):
        # Reset agent and start new episode
        rl_agent.reset()
        true_z = rl_env.start_trial(max_episode_length=max_rounds)
        
        episode_reward = 0
        done = False
        step = 0
        
        while not done and step < max_rounds:
            # Get current state
            current_state = rl_agent.discretize_state(rl_agent.posterior)
            
            # Get available cues
            if use_hidden_cues:
                n_available = rng.integers(min_available_cues or 1, (max_available_cues or k) + 1)
                available_cues = sorted(rng.choice(k, size=n_available, replace=False))
            else:
                available_cues = list(range(k))
            
            # Select action
            action = rl_agent.select_action(available_cues, current_state)
            
            # Take step in environment
            cue, color, _, reward, done = rl_env.step(action, is_final_step=(step == max_rounds - 1))
            
            if not done:
                # Update beliefs and get next state
                rl_agent.update(cue, color)
                next_state = rl_agent.discretize_state(rl_agent.posterior)
                
                # Update Q-values
                rl_agent.update_q_values(current_state, action, reward, next_state, done)
            else:
                # Final step - action is the decision
                final_decision = action
                reward = 1.0 if final_decision == true_z else 0.0
                rl_agent.update_q_values(current_state, action, reward, current_state, done)
            
            episode_reward += reward
            step += 1
        
        training_rewards.append(episode_reward)
        
        if episode % 200 == 0:
            recent_avg = np.mean(training_rewards[-100:]) if len(training_rewards) >= 100 else np.mean(training_rewards)
            print(f"Episode {episode}, Recent avg reward: {recent_avg:.3f}")
    
    # Testing phase
    print("Testing trained RL agent...")
    results = {}
    
    for n_rounds in range(1, max_rounds + 1):
        trial_results = []
        
        for trial in range(n_trials):
            # Reset for testing (don't reset Q-table)
            rl_agent.posterior = np.full(k, 1.0 / k)
            bayes_agent = BayesAgent(k, p_t, p_f)
            map_agent = MAPAgent(k)
            
            # Start trial
            true_z = base_env.start_trial()
            
            # Run rounds
            q_values_trial = []
            for round_num in range(n_rounds):
                # Get available cues
                if use_hidden_cues:
                    n_available = rng.integers(min_available_cues or 1, (max_available_cues or k) + 1)
                    available_cues = sorted(rng.choice(k, size=n_available, replace=False))
                else:
                    available_cues = list(range(k))
                
                # RL agent selects cue
                current_state = rl_agent.discretize_state(rl_agent.posterior)
                selected_cue = rl_agent.select_action(available_cues, current_state)
                
                # Environment responds
                cue, color, _ = base_env.sample_round(true_z)
                # Override with RL agent's choice
                if selected_cue == true_z:
                    p_color_1 = p_t
                else:
                    p_color_1 = p_f
                color = int(rng.random() < p_color_1)
                
                # Update agents
                rl_agent.update(selected_cue, color)
                bayes_agent.update(selected_cue, color)
                
                # Record Q-values
                q_values_trial.append(np.mean(rl_agent.q_values[current_state, :]))
            
            # Final decisions
            rl_decision = rl_agent.get_decision()
            map_decision = map_agent.get_decision(rl_agent.posterior)
            
            # Store results
            trial_results.append({
                'rl_correct': int(rl_decision == true_z),
                'map_correct': int(map_decision == true_z),
                'entropy': rl_agent.entropy,
                'posterior_true_target': rl_agent.posterior[true_z],
                'map_probability': np.max(rl_agent.posterior),
                'mean_q_value': np.mean(q_values_trial) if q_values_trial else 0
            })
        
        # Aggregate trial results
        metrics = {}
        for metric_name in ['rl_correct', 'map_correct', 'entropy', 'posterior_true_target', 
                           'map_probability', 'mean_q_value']:
            values = [t[metric_name] for t in trial_results]
            metrics[metric_name] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'values': values
            }
        
        # Rename for consistency
        metrics['map_accuracy'] = metrics.pop('map_correct')
        metrics['entropy_values'] = metrics.pop('entropy')
        metrics['q_value_evolution'] = metrics.pop('mean_q_value')
        metrics['learning_progress'] = {'mean': np.mean(training_rewards[-100:]), 'std': 0, 'values': []}
        
        results[n_rounds] = metrics
    
    return results, training_rewards

#%%
# Test different learning rates
print("Running RL simulations with different learning rates...")

results_alpha_low, _ = run_rl_simulation(alpha=0.05, max_rounds=20, n_trials=50, n_training_episodes=500)
results_alpha_medium, _ = run_rl_simulation(alpha=0.1, max_rounds=20, n_trials=50, n_training_episodes=500)
results_alpha_high, _ = run_rl_simulation(alpha=0.3, max_rounds=20, n_trials=50, n_training_episodes=500)

#%%
# Compare learning rates
results_alpha_comparison = {
    'α=0.05 (Slow Learning)': results_alpha_low,
    'α=0.1 (Medium Learning)': results_alpha_medium,
    'α=0.3 (Fast Learning)': results_alpha_high
}

plot_rl_agent_comparison(results_alpha_comparison, save_plots=True)

#%%
# Test different exploration parameters (beta)
print("Running RL simulations with different exploration parameters...")

results_beta_low, _ = run_rl_simulation(beta=0.5, max_rounds=20, n_trials=50, n_training_episodes=500)
results_beta_medium, _ = run_rl_simulation(beta=1.0, max_rounds=20, n_trials=50, n_training_episodes=500)
results_beta_high, _ = run_rl_simulation(beta=2.0, max_rounds=20, n_trials=50, n_training_episodes=500)

#%%
# Compare exploration parameters
results_beta_comparison = {
    'β=0.5 (High Exploration)': results_beta_low,
    'β=1.0 (Balanced)': results_beta_medium,
    'β=2.0 (Low Exploration)': results_beta_high
}

plot_rl_agent_comparison(results_beta_comparison, save_plots=True)

#%%
# Test different state discretizations
print("Running RL simulations with different state discretizations...")

results_disc_coarse, _ = run_rl_simulation(state_discretization=5, max_rounds=20, n_trials=50, n_training_episodes=500)
results_disc_medium, _ = run_rl_simulation(state_discretization=10, max_rounds=20, n_trials=50, n_training_episodes=500)
results_disc_fine, _ = run_rl_simulation(state_discretization=15, max_rounds=20, n_trials=50, n_training_episodes=500)

#%%
# Compare state discretizations
results_disc_comparison = {
    '5 bins (Coarse)': results_disc_coarse,
    '10 bins (Medium)': results_disc_medium,
    '15 bins (Fine)': results_disc_fine
}

plot_rl_agent_comparison(results_disc_comparison, save_plots=True)

#%%
# Learning curve analysis
def plot_learning_curves(training_rewards_dict, save_plots=True):
    """Plot learning curves for different RL configurations."""
    sns.set_style("white")
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    plt.rcParams.update({'font.size': 12, 'axes.titlesize': 14, 'axes.labelsize': 12, 'legend.fontsize': 11})
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    for i, (condition_name, rewards) in enumerate(training_rewards_dict.items()):
        # Smooth the learning curve with moving average
        window_size = 50
        smoothed_rewards = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
        episodes = np.arange(window_size-1, len(rewards))
        
        color = colors[i % len(colors)]
        ax.plot(episodes, smoothed_rewards, label=condition_name, color=color, linewidth=2)
    
    ax.set_xlabel("Training Episode")
    ax.set_ylabel("Average Reward (50-episode window)")
    ax.set_title("RL Agent Learning Curves")
    ax.legend()
    ax.grid(True, alpha=0.3)
    sns.despine(ax=ax)
    
    if save_plots:
        plt.savefig("bayesian_plots/rl_learning_curves.png", dpi=300, bbox_inches='tight')
    
    plt.show()

#%%
# Generate learning curves for different configurations
print("Generating learning curves...")

_, rewards_alpha_low = run_rl_simulation(alpha=0.05, max_rounds=15, n_trials=20, n_training_episodes=1000)
_, rewards_alpha_medium = run_rl_simulation(alpha=0.1, max_rounds=15, n_trials=20, n_training_episodes=1000)
_, rewards_alpha_high = run_rl_simulation(alpha=0.3, max_rounds=15, n_trials=20, n_training_episodes=1000)

learning_curves = {
    'α=0.05': rewards_alpha_low,
    'α=0.1': rewards_alpha_medium,
    'α=0.3': rewards_alpha_high
}

plot_learning_curves(learning_curves, save_plots=True)

#%%
# RL Agent vs MAP Bayes Agent Comparison
def run_rl_vs_bayes_comparison(k=4, p_t=0.9, p_f=0.5, n_trials=100, max_rounds=30, 
                               seed=42, n_training_episodes=1000):
    """
    Run simulation comparing RL agent (alpha=0.1, beta=1) vs MAP Bayes agent.
    """
    rng = np.random.default_rng(seed)
    env = TemporalReasoningEnvironment(k, p_t, p_f, rng)
    rl_env = RLEnvironmentWrapper(env)
    
    # Initialize and train RL agent
    rl_agent = RLAgent(k, p_t, p_f, alpha=0.1, gamma=0.9, beta=1.0, state_discretization=10)
    
    print("Training RL agent for comparison...")
    training_rewards = []
    
    # Training phase
    for episode in range(n_training_episodes):
        rl_agent.reset()
        true_z = rl_env.start_trial(max_episode_length=max_rounds)
        
        episode_reward = 0
        done = False
        step = 0
        
        while not done and step < max_rounds:
            current_state = rl_agent.discretize_state(rl_agent.posterior)
            available_cues = list(range(k))
            action = rl_agent.select_action(available_cues, current_state)
            
            cue, color, _, reward, done = rl_env.step(action, is_final_step=(step == max_rounds - 1))
            
            if not done:
                rl_agent.update(cue, color)
                next_state = rl_agent.discretize_state(rl_agent.posterior)
                rl_agent.update_q_values(current_state, action, reward, next_state, done)
            else:
                final_decision = action
                reward = 1.0 if final_decision == true_z else 0.0
                rl_agent.update_q_values(current_state, action, reward, current_state, done)
            
            episode_reward += reward
            step += 1
        
        training_rewards.append(episode_reward)
        
        if episode % 200 == 0:
            recent_avg = np.mean(training_rewards[-100:]) if len(training_rewards) >= 100 else np.mean(training_rewards)
            print(f"  Episode {episode}, Recent avg reward: {recent_avg:.3f}")
    
    # Testing phase
    print("Testing RL agent vs MAP Bayes agent...")
    results = {'RL_Agent': {}, 'MAP_Bayes': {}}
    
    for n_rounds in range(1, max_rounds + 1):
        if n_rounds % 5 == 0:
            print(f"  Processing {n_rounds} rounds...")
        
        rl_trials = []
        bayes_trials = []
        
        for trial in range(n_trials):
            # Start trial
            true_z = env.start_trial()
            
            # Initialize agents
            rl_agent.posterior = np.full(k, 1.0 / k)  # Reset beliefs but keep Q-table
            bayes_agent = BayesAgent(k, p_t, p_f)
            map_agent = MAPAgent(k)
            
            # Run rounds - RL agent selects cues
            for round_num in range(n_rounds):
                available_cues = list(range(k))
                
                # RL agent selects cue
                current_state = rl_agent.discretize_state(rl_agent.posterior)
                rl_cue = rl_agent.select_action(available_cues, current_state)
                if rl_cue == true_z:
                    p_color_1 = p_t
                else:
                    p_color_1 = p_f
                rl_color = int(rng.random() < p_color_1)
                rl_agent.update(rl_cue, rl_color)
                
                # Bayes agent uses random cue selection
                bayes_cue = rng.choice(available_cues)
                if bayes_cue == true_z:
                    p_color_1 = p_t
                else:
                    p_color_1 = p_f
                bayes_color = int(rng.random() < p_color_1)
                bayes_agent.update(bayes_cue, bayes_color)
            
            # Final decisions
            rl_decision = rl_agent.get_decision()  # Uses MAP
            bayes_decision = bayes_agent.get_decision()  # Uses MAP
            
            # Store results
            rl_trials.append({
                'correct': int(rl_decision == true_z),
                'entropy': rl_agent.entropy,
                'posterior_true_target': rl_agent.posterior[true_z],
                'map_probability': np.max(rl_agent.posterior)
            })
            
            bayes_trials.append({
                'correct': int(bayes_decision == true_z),
                'entropy': bayes_agent.entropy,
                'posterior_true_target': bayes_agent.posterior[true_z],
                'map_probability': np.max(bayes_agent.posterior)
            })
        
        # Aggregate results
        for agent_name, trials in [('RL_Agent', rl_trials), ('MAP_Bayes', bayes_trials)]:
            metrics = {}
            for metric_name in ['correct', 'entropy', 'posterior_true_target', 'map_probability']:
                values = [t[metric_name] for t in trials]
                metrics[metric_name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'values': values
                }
            results[agent_name][n_rounds] = metrics
    
    return results, training_rewards

#%%
def plot_rl_vs_bayes_comparison(results, save_plots=True):
    """Plot comparison between RL agent and MAP Bayes agent."""
    sns.set_style("white")
    colors = {'RL_Agent': '#2E86AB', 'MAP_Bayes': '#A23B72'}
    plt.rcParams.update({'font.size': 12, 'axes.titlesize': 14, 'axes.labelsize': 12, 'legend.fontsize': 11})
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    
    rounds_list = sorted(results['RL_Agent'].keys())
    
    metrics = ['correct', 'entropy', 'posterior_true_target', 'map_probability']
    titles = {
        'correct': 'Decision Accuracy',
        'entropy': 'Posterior Entropy (nats)',
        'posterior_true_target': 'P(true target)',
        'map_probability': 'MAP Probability'
    }
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        for agent_name in ['RL_Agent', 'MAP_Bayes']:
            means = []
            stds = []
            
            for n_rounds in rounds_list:
                data = results[agent_name][n_rounds][metric]
                means.append(data['mean'])
                stds.append(data['std'])
            
            means = np.array(means)
            stds = np.array(stds)
            
            label = 'RL Agent (α=0.1, β=1)' if agent_name == 'RL_Agent' else 'MAP Bayes Agent'
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
        plt.savefig("observers/bayesian_plots/rl_vs_bayes_comparison.png", dpi=300, bbox_inches='tight')
        print("Plot saved to observers/bayesian_plots/rl_vs_bayes_comparison.png")
    
    plt.show()

#%%
# Run the RL vs Bayes comparison
print("Starting RL Agent vs MAP Bayes Agent comparison...")
results, training_rewards = run_rl_vs_bayes_comparison(
    k=4, p_t=0.9, p_f=0.5, n_trials=100, max_rounds=30, seed=42, n_training_episodes=1000
)

# Plot results
plot_rl_vs_bayes_comparison(results, save_plots=True)

# %%

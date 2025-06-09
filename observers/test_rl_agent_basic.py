#%%
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from observers.environment import TemporalReasoningEnvironment

#%%
class VanillaRLAgent:
    """Simple RL agent using round-based states."""
    
    def __init__(self, k: int, max_rounds: int, alpha: float = 0.1, 
                 gamma: float = 0.9, beta: float = 1.0):
        """
        Initialize the vanilla RL agent.
        
        Parameters:
        -----------
        k : int
            Number of possible cues/locations
        max_rounds : int
            Maximum number of rounds per episode
        alpha : float
            Learning rate
        gamma : float
            Discount factor
        beta : float
            Inverse temperature for softmax action selection
        """
        self.k = k
        self.max_rounds = max_rounds
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta
        
        # Q-table: Q(round, action)
        self.q_values = np.zeros((max_rounds, k))
    
    def select_action(self, round_num: int, available_cues: list):
        """
        Select action using softmax policy based on Q-values.
        
        Parameters:
        -----------
        round_num : int
            Current round (1-indexed)
        available_cues : list
            List of available cues
            
        Returns:
        --------
        int
            Selected cue
        """
        state = round_num - 1  # Convert to 0-indexed
        
        # Get Q-values for available actions
        q_vals = np.array([self.q_values[state, cue] for cue in available_cues])
        
        # Softmax action selection
        exp_vals = np.exp(self.beta * q_vals)
        probs = exp_vals / np.sum(exp_vals)
        
        # Sample action
        action_idx = np.random.choice(len(available_cues), p=probs)
        return available_cues[action_idx]
    
    def update_q_values(self, round_num: int, action: int, reward: float, 
                       next_round: int = None, done: bool = False):
        """
        Update Q-values using TD learning.
        
        Parameters:
        -----------
        round_num : int
            Current round (1-indexed)
        action : int
            Action taken
        reward : float
            Reward received
        next_round : int
            Next round number (1-indexed)
        done : bool
            Whether episode is finished
        """
        state = round_num - 1  # Convert to 0-indexed
        
        if done:
            target = reward
        else:
            next_state = next_round - 1
            target = reward + self.gamma * np.max(self.q_values[next_state, :])
        
        td_error = target - self.q_values[state, action]
        self.q_values[state, action] += self.alpha * td_error

#%%
# Training and testing simulation
def run_vanilla_rl_simulation(k=4, p_t=0.9, p_f=0.5, max_rounds=10, 
                             n_training_trials=1000, n_test_trials=100, 
                             alpha=0.1, gamma=0.9, beta=1.0, seed=42):
    """
    Run vanilla RL simulation with training and testing phases.
    """
    rng = np.random.default_rng(seed)
    env = TemporalReasoningEnvironment(k, p_t, p_f, rng)
    
    # Initialize agent
    agent = VanillaRLAgent(k, max_rounds, alpha, gamma, beta)
    
    # Training phase
    print(f"Training vanilla RL agent for {n_training_trials} trials...")
    training_rewards = []
    q_value_evolution = []
    training_cue_rewards = np.zeros((max_rounds, k))  # Track rewards per (round, cue)
    training_cue_counts = np.zeros((max_rounds, k))   # Track selection counts
    
    for trial in range(n_training_trials):
        # Start new trial
        true_target = env.start_trial()
        episode_reward = 0
        
        # Run episode for max_rounds
        for round_num in range(1, max_rounds + 1):
            available_cues = list(range(k))  # All cues available
            
            # Select action
            action = agent.select_action(round_num, available_cues)
            
            # Track selection
            training_cue_counts[round_num - 1, action] += 1
            
            # Observe outcome (but don't use it for decision making)
            if action == true_target:
                color = int(rng.random() < p_t)
            else:
                color = int(rng.random() < p_f)
            
            if round_num < max_rounds:
                # Intermediate round: no reward
                reward = 0
                agent.update_q_values(round_num, action, reward, round_num + 1, done=False)
            else:
                # Final round: action IS the decision
                reward = 1.0 if action == true_target else 0.0
                training_cue_rewards[round_num - 1, action] += reward
                agent.update_q_values(round_num, action, reward, done=True)
                episode_reward = reward
        
        training_rewards.append(episode_reward)
        
        # Record Q-values evolution (every 50 trials)
        if trial % 50 == 0:
            q_value_evolution.append(agent.q_values.copy())
        
        if trial % 200 == 0:
            recent_avg = np.mean(training_rewards[-100:]) if len(training_rewards) >= 100 else np.mean(training_rewards)
            print(f"  Trial {trial}, Recent avg reward: {recent_avg:.3f}")
    
    # Testing phase
    print(f"Testing trained agent on {n_test_trials} trials...")
    test_rewards = []
    test_cue_rewards = np.zeros((max_rounds, k))  # Track rewards per (round, cue)
    test_cue_counts = np.zeros((max_rounds, k))   # Track selection counts
    
    for trial in range(n_test_trials):
        true_target = env.start_trial()
        
        # Run episode (no learning)
        for round_num in range(1, max_rounds + 1):
            available_cues = list(range(k))
            action = agent.select_action(round_num, available_cues)
            
            # Track selection
            test_cue_counts[round_num - 1, action] += 1
            
            # Only care about final decision
            if round_num == max_rounds:
                reward = 1.0 if action == true_target else 0.0
                test_cue_rewards[round_num - 1, action] += reward
                test_rewards.append(reward)
    
    test_accuracy = np.mean(test_rewards)
    print(f"Final test accuracy: {test_accuracy:.3f}")
    
    return agent, training_rewards, test_accuracy, q_value_evolution, training_cue_rewards, training_cue_counts, test_cue_rewards, test_cue_counts

#%%
# Run simulation
print("Starting vanilla RL simulation...")
agent, training_rewards, test_accuracy, q_evolution, train_cue_rewards, train_cue_counts, test_cue_rewards, test_cue_counts = run_vanilla_rl_simulation(
    k=4, p_t=0.9, p_f=0.5, max_rounds=10, 
    n_training_trials=1000, n_test_trials=100,
    alpha=0.1, gamma=0.9, beta=1.0, seed=42
)

#%%
# Plot results
sns.set_style("white")
plt.rcParams.update({'font.size': 12, 'axes.titlesize': 14, 'axes.labelsize': 12})

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Training rewards over time
ax = axes[0, 0]
window_size = 50
smoothed_rewards = np.convolve(training_rewards, np.ones(window_size)/window_size, mode='valid')
trials = np.arange(window_size-1, len(training_rewards))
ax.plot(trials, smoothed_rewards, linewidth=2, color='#2E86AB')
ax.axhline(y=0.25, color="gray", linestyle="--", alpha=0.7, label="Chance level")
ax.set_xlabel("Training Trial")
ax.set_ylabel("Average Reward (50-trial window)")
ax.set_title("Training Progress")
ax.legend()
ax.grid(True, alpha=0.3)
sns.despine(ax=ax)

# Plot 2: Final Q-values heatmap
ax = axes[0, 1]
im = ax.imshow(agent.q_values, cmap='viridis', aspect='auto')
ax.set_xlabel("Cue/Action")
ax.set_ylabel("Round")
ax.set_title("Final Q-Values Q(round, action)")
ax.set_xticks(range(4))
ax.set_xticklabels([f'Cue {i}' for i in range(4)])
ax.set_yticks(range(10))
ax.set_yticklabels([f'R{i+1}' for i in range(10)])
plt.colorbar(im, ax=ax)

# Plot 3: Q-value evolution over training
ax = axes[1, 0]
if len(q_evolution) > 1:
    # Show evolution of Q-values for round 10 (final round)
    final_round_q = [q_vals[9, :] for q_vals in q_evolution]  # Round 10 = index 9
    final_round_q = np.array(final_round_q)
    
    for cue in range(4):
        trials_recorded = np.arange(0, len(training_rewards), 50)[:len(final_round_q)]
        ax.plot(trials_recorded, final_round_q[:, cue], 
               label=f'Cue {cue}', linewidth=2, marker='o', markersize=4)
    
    ax.set_xlabel("Training Trial")
    ax.set_ylabel("Q-Value")
    ax.set_title("Q-Value Evolution (Final Round)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    sns.despine(ax=ax)

# Plot 4: Training reward distribution heatmap
ax = axes[1, 1]
# Calculate reward rates for training (only final round has rewards)
train_reward_rates = np.zeros((10, 4))
train_reward_rates[9, :] = train_cue_rewards[9, :] / np.maximum(train_cue_counts[9, :], 1)  # Avoid division by zero

im = ax.imshow(train_reward_rates, cmap='Reds', aspect='auto', vmin=0, vmax=1)
ax.set_xlabel("Cue/Action")
ax.set_ylabel("Round")
ax.set_title("Training Reward Rate by (Round, Cue)")
ax.set_xticks(range(4))
ax.set_xticklabels([f'Cue {i}' for i in range(4)])
ax.set_yticks(range(10))
ax.set_yticklabels([f'R{i+1}' for i in range(10)])

# Add text annotations for final round
for cue in range(4):
    if train_cue_counts[9, cue] > 0:
        rate = train_reward_rates[9, cue]
        count = int(train_cue_counts[9, cue])
        ax.text(cue, 9, f'{rate:.2f}\n({count})', ha='center', va='center', 
               color='white' if rate > 0.5 else 'black', fontsize=10)

plt.colorbar(im, ax=ax, label='Reward Rate')

plt.tight_layout()

# Save plot
os.makedirs("observers/bayesian_plots", exist_ok=True)
plt.savefig("observers/bayesian_plots/vanilla_rl_basic_results.png", dpi=300, bbox_inches='tight')
print("Plot saved to observers/bayesian_plots/vanilla_rl_basic_results.png")

plt.show()

#%%
# Additional plot: Testing reward distribution heatmap
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Training reward distribution
train_reward_rates = np.zeros((10, 4))
train_reward_rates[9, :] = train_cue_rewards[9, :] / np.maximum(train_cue_counts[9, :], 1)

im1 = ax1.imshow(train_reward_rates, cmap='Reds', aspect='auto', vmin=0, vmax=1)
ax1.set_xlabel("Cue/Action")
ax1.set_ylabel("Round")
ax1.set_title("Training: Reward Rate by (Round, Cue)")
ax1.set_xticks(range(4))
ax1.set_xticklabels([f'Cue {i}' for i in range(4)])
ax1.set_yticks(range(10))
ax1.set_yticklabels([f'R{i+1}' for i in range(10)])

# Add text annotations for final round
for cue in range(4):
    if train_cue_counts[9, cue] > 0:
        rate = train_reward_rates[9, cue]
        count = int(train_cue_counts[9, cue])
        ax1.text(cue, 9, f'{rate:.2f}\n({count})', ha='center', va='center', 
                color='white' if rate > 0.5 else 'black', fontsize=10)

plt.colorbar(im1, ax=ax1, label='Reward Rate')

# Testing reward distribution
test_reward_rates = np.zeros((10, 4))
test_reward_rates[9, :] = test_cue_rewards[9, :] / np.maximum(test_cue_counts[9, :], 1)

im2 = ax2.imshow(test_reward_rates, cmap='Blues', aspect='auto', vmin=0, vmax=1)
ax2.set_xlabel("Cue/Action")
ax2.set_ylabel("Round")
ax2.set_title("Testing: Reward Rate by (Round, Cue)")
ax2.set_xticks(range(4))
ax2.set_xticklabels([f'Cue {i}' for i in range(4)])
ax2.set_yticks(range(10))
ax2.set_yticklabels([f'R{i+1}' for i in range(10)])

# Add text annotations for final round
for cue in range(4):
    if test_cue_counts[9, cue] > 0:
        rate = test_reward_rates[9, cue]
        count = int(test_cue_counts[9, cue])
        ax2.text(cue, 9, f'{rate:.2f}\n({count})', ha='center', va='center', 
                color='white' if rate > 0.5 else 'black', fontsize=10)

plt.colorbar(im2, ax=ax2, label='Reward Rate')

plt.tight_layout()
plt.savefig("observers/bayesian_plots/vanilla_rl_reward_distributions.png", dpi=300, bbox_inches='tight')
print("Reward distribution plot saved to observers/bayesian_plots/vanilla_rl_reward_distributions.png")
plt.show()



# %%

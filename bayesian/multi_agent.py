#%%

"""
Multi-agent simulation for comparing different agent types in bias detection tasks.
"""


import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, List
from tqdm import tqdm

# Try to import notebook tqdm with better error handling
def get_tqdm_class():
    """Get the appropriate tqdm class based on environment."""
    try:
        # First check if we're in a notebook environment
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            # We're in Jupyter, try to import notebook tqdm
            try:
                from tqdm.notebook import tqdm as tqdm_notebook
                return tqdm_notebook
            except ImportError:
                # Fallback to regular tqdm if notebook version fails
                print("Warning: tqdm.notebook not available, using standard tqdm. "
                      "For better Jupyter support, install/update ipywidgets: pip install ipywidgets")
                return tqdm
        else:
            # Terminal IPython or other
            return tqdm
    except NameError:
        # Not in IPython at all
        return tqdm

# Detect if running in Jupyter notebook
def is_notebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter

from bayesian.environment import TemporalReasoningEnvironment, RLEnvironmentWrapper
from bayesian.bayesian_agent import BayesAgent, RandomPolicyAgent, MAPAgent, ActiveInferenceAgent, RLAgent

#%%

class MultiAgentSimulation:
    """Simulation manager for comparing multiple agent types."""
    
    def __init__(self, 
                 k: int = 4,
                 p_t: float = 0.9,
                 p_f: float = 0.5,
                 n_trials: int = 100,
                 rounds: List[int] = None,
                 verbose: bool = False,
                 log_results: bool = True,
                 seed: int = 42,
                 use_hidden_cues: bool = True,
                 min_available_cues: int = 1,
                 max_available_cues: int = 3):
        """
        Initialize the multi-agent simulation.
        
        Parameters:
        -----------
        k : int
            Number of possible locations/targets
        p_t : float
            Probability of correct color when cue matches true target
        p_f : float
            Probability of correct color when cue doesn't match true target
        n_trials : int
            Number of trials per round configuration
        rounds : List[int]
            List of round counts to test
        verbose : bool
            Whether to print detailed output
        log_results : bool
            Whether to save results to JSON
        seed : int
            Random seed
        use_hidden_cues : bool
            Whether to use hidden cues
        min_available_cues : int
            Minimum number of cues available per round
        max_available_cues : int
            Maximum number of cues available per round
        """
        self.k = k
        self.p_t = p_t
        self.p_f = p_f
        self.n_trials = n_trials
        self.rounds = rounds if rounds is not None else [5, 10, 15, 20]
        self.verbose = verbose
        self.log_results = log_results
        self.use_hidden_cues = use_hidden_cues
        self.min_available_cues = min_available_cues
        self.max_available_cues = max_available_cues
        
        self.rng = np.random.default_rng(seed)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create directories
        self.logs_dir = "logs"
        self.plots_dir = "bayesian_plots"
        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)
        
        # Initialize environment
        self.env = TemporalReasoningEnvironment(
            k, p_t, p_f, self.rng,
            use_hidden_cues=use_hidden_cues,
            min_available_cues=min_available_cues,
            max_available_cues=max_available_cues
        )
        
        # Initialize RL environment wrapper
        self.rl_env = RLEnvironmentWrapper(self.env, reward_value=1.0)
        
        # Results storage
        self.results = {}

    def initialize_agents(self):
        """Initialize all agent types."""
        agents = {
            'BayesAgent': BayesAgent(self.k, self.p_t, self.p_f),
            'RandomAgent': RandomPolicyAgent(self.k, self.rng),
            'MAPAgent': MAPAgent(self.k),
            'ActiveInference': ActiveInferenceAgent(self.k, self.p_t, self.p_f, beta=2.0),
            'RLAgent': RLAgent(self.k, self.p_t, self.p_f, alpha=0.1, gamma=0.9, 
                              beta=1.0, lambda_surprise=0.1, state_discretization=5)
        }
        return agents

    def run_single_trial_bayesian(self, agent, agent_name: str, n_rounds: int, trial_num: int) -> Dict:
        """
        Run a single trial for Bayesian-type agents (BayesAgent, ActiveInference).
        
        Parameters:
        -----------
        agent : Agent
            The agent instance
        agent_name : str
            Name of the agent type
        n_rounds : int
            Number of rounds for this trial
        trial_num : int
            Trial number (0-indexed)
            
        Returns:
        --------
        dict
            Trial results and statistics
        """
        # Reset agent
        agent.reset()
        
        # Start new trial
        true_z = self.env.start_trial()
        
        # Trial statistics
        trial_stats = {
            'trial_num': int(trial_num),
            'true_target': int(true_z),
            'n_rounds': int(n_rounds),
            'rounds_data': [],
            'final_decision': None,
            'final_metrics': {}
        }
        
        # Run n_rounds of observations and updates
        for round_num in range(n_rounds):
            cue, color, available_cues = self.env.sample_round(true_z)
            
            # For active inference agent, let it select the cue
            if hasattr(agent, 'select_action'):
                selected_cue = agent.select_action(available_cues)
                # Re-sample with the selected cue
                if selected_cue == cue:
                    # Use the sampled outcome
                    pass
                else:
                    # Need to sample again with the selected cue
                    if selected_cue == true_z:
                        p_color_1 = self.p_t
                    else:
                        p_color_1 = self.p_f
                    color = int(self.rng.random() < p_color_1)
                    cue = selected_cue
            
            # Update agent beliefs
            shannon_s, bayesian_s, pred_prob = agent.update(cue, color)
            
            round_data = {
                'round': int(round_num),
                'cue': int(cue),
                'color': int(color),
                'available_cues': [int(c) for c in available_cues],
                'shannon_surprise': float(shannon_s),
                'bayesian_surprise': float(bayesian_s),
                'predictive_prob': float(pred_prob),
                'posterior': agent.posterior.copy().tolist(),
                'entropy': float(agent.entropy)
            }
            trial_stats['rounds_data'].append(round_data)
        
        # Make final decision
        final_decision = agent.get_decision()
        trial_stats['final_decision'] = int(final_decision)
        
        # Calculate final metrics
        trial_stats['final_metrics'] = {
            'correct': int(agent.is_correct(final_decision, true_z)),
            'posterior_true_target': float(agent.posterior[true_z]),
            'entropy': float(agent.entropy),
            'map_probability': float(np.max(agent.posterior)),
            'mean_shannon_surprise': float(np.mean([r['shannon_surprise'] for r in trial_stats['rounds_data']])),
            'mean_bayesian_surprise': float(np.mean([r['bayesian_surprise'] for r in trial_stats['rounds_data']])),
            'mean_predictive_prob': float(np.mean([r['predictive_prob'] for r in trial_stats['rounds_data']]))
        }
        
        return trial_stats

    def run_single_trial_rl(self, agent: RLAgent, n_rounds: int, trial_num: int) -> Dict:
        """
        Run a single trial for RL agent using the reward wrapper.
        
        Parameters:
        -----------
        agent : RLAgent
            The RL agent instance
        n_rounds : int
            Number of rounds for this trial
        trial_num : int
            Trial number (0-indexed)
            
        Returns:
        --------
        dict
            Trial results and statistics
        """
        # Reset agent
        agent.reset()
        
        # Start new trial
        true_z = self.rl_env.start_trial(max_episode_length=n_rounds)
        
        # Trial statistics
        trial_stats = {
            'trial_num': int(trial_num),
            'true_target': int(true_z),
            'n_rounds': int(n_rounds),
            'rounds_data': [],
            'final_decision': None,
            'final_metrics': {}
        }
        
        # Run n_rounds of sampling
        for round_num in range(n_rounds):
            # Get current state
            current_state = agent.discretize_state(agent.posterior)
            
            # Get available cues (sample from environment to get them)
            cue, color, available_cues, reward, done = self.rl_env.step(0, is_final_step=False)
            
            # Agent selects action
            selected_action = agent.select_action(available_cues, current_state)
            
            # Re-sample with selected action if different
            if selected_action != cue:
                if selected_action == true_z:
                    p_color_1 = self.p_t
                else:
                    p_color_1 = self.p_f
                color = int(self.rng.random() < p_color_1)
                cue = selected_action
            
            # Update agent beliefs
            shannon_s, bayesian_s, pred_prob = agent.update(cue, color)
            
            # Calculate surprise reward
            surprise_reward = -bayesian_s  # Negative Bayesian surprise
            
            # Get next state
            next_state = agent.discretize_state(agent.posterior)
            
            # Update Q-values
            agent.update_q_values(current_state, selected_action, 0.0, surprise_reward, next_state, False)
            
            round_data = {
                'round': int(round_num),
                'cue': int(cue),
                'color': int(color),
                'available_cues': [int(c) for c in available_cues],
                'selected_action': int(selected_action),
                'shannon_surprise': float(shannon_s),
                'bayesian_surprise': float(bayesian_s),
                'predictive_prob': float(pred_prob),
                'surprise_reward': float(surprise_reward),
                'posterior': agent.posterior.copy().tolist(),
                'entropy': float(agent.entropy)
            }
            trial_stats['rounds_data'].append(round_data)
        
        # Make final decision
        final_decision = agent.get_decision()
        trial_stats['final_decision'] = int(final_decision)
        
        # Calculate final reward
        final_reward = 1.0 if final_decision == true_z else 0.0
        
        # Update Q-values for final decision
        final_state = agent.discretize_state(agent.posterior)
        agent.update_q_values(final_state, final_decision, final_reward, 0.0, final_state, True)
        
        # Calculate final metrics
        trial_stats['final_metrics'] = {
            'correct': int(agent.is_correct(final_decision, true_z)),
            'final_reward': float(final_reward),
            'posterior_true_target': float(agent.posterior[true_z]),
            'entropy': float(agent.entropy),
            'map_probability': float(np.max(agent.posterior)),
            'mean_shannon_surprise': float(np.mean([r['shannon_surprise'] for r in trial_stats['rounds_data']])),
            'mean_bayesian_surprise': float(np.mean([r['bayesian_surprise'] for r in trial_stats['rounds_data']])),
            'mean_predictive_prob': float(np.mean([r['predictive_prob'] for r in trial_stats['rounds_data']]))
        }
        
        return trial_stats

    def run_single_trial_decision_only(self, agent, agent_name: str, learned_posterior, true_z: int, trial_num: int) -> Dict:
        """
        Run a single trial for decision-only agents (RandomAgent, MAPAgent).
        
        Parameters:
        -----------
        agent : Agent
            The agent instance
        agent_name : str
            Name of the agent type
        learned_posterior : np.array
            Posterior learned by Bayesian agent
        true_z : int
            True target location
        trial_num : int
            Trial number
            
        Returns:
        --------
        dict
            Trial results and statistics
        """
        # Make decision
        final_decision = agent.get_decision(learned_posterior)
        
        # Trial statistics
        trial_stats = {
            'trial_num': int(trial_num),
            'true_target': int(true_z),
            'final_decision': int(final_decision),
            'final_metrics': {
                'correct': int(agent.is_correct(final_decision, true_z)),
                'posterior_true_target': float(learned_posterior[true_z]),
                'entropy': float(-np.sum(learned_posterior * np.log(learned_posterior + 1e-12))),
                'map_probability': float(np.max(learned_posterior))
            }
        }
        
        return trial_stats

    def run_simulation(self, n_rounds: int, pbar=None) -> Dict:
        """
        Run simulation for all agents with a specific round count.
        
        Parameters:
        -----------
        n_rounds : int
            Number of rounds
        pbar : tqdm, optional
            Global progress bar to update
            
        Returns:
        --------
        dict
            Aggregated simulation results for all agents
        """
        agents = self.initialize_agents()
        all_results = {}
        
        for agent_name, agent in agents.items():
            if self.verbose:
                print(f"Running {agent_name} with {n_rounds} rounds...")
            
            agent_trials = []
            
            for trial_num in range(self.n_trials):
                if agent_name in ['BayesAgent', 'ActiveInference']:
                    trial_stats = self.run_single_trial_bayesian(agent, agent_name, n_rounds, trial_num)
                elif agent_name == 'RLAgent':
                    trial_stats = self.run_single_trial_rl(agent, n_rounds, trial_num)
                else:
                    # For decision-only agents, use Bayesian posterior
                    bayes_agent = BayesAgent(self.k, self.p_t, self.p_f)
                    bayes_trial = self.run_single_trial_bayesian(bayes_agent, 'BayesAgent', n_rounds, trial_num)
                    trial_stats = self.run_single_trial_decision_only(
                        agent, agent_name, bayes_agent.posterior, bayes_trial['true_target'], trial_num
                    )
                
                agent_trials.append(trial_stats)
                
                if pbar:
                    pbar.update(1)
            
            # Aggregate results for this agent
            all_results[agent_name] = self.aggregate_trials(agent_trials, agent_name, n_rounds)
        
        return all_results

    def aggregate_trials(self, trials: List[Dict], agent_name: str, n_rounds: int) -> Dict:
        """
        Aggregate results from multiple trials for a single agent.
        
        Parameters:
        -----------
        trials : List[Dict]
            List of trial results
        agent_name : str
            Name of the agent type
        n_rounds : int
            Number of rounds
            
        Returns:
        --------
        dict
            Aggregated metrics
        """
        # Extract metrics
        accuracy = [t['final_metrics']['correct'] for t in trials]
        posterior_true = [t['final_metrics']['posterior_true_target'] for t in trials]
        entropy_values = [t['final_metrics']['entropy'] for t in trials]
        map_probabilities = [t['final_metrics']['map_probability'] for t in trials]
        
        # For agents with round data
        if 'rounds_data' in trials[0]:
            shannon_surprises = [t['final_metrics']['mean_shannon_surprise'] for t in trials]
            bayesian_surprises = [t['final_metrics']['mean_bayesian_surprise'] for t in trials]
            predictive_probs = [t['final_metrics']['mean_predictive_prob'] for t in trials]
        else:
            shannon_surprises = [0.0] * len(trials)
            bayesian_surprises = [0.0] * len(trials)
            predictive_probs = [1.0] * len(trials)
        
        aggregated = {
            'agent_name': agent_name,
            'timestamp': self.timestamp,
            'n_rounds': n_rounds,
            'n_trials': self.n_trials,
            'k': self.k,
            'p_t': self.p_t,
            'p_f': self.p_f,
            'use_hidden_cues': self.use_hidden_cues,
            'min_available_cues': self.min_available_cues,
            'max_available_cues': self.max_available_cues,
            'metrics': {
                'accuracy': {
                    'mean': float(np.mean(accuracy)),
                    'std': float(np.std(accuracy)),
                    'values': accuracy
                },
                'posterior_true_target': {
                    'mean': float(np.mean(posterior_true)),
                    'std': float(np.std(posterior_true)),
                    'values': posterior_true
                },
                'entropy_values': {
                    'mean': float(np.mean(entropy_values)),
                    'std': float(np.std(entropy_values)),
                    'values': entropy_values
                },
                'map_probabilities': {
                    'mean': float(np.mean(map_probabilities)),
                    'std': float(np.std(map_probabilities)),
                    'values': map_probabilities
                },
                'shannon_surprises': {
                    'mean': float(np.mean(shannon_surprises)),
                    'std': float(np.std(shannon_surprises)),
                    'values': shannon_surprises
                },
                'bayesian_surprises': {
                    'mean': float(np.mean(bayesian_surprises)),
                    'std': float(np.std(bayesian_surprises)),
                    'values': bayesian_surprises
                },
                'predictive_probabilities': {
                    'mean': float(np.mean(predictive_probs)),
                    'std': float(np.std(predictive_probs)),
                    'values': predictive_probs
                }
            },
            'raw_trials': trials
        }
        
        return aggregated

    def run_all_simulations(self) -> Dict:
        """
        Run simulations for all agents and round configurations.
        
        Returns:
        --------
        dict
            Complete simulation results
        """
        start_time = time.time()
        
        # Calculate total number of trials
        total_trials = len(self.rounds) * self.n_trials * 5  # 5 agent types
        
        # Create global progress bar - get appropriate tqdm class
        tqdm_class = get_tqdm_class()
        with tqdm_class(total=total_trials, desc="Running multi-agent simulations", 
                       disable=not self.verbose) as pbar:
            
            for n_rounds in self.rounds:
                pbar.set_postfix_str(f"{n_rounds} rounds")
                simulation_results = self.run_simulation(n_rounds, pbar)
                self.results[n_rounds] = simulation_results
                
                # Save individual results
                if self.log_results:
                    self.save_simulation_results(n_rounds, simulation_results)
        
        total_elapsed = time.time() - start_time
        
        if self.verbose:
            print(f"All multi-agent simulations completed in {total_elapsed:.2f} seconds")
        
        return self.results

    def save_simulation_results(self, n_rounds: int, results: Dict):
        """Save simulation results to JSON file."""
        # Create multi-agent log directory
        multi_agent_log_dir = os.path.join(self.logs_dir, "multi_agent")
        os.makedirs(multi_agent_log_dir, exist_ok=True)
        
        # Create filename
        filename = f"multi_agent_{n_rounds}r_{self.k}k_{self.n_trials}t_{self.timestamp}.json"
        filepath = os.path.join(multi_agent_log_dir, filename)
        
        # Custom JSON encoder to handle numpy types
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super(NumpyEncoder, self).default(obj)
        
        # Save results
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, cls=NumpyEncoder)
        
        if self.verbose:
            print(f"Multi-agent results saved to {filepath}")

    def plot_comparison(self, save_plots: bool = True):
        """
        Create comparison plots for all agents.
        
        Parameters:
        -----------
        save_plots : bool
            Whether to save plots to files
        """
        if not self.results:
            print("No results to plot. Run simulations first.")
            return
        
        # Set style
        sns.set_style("white")
        colors = sns.color_palette("Set2", 5)
        agent_colors = {
            'BayesAgent': colors[0],
            'ActiveInference': colors[1], 
            'RLAgent': colors[2],
            'MAPAgent': colors[3],
            'RandomAgent': colors[4]
        }
        
        plt.rcParams.update({'font.size': 12, 'axes.titlesize': 14, 'axes.labelsize': 12, 'legend.fontsize': 11})
        
        # Create comparison plots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        rounds_list = sorted(self.results.keys())
        
        # Plot 1: Accuracy comparison
        ax = axes[0]
        for agent_name in ['BayesAgent', 'ActiveInference', 'RLAgent', 'MAPAgent', 'RandomAgent']:
            accuracies = []
            stds = []
            for n_rounds in rounds_list:
                if agent_name in self.results[n_rounds]:
                    acc_data = self.results[n_rounds][agent_name]['metrics']['accuracy']
                    accuracies.append(acc_data['mean'])
                    stds.append(acc_data['std'])
                else:
                    accuracies.append(0)
                    stds.append(0)
            
            accuracies = np.array(accuracies)
            stds = np.array(stds)
            
            ax.plot(rounds_list, accuracies, "-o", label=agent_name, 
                   color=agent_colors[agent_name], linewidth=2, markersize=6)
            ax.fill_between(rounds_list, accuracies - stds, accuracies + stds, 
                           color=agent_colors[agent_name], alpha=0.2)
        
        ax.axhline(y=1/self.k, color="gray", linestyle="--", alpha=0.7, label="Chance level")
        ax.set_xlabel("Number of Rounds")
        ax.set_ylabel("Accuracy")
        ax.set_title("Decision Accuracy Comparison")
        ax.legend()
        ax.grid(True, alpha=0.3)
        sns.despine(ax=ax)
        
        # Plot 2: Entropy comparison
        ax = axes[1]
        for agent_name in ['BayesAgent', 'ActiveInference', 'RLAgent']:
            entropies = []
            stds = []
            for n_rounds in rounds_list:
                if agent_name in self.results[n_rounds]:
                    ent_data = self.results[n_rounds][agent_name]['metrics']['entropy_values']
                    entropies.append(ent_data['mean'])
                    stds.append(ent_data['std'])
                else:
                    entropies.append(np.log(self.k))
                    stds.append(0)
            
            entropies = np.array(entropies)
            stds = np.array(stds)
            
            ax.plot(rounds_list, entropies, "-o", label=agent_name, 
                   color=agent_colors[agent_name], linewidth=2, markersize=6)
            ax.fill_between(rounds_list, entropies - stds, entropies + stds, 
                           color=agent_colors[agent_name], alpha=0.2)
        
        ax.axhline(y=np.log(self.k), color="gray", linestyle="--", alpha=0.7, label="Max entropy")
        ax.set_xlabel("Number of Rounds")
        ax.set_ylabel("Posterior Entropy (nats)")
        ax.set_title("Posterior Uncertainty Comparison")
        ax.legend()
        ax.grid(True, alpha=0.3)
        sns.despine(ax=ax)
        
        # Plot 3: Shannon surprise comparison
        ax = axes[2]
        for agent_name in ['BayesAgent', 'ActiveInference', 'RLAgent']:
            surprises = []
            stds = []
            for n_rounds in rounds_list:
                if agent_name in self.results[n_rounds]:
                    surp_data = self.results[n_rounds][agent_name]['metrics']['shannon_surprises']
                    surprises.append(surp_data['mean'])
                    stds.append(surp_data['std'])
                else:
                    surprises.append(0)
                    stds.append(0)
            
            surprises = np.array(surprises)
            stds = np.array(stds)
            
            ax.plot(rounds_list, surprises, "-o", label=agent_name, 
                   color=agent_colors[agent_name], linewidth=2, markersize=6)
            ax.fill_between(rounds_list, surprises - stds, surprises + stds, 
                           color=agent_colors[agent_name], alpha=0.2)
        
        ax.set_xlabel("Number of Rounds")
        ax.set_ylabel("Mean Shannon Surprise")
        ax.set_title("Shannon Surprise Comparison")
        ax.legend()
        ax.grid(True, alpha=0.3)
        sns.despine(ax=ax)
        
        # Plot 4: Bayesian surprise comparison
        ax = axes[3]
        for agent_name in ['BayesAgent', 'ActiveInference', 'RLAgent']:
            surprises = []
            stds = []
            for n_rounds in rounds_list:
                if agent_name in self.results[n_rounds]:
                    surp_data = self.results[n_rounds][agent_name]['metrics']['bayesian_surprises']
                    surprises.append(surp_data['mean'])
                    stds.append(surp_data['std'])
                else:
                    surprises.append(0)
                    stds.append(0)
            
            surprises = np.array(surprises)
            stds = np.array(stds)
            
            ax.plot(rounds_list, surprises, "-o", label=agent_name, 
                   color=agent_colors[agent_name], linewidth=2, markersize=6)
            ax.fill_between(rounds_list, surprises - stds, surprises + stds, 
                           color=agent_colors[agent_name], alpha=0.2)
        
        ax.set_xlabel("Number of Rounds")
        ax.set_ylabel("Mean Bayesian Surprise")
        ax.set_title("Bayesian Surprise Comparison")
        ax.legend()
        ax.grid(True, alpha=0.3)
        sns.despine(ax=ax)
        
        plt.tight_layout()
        
        if save_plots:
            plot_filename = f"{self.plots_dir}/multi_agent_comparison_{self.timestamp}.png"
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            if self.verbose:
                print(f"Multi-agent comparison plot saved to {plot_filename}")
        
        plt.show()

#%%

def main():
    """Main function to run multi-agent simulation."""
    # Initialize simulation
    sim = MultiAgentSimulation(
        k=4,
        p_t=0.9,
        p_f=0.5,
        n_trials=50,
        rounds=np.arange(1, 100),  # Test rounds from 1 to 100
        verbose=True,
        log_results=True,
        seed=42,
        use_hidden_cues=True,
        min_available_cues=1,
        max_available_cues=3
    )
    
    # Run all simulations
    results = sim.run_all_simulations()
    
    # Create comparison plots
    sim.plot_comparison(save_plots=True)
    
    return results


if __name__ == "__main__":
    main()

# %%

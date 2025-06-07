import os
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

from .environment import TemporalReasoningEnvironment
from .bayesian_agent import BayesAgent, RandomPolicyAgent, MAPAgent


class BayesianSimulation:
    """Simulation manager for Bayesian agents with JSON logging."""
    
    def __init__(self, 
                 k: int = 4,
                 p_t: float = 0.9,
                 p_f: float = 0.5,
                 n_trials: int = 100,
                 rounds: List[int] = None,
                 agent_types: List[str] = None,
                 verbose: bool = False,
                 log_results: bool = True,
                 seed: int = 42,
                 use_hidden_cues: bool = False,
                 min_available_cues: int = None,
                 max_available_cues: int = None):
        """
        Initialize the simulation.
        
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
        agent_types : List[str]
            List of agent types to test
        verbose : bool
            Whether to print detailed output
        log_results : bool
            Whether to save results to JSON
        seed : int
            Random seed
        use_hidden_cues : bool
            Whether to use hidden cues (subset of cues available each round)
        min_available_cues : int
            Minimum number of cues available per round (default: 1)
        max_available_cues : int
            Maximum number of cues available per round (default: k)
        """
        self.k = k
        self.p_t = p_t
        self.p_f = p_f
        self.n_trials = n_trials
        # Convert numpy arrays to regular Python lists to avoid JSON serialization issues
        if rounds is not None:
            self.rounds = [int(r) for r in rounds]
        else:
            self.rounds = list(range(1, 15))
        self.agent_types = agent_types if agent_types is not None else ["BayesAgent", "RandomPolicyAgent", "MAPAgent"]
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
        
        # Initialize environment with hidden cues parameters
        self.env = TemporalReasoningEnvironment(
            k, p_t, p_f, self.rng,
            use_hidden_cues=use_hidden_cues,
            min_available_cues=min_available_cues,
            max_available_cues=max_available_cues
        )
        
        # Results storage with compact dictionary structure
        self.results = {}

    def run_single_trial(self, agent_name: str, n_rounds: int, trial_num: int) -> Dict:
        """
        Run a single trial for a specific agent and round count.
        
        Parameters:
        -----------
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
        # Start new trial with random true target
        true_z = self.env.start_trial()
        
        # Initialize agents
        bayes_agent = BayesAgent(self.k, self.p_t, self.p_f)
        random_agent = RandomPolicyAgent(self.k, self.rng)
        map_agent = MAPAgent(self.k)
        
        # Trial statistics
        trial_stats = {
            'trial_num': int(trial_num),
            'true_target': int(true_z),
            'n_rounds': int(n_rounds),
            'rounds_data': [],
            'final_decisions': {},
            'final_metrics': {}
        }
        
        # Run n_rounds of observations and updates
        for round_num in range(n_rounds):
            cue, color, available_cues = self.env.sample_round(true_z)
            shannon_s, bayesian_s, pred_prob = bayes_agent.update(cue, color)
            
            round_data = {
                'round': int(round_num),
                'cue': int(cue),
                'color': int(color),
                'available_cues': [int(c) for c in available_cues],
                'shannon_surprise': float(shannon_s),
                'bayesian_surprise': float(bayesian_s),
                'predictive_prob': float(pred_prob),
                'posterior': bayes_agent.posterior.copy().tolist(),
                'entropy': float(bayes_agent.entropy)
            }
            trial_stats['rounds_data'].append(round_data)
        
        # After all rounds, get final posterior and make decisions
        learned_posterior = bayes_agent.posterior.copy()
        
        # Make final decisions
        random_decision = random_agent.get_decision(learned_posterior)
        map_decision = map_agent.get_decision(learned_posterior)
        
        # Store final decisions and correctness
        trial_stats['final_decisions'] = {
            'random': int(random_decision),
            'map': int(map_decision),
            'random_correct': int(random_agent.is_correct(random_decision, true_z)),
            'map_correct': int(map_agent.is_correct(map_decision, true_z))
        }
        
        # Store final metrics
        trial_stats['final_metrics'] = {
            'posterior_true_target': float(learned_posterior[true_z]),
            'entropy': float(bayes_agent.entropy),
            'map_probability': float(map_agent.get_map_probability(learned_posterior)),
            'mean_shannon_surprise': float(np.mean([r['shannon_surprise'] for r in trial_stats['rounds_data']])),
            'mean_bayesian_surprise': float(np.mean([r['bayesian_surprise'] for r in trial_stats['rounds_data']])),
            'mean_predictive_prob': float(np.mean([r['predictive_prob'] for r in trial_stats['rounds_data']]))
        }
        
        return trial_stats

    def run_simulation(self, agent_name: str, n_rounds: int, pbar=None) -> Dict:
        """
        Run simulation for a specific agent and round count.
        
        Parameters:
        -----------
        agent_name : str
            Name of the agent type
        n_rounds : int
            Number of rounds
        pbar : tqdm, optional
            Global progress bar to update
            
        Returns:
        --------
        dict
            Aggregated simulation results
        """
        all_trials = []
        start_time = time.time()
        
        for trial_num in range(self.n_trials):
            trial_stats = self.run_single_trial(agent_name, n_rounds, trial_num)
            all_trials.append(trial_stats)
            if pbar:
                pbar.update(1)
        
        total_time = time.time() - start_time
        
        # Aggregate results
        aggregated_results = self.aggregate_trials(all_trials, agent_name, n_rounds, total_time)
        
        return aggregated_results

    def aggregate_trials(self, trials: List[Dict], agent_name: str, n_rounds: int, total_time: float) -> Dict:
        """
        Aggregate results from multiple trials.
        
        Parameters:
        -----------
        trials : List[Dict]
            List of trial results
        agent_name : str
            Name of the agent type
        n_rounds : int
            Number of rounds
        total_time : float
            Total simulation time
            
        Returns:
        --------
        dict
            Aggregated metrics
        """
        # Extract metrics from all trials
        metrics = {
            'random_accuracy': [t['final_decisions']['random_correct'] for t in trials],
            'map_accuracy': [t['final_decisions']['map_correct'] for t in trials],
            'map_probabilities': [t['final_metrics']['map_probability'] for t in trials],
            'posterior_true_target': [t['final_metrics']['posterior_true_target'] for t in trials],
            'entropy_values': [t['final_metrics']['entropy'] for t in trials],
            'shannon_surprises': [t['final_metrics']['mean_shannon_surprise'] for t in trials],
            'bayesian_surprises': [t['final_metrics']['mean_bayesian_surprise'] for t in trials],
            'predictive_probabilities': [t['final_metrics']['mean_predictive_prob'] for t in trials]
        }
        
        # Calculate means and standard deviations
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
            'total_time': total_time,
            'metrics': {}
        }
        
        for metric_name, values in metrics.items():
            aggregated['metrics'][metric_name] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'values': values  # Store raw values for plotting
            }
        
        # Store raw trial data
        aggregated['raw_trials'] = trials
        
        return aggregated

    def run_all_simulations(self) -> Dict:
        """
        Run simulations for all agent types and round configurations.
        
        Returns:
        --------
        dict
            Complete simulation results
        """
        start_time = time.time()
        
        # Calculate total number of trials across all configurations
        total_trials = len(self.agent_types) * len(self.rounds) * self.n_trials
        
        # Create global progress bar - get appropriate tqdm class
        tqdm_class = get_tqdm_class()
        with tqdm_class(total=total_trials, desc="Running simulations", disable=not self.verbose, 
                       bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}') as pbar:
            for agent_name in self.agent_types:
                if agent_name not in self.results:
                    self.results[agent_name] = {}
                
                for n_rounds in self.rounds:
                    pbar.set_postfix_str(f"{agent_name} - {n_rounds}r")
                    simulation_results = self.run_simulation(agent_name, n_rounds, pbar)
                    self.results[agent_name][n_rounds] = simulation_results
                    
                    # Save individual simulation results
                    if self.log_results:
                        self.save_simulation_results(agent_name, n_rounds, simulation_results)
        
        total_elapsed = time.time() - start_time
        
        if self.verbose:
            print(f"All simulations completed in {total_elapsed:.2f} seconds")
        
        return self.results

    def save_simulation_results(self, agent_name: str, n_rounds: int, results: Dict):
        """
        Save simulation results to JSON file.
        
        Parameters:
        -----------
        agent_name : str
            Name of the agent type
        n_rounds : int
            Number of rounds
        results : Dict
            Simulation results
        """
        # Create agent-specific log directory
        agent_log_dir = os.path.join(self.logs_dir, "bayesian", agent_name)
        os.makedirs(agent_log_dir, exist_ok=True)
        
        # Create filename following TaskManager pattern
        filename = f"{agent_name}_{n_rounds}r_{self.k}k_{self.n_trials}t_{self.timestamp}.json"
        filepath = os.path.join(agent_log_dir, filename)
        
        # Save results
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        if self.verbose:
            print(f"Results saved to {filepath}")

    def plot_results(self, save_plots: bool = True):
        """
        Create plots of the simulation results.
        
        Parameters:
        -----------
        save_plots : bool
            Whether to save plots to files
        """
        # Set seaborn style and rocket color palette (no grid)
        sns.set_style("white")
        rocket_colors = sns.color_palette("rocket", 6)
        
        # Set larger font sizes
        plt.rcParams.update({'font.size': 14, 'axes.titlesize': 16, 'axes.labelsize': 14, 'legend.fontsize': 12})
        
        for agent_name in self.agent_types:
            if agent_name not in self.results:
                continue
                
            fig, ax = plt.subplots(2, 3, figsize=(15, 8))
            
            # Extract data for plotting
            rounds_list = sorted(self.results[agent_name].keys())
            
            # Initialize arrays for each metric
            plot_data = {}
            for metric in ['random_accuracy', 'map_accuracy', 'map_probabilities', 
                          'posterior_true_target', 'entropy_values', 'shannon_surprises', 'bayesian_surprises']:
                plot_data[metric] = {
                    'means': [],
                    'stds': []
                }
                
                for n_rounds in rounds_list:
                    results = self.results[agent_name][n_rounds]
                    plot_data[metric]['means'].append(results['metrics'][metric]['mean'])
                    plot_data[metric]['stds'].append(results['metrics'][metric]['std'])
                
                plot_data[metric]['means'] = np.array(plot_data[metric]['means'])
                plot_data[metric]['stds'] = np.array(plot_data[metric]['stds'])
            
            # Plot 1: Decision Policy Performance
            ax[0,0].plot(rounds_list, plot_data['random_accuracy']['means'], "-", 
                        label="Random Policy", color=rocket_colors[0], linewidth=2)
            ax[0,0].fill_between(rounds_list, 
                               plot_data['random_accuracy']['means'] - plot_data['random_accuracy']['stds'],
                               plot_data['random_accuracy']['means'] + plot_data['random_accuracy']['stds'], 
                               color=rocket_colors[0], alpha=0.2)
            ax[0,0].plot(rounds_list, plot_data['map_accuracy']['means'], "-", 
                        label="MAP Agent", color=rocket_colors[5], linewidth=2)
            ax[0,0].fill_between(rounds_list, 
                               plot_data['map_accuracy']['means'] - plot_data['map_accuracy']['stds'],
                               plot_data['map_accuracy']['means'] + plot_data['map_accuracy']['stds'], 
                               color=rocket_colors[5], alpha=0.2)
            ax[0,0].axhline(y=1/self.k, color="gray", linestyle="--", alpha=0.7, label="Chance level")
            ax[0,0].set(xlabel="Rounds", ylabel="Decision Accuracy", title="Decision Policy Performance")
            ax[0,0].legend()
            ax[0,0].set_ylim(0, 1)
            sns.despine(ax=ax[0,0])
            
            # Plot 2: MAP Probability
            ax[0,1].plot(rounds_list, plot_data['map_probabilities']['means'], "-", 
                        label="MAP Probability", color=rocket_colors[4], linewidth=2)
            ax[0,1].fill_between(rounds_list, 
                               plot_data['map_probabilities']['means'] - plot_data['map_probabilities']['stds'],
                               plot_data['map_probabilities']['means'] + plot_data['map_probabilities']['stds'], 
                               color=rocket_colors[4], alpha=0.2)
            ax[0,1].set(xlabel="Rounds", ylabel="Mean MAP Probability", title="Confidence in MAP Decision")
            ax[0,1].legend()
            sns.despine(ax=ax[0,1])
            
            # Plot 3: Posterior Accuracy
            ax[0,2].plot(rounds_list, plot_data['posterior_true_target']['means'], "-", 
                        label="P(true target)", color=rocket_colors[1], linewidth=2)
            ax[0,2].fill_between(rounds_list, 
                               plot_data['posterior_true_target']['means'] - plot_data['posterior_true_target']['stds'],
                               plot_data['posterior_true_target']['means'] + plot_data['posterior_true_target']['stds'], 
                               color=rocket_colors[1], alpha=0.2)
            ax[0,2].set(xlabel="Rounds", ylabel="Mean P(true target)", title="Posterior Accuracy")
            ax[0,2].legend()
            sns.despine(ax=ax[0,2])
            
            # Plot 4: Entropy
            ax[1,0].plot(rounds_list, plot_data['entropy_values']['means'], "-", 
                        label="Entropy", color=rocket_colors[2], linewidth=2)
            ax[1,0].fill_between(rounds_list, 
                               plot_data['entropy_values']['means'] - plot_data['entropy_values']['stds'],
                               plot_data['entropy_values']['means'] + plot_data['entropy_values']['stds'], 
                               color=rocket_colors[2], alpha=0.2)
            ax[1,0].axhline(y=np.log(self.k), color="gray", linestyle="--", alpha=0.7, label="Max entropy")
            ax[1,0].set(xlabel="Rounds", ylabel="Mean Entropy (nats)", title="Posterior Uncertainty")
            ax[1,0].legend()
            sns.despine(ax=ax[1,0])
            
            # Plot 5: Shannon Surprise
            ax[1,1].plot(rounds_list, plot_data['shannon_surprises']['means'], "-", 
                        label="Shannon Surprise", color=rocket_colors[3], linewidth=2)
            ax[1,1].fill_between(rounds_list, 
                               plot_data['shannon_surprises']['means'] - plot_data['shannon_surprises']['stds'],
                               plot_data['shannon_surprises']['means'] + plot_data['shannon_surprises']['stds'], 
                               color=rocket_colors[3], alpha=0.2)
            ax[1,1].set(xlabel="Rounds", ylabel="Mean Shannon Surprise", title="Shannon Surprise")
            ax[1,1].legend()
            sns.despine(ax=ax[1,1])
            
            # Plot 6: Bayesian Surprise
            ax[1,2].plot(rounds_list, plot_data['bayesian_surprises']['means'], "-", 
                        label="Bayesian Surprise", color=rocket_colors[4], linewidth=2)
            ax[1,2].fill_between(rounds_list, 
                               plot_data['bayesian_surprises']['means'] - plot_data['bayesian_surprises']['stds'],
                               plot_data['bayesian_surprises']['means'] + plot_data['bayesian_surprises']['stds'], 
                               color=rocket_colors[4], alpha=0.2)
            ax[1,2].set(xlabel="Rounds", ylabel="Mean Bayesian Surprise", title="Bayesian Surprise")
            ax[1,2].legend()
            sns.despine(ax=ax[1,2])
            
            plt.tight_layout()
            
            if save_plots:
                plot_filename = f"{self.plots_dir}/{agent_name}_results_{self.timestamp}.png"
                plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
                if self.verbose:
                    print(f"Plot saved to {plot_filename}")
            
            plt.show()


def main():
    """Main function to run the simulation."""
    # Initialize simulation
    sim = BayesianSimulation(
        k=4,
        p_t=0.9,
        p_f=0.5,
        n_trials=100,
        rounds=list(range(1, 15)),
        agent_types=["BayesAgent"],
        verbose=True,
        log_results=True,
        seed=42
    )
    
    # Run all simulations
    results = sim.run_all_simulations()
    
    # Create plots
    sim.plot_results(save_plots=True)
    
    return results


if __name__ == "__main__":
    main()

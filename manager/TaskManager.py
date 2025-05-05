import os
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from datetime import datetime
from typing import Dict
from tqdm import tqdm
from sae.hooks import Hook
from transformers import AutoTokenizer

class TaskManager:
    def __init__(self,
                 agents=None,
                 rounds=None,
                 quadrants=None,
                 n_simulations=10,
                 n_trials=1,
                 num_cues=1,
                 device="cuda:0",
                 verbose=False,
                 output_dir="simulation_results",
                 openai_api_key=None,
                 anthropic_api_key=None,
                 use_unsloth=True,
                 reasoning_mode="time",
                 min_thinking_time=5.0,
                 max_thinking_time=10.0,
                 min_thinking_tokens=200,
                 max_thinking_tokens=500,
                 task_type=None,
                 log_stats = False,
                 activation_layer=None,
                 ):
        """
        Initialize task manager with benchmark capabilities.

        Parameters:
        -----------
        agents : list or str
            List of agent model names or a single agent model name
        rounds : list or int
            List of round counts or a single round count to test
        quadrants : list or int
            List of quadrant counts or a single quadrant count to test
        n_simulations : int
            Number of simulations per configuration
        n_trials : int
            Number of trials to run for each simulation
        num_cues : int
            Number of cues per quadrant
        device : str
            Device to use for model inference
        verbose : bool
            Whether to print detailed output
        output_dir : str
            Directory to save results (not used, kept for backward compatibility)
        openai_api_key : str
            OpenAI API key (if applicable)
        anthropic_api_key : str
            Anthropic API key (if applicable)
        use_unsloth : bool
            Whether to use Unsloth optimization
        reasoning_mode : str
            Mode for reasoning models: 'time' or 'tokens'
        min_thinking_time : float
            Minimum thinking time in seconds (for time mode)
        max_thinking_time : float
            Maximum thinking time in seconds (for time mode)
        min_thinking_tokens : int
            Minimum number of thinking tokens (for token mode)
        max_thinking_tokens : int
            Maximum number of thinking tokens (for token mode)
        task_type : TaskSelector
            Type of task to run
        log_stats : bool
            Whether to log statistics during the benchmark
        """
        from agents.LLMagent import LLMagent  # Import here to avoid circular imports

        # Convert single values to lists for consistent processing
        self.agents = [agents] if isinstance(agents, str) else agents
        self.rounds = [rounds] if isinstance(rounds, int) else rounds
        self.quadrants = [quadrants] if isinstance(quadrants, int) else quadrants
        self.task_type = task_type

        self.n_simulations = n_simulations
        self.n_trials = n_trials
        self.num_cues = num_cues
        self.device = device
        self.verbose = verbose
        self.openai_api_key = openai_api_key
        self.anthropic_api_key = anthropic_api_key
        self.use_unsloth = use_unsloth

        # Store reasoning parameters
        self.reasoning_mode = reasoning_mode
        self.min_thinking_time = min_thinking_time
        self.max_thinking_time = max_thinking_time
        self.min_thinking_tokens = min_thinking_tokens
        self.max_thinking_tokens = max_thinking_tokens

        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.conversation_history = None
        self.current_agent = None
        self.is_reasoning_model = False
        self.thinking_times = []

        # Create new directory structure
        self.benchmarks_plots_dir = "benchmarks_plots"
        self.logs_dir = "logs"
        os.makedirs(self.benchmarks_plots_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)

        # Initialize results storage
        self.results = {}

        # Get reasoning models list from LLMagent
        self.reasoning_models = LLMagent.get_reasoning_models()
        
        self.log_stats = log_stats

        self.activations_layer = activation_layer
        

    def initialize_agent(self, agent_name):
        """Initialize an LLM agent with the specified model."""
        from agents.LLMagent import LLMagent  # Import here to avoid circular imports

        self.current_agent = agent_name

        # Check if this is a reasoning model - using the list from LLMagent
        self.is_reasoning_model = agent_name in self.reasoning_models

        # Initialize the LLM agent with reasoning parameters if applicable
        self.agent = LLMagent(
            model_name=agent_name,
            use_unsloth=self.use_unsloth,
            device_map=self.device,
            reasoning_mode=self.reasoning_mode,
            min_thinking_time=self.min_thinking_time,
            max_thinking_time=self.max_thinking_time,
            min_thinking_tokens=self.min_thinking_tokens,
            max_thinking_tokens=self.max_thinking_tokens
        )

        # Set up the hook for saving activations if specified
        if not self.is_reasoning_model and self.activations_layer is not None:
            path_parts = self.activations_layer.split('.')
            layer = self.agent.model
            # Get the model component from the input string
            for part in path_parts:
                if '[' in part and ']' in part:
                    list_name, index = part.split('[')
                    index = int(index[:-1])
                    layer = getattr(layer, list_name)[index]
                else:
                    layer = getattr(layer, part)

            # Create the directory for saving activations
            path = os.path.join("./activations", self.current_agent, f"{'_'.join(path_parts)}_{self.timestamp}")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            print(f"Saving activations to {path}")

            self.hook = Hook(layer, save_path=path)

        # Get the reasoning parameters that were actually set
        self.reasoning_params = self.agent.get_reasoning_parameters()
        self.tokenizer = self.agent.tokenizer

        return self.agent

    def run_single_trial(self, n_rounds: int, num_quadrants: int, trial_num: int) -> Dict:
        """
        Run a single trial of the VST task and return results.

        Parameters:
        -----------
        n_rounds : int
            Number of rounds for this trial
        num_quadrants : int
            Number of quadrants for this trial
        trial_num : int
            Current trial number (0-indexed)

        Returns:
        --------
        dict
            Dictionary containing trial results and statistics
        """
        task = self.task_type.get_task()
        self.task = task(n_rounds, num_quadrants, self.num_cues)
        self.task.verbose = self.verbose # This is not a nice fix, kinda scrappy
        self.task.update_trial(trial_num)
        self.thinking_times = []  # Reset thinking times for this trial

        trial_stats = {
            'rounds': [],
            'final_choice': None,
            'correct_quadrant': self.task.get_correct_answer(),
            'success': False,
            'agent': self.current_agent,
            'round_times': [],
            'thinking_times': []
        }

        # Add a trial separator in the conversation history if this is not the first trial
        if trial_num > 0:
            self.conversation_history += self.task.get_trial_separator()

        # If this is the first trial or we want to start fresh, initialize conversation with task description
        if not self.conversation_history:
            task_description = self.task.get_initial_prompt()
            self.conversation_history = task_description + "\n" + self.task.get_trial_separator()

            if self.verbose:
                tqdm.write("\n=== Task Description ===")
                tqdm.write(task_description)
                tqdm.write(f"\n=== Beginning Trial {trial_num + 1} ===")
        elif self.verbose:
            tqdm.write(f"\n=== Beginning Trial {trial_num + 1} ===")

        # Run all rounds
        for round_num in range(n_rounds):
            self.task.update_round(round_num)

            if self.verbose:
                tqdm.write(f"\n--- Trial {trial_num + 1}, Round {round_num + 1} ---")

            # Build and show prompt with accumulated history
            prompt = self.task.get_intermediate_prompt()

            history_and_prompt = self.conversation_history + prompt

            if self.verbose:
                tqdm.write("\nAccumulated prompt shown to LLM:")
                tqdm.write("--------------------")
                tqdm.write(history_and_prompt)
                tqdm.write("--------------------")

            # Get agent's choice and track round time
            round_start_time = time.time()
            if hasattr(self, "hook"):
                self.hook.current_text = history_and_prompt
                tokens = self.tokenizer(history_and_prompt, return_tensors="pt")["input_ids"]
                self.hook.current_tokens = tokens.squeeze(0)  # Remove batch dimension
                self.hook.current_file_name = f"{self.current_agent}_trial{trial_num}_round{round_num}"
            choice = self.agent.get_response(history_and_prompt)
            choice = "A"
            self.task.update_answer(choice)
            round_time = time.time() - round_start_time
            self.task.round_time = round_time
            trial_stats['round_times'].append(round_time)

            # Extract thinking time if this is a reasoning model
            thinking_time = 0
            self.task.thinking_time = thinking_time # Not a nice fix
            thinking_tokens = ""
            if self.is_reasoning_model and hasattr(self.agent, 'thinking_time'):
                self.task.thinking_time = self.agent.thinking_time
                if hasattr(self.agent, 'last_thinking_tokens'):
                    thinking_tokens = self.agent.last_thinking_tokens
                self.thinking_times.append(thinking_time)
                trial_stats['thinking_times'].append(thinking_time)

            if self.verbose:
                tqdm.write(f"\nLLM chose: {choice}")
                if self.is_reasoning_model:
                    tqdm.write(f"Thinking time: {thinking_time:.2f} seconds")

            # Process choice
            result = self.task.process_choice()
            self.task.update_result(result)

            # Update conversation history with feedback
            self.conversation_history += self.task.give_feedback()

            round_stats = self.task.create_round_stats()

            # For reasoning models, add thinking tokens
            if self.is_reasoning_model and thinking_tokens:
                round_stats['thinking_tokens'] = thinking_tokens

            trial_stats['rounds'].append(round_stats)

        # Get final answer
        if self.verbose:
            tqdm.write(f"\n=== Trial {trial_num + 1} Final Decision ===")

        history_and_prompt = self.conversation_history + self.task.get_final_prompt()

        #self.conversation_history += self.task.get_final_prompt()

        if self.verbose:
            tqdm.write("\nFinal accumulated prompt shown to LLM:")
            tqdm.write("-------------------------")
            tqdm.write(history_and_prompt)
            tqdm.write("-------------------------")

        if hasattr(self, "hook"):
            self.hook.current_text = history_and_prompt
            tokens = self.tokenizer(history_and_prompt, return_tensors="pt")["input_ids"]
            self.hook.current_tokens = tokens.squeeze(0)  # Remove batch dimension
            self.hook.current_file_name = f"{self.current_agent}_trial{trial_num}_round{round_num}"

        final_choice = self.agent.get_response(history_and_prompt)
        final_choice = "A"

        self.task.update_answer(final_choice)

        self.conversation_history += self.task.give_final_feedback()

        success = self.task.process_final_choice()
        trial_stats['final_choice'] = final_choice
        if success is not None:
            trial_stats['success'] = success
            if self.verbose:
                tqdm.write(f"\n=== Trial {trial_num + 1} Results ===")
                self.task.print_final_log()

        return trial_stats

    def run_trials(self, n_rounds: int, num_quadrants: int) -> Dict:
        """
        Run multiple trials of the VST task and return results.

        Parameters:
        -----------
        n_rounds : int
            Number of rounds for each trial
        num_quadrants : int
            Number of quadrants for each trial

        Returns:
        --------
        dict
            Dictionary containing aggregated trial results and statistics
        """
        all_trials = []
        trial_times = []
        success_rates = []

        # Reset conversation history for new set of trials
        self.conversation_history = ""

        for trial_num in range(self.n_trials):
            # Run a single trial and track time
            trial_start_time = time.time()
            trial_stats = self.run_single_trial(n_rounds, num_quadrants, trial_num)
            trial_time = time.time() - trial_start_time

            # Add trial time to stats
            trial_stats['trial_time'] = trial_time
            trial_times.append(trial_time)
            success_rates.append(1 if trial_stats['success'] else 0)

            # Add to trials
            all_trials.append(trial_stats)

        # Calculate aggregate trial statistics
        return {
            'trials': all_trials,
            'avg_trial_time': float(np.mean(trial_times)),
            'std_trial_time': float(np.std(trial_times)),
            'success_rate': float(np.mean(success_rates))
        }

    def run_simulations(self, n_rounds: int, num_quadrants: int) -> Dict:
        """
        Run multiple simulations with fixed round and quadrant count.

        Parameters:
        -----------
        n_rounds : int
            Number of rounds for all simulations
        num_quadrants : int
            Number of quadrants for all simulations

        Returns:
        --------
        dict
            Dictionary containing aggregated metrics from all simulations
        """
        all_results = []
        total_time_start = time.time()

        # Create agent-specific log directory
        agent_log_dir = os.path.join(self.logs_dir, self.current_agent)
        os.makedirs(agent_log_dir, exist_ok=True)

        # Create a unique log filename based on agent, rounds, quadrants, trials
        log_filename = f"{agent_log_dir}/{self.current_agent}_{n_rounds}r_{num_quadrants}q_{self.n_trials}t_{self.timestamp}.json"

        for sim_num in tqdm(range(self.n_simulations)):
            if self.verbose:
                tqdm.write(f"\nSimulation {sim_num + 1}/{self.n_simulations}")
                tqdm.write("=" * 50)

            # Run multiple trials for this simulation
            sim_stats = self.run_trials(n_rounds, num_quadrants)
            all_results.append(sim_stats)

        # Calculate the total time for all simulations
        total_time = time.time() - total_time_start


        # Calculate metrics for this configuration
        
        # analyze_simulations only work(ed) for bias task (it throws out an error now hehe)
        
        try:
            metrics = self.analyze_simulations(all_results, n_rounds, num_quadrants, total_time)
        except Exception as e:
            print(f"analyze_simulations failed: {e}")
            metrics = {
                "agent": self.current_agent,
                "timestamp": self.timestamp,
                "n_simulations": self.n_simulations,
                "n_trials": self.n_trials,
                "n_rounds": n_rounds,
                "n_quadrants": num_quadrants,
                "n_cues": self.num_cues,
                "success_rate": 0.0,
                "total_time": total_time,
                "avg_trial_time": 0.0,
                "std_trial_time": 0.0,
                "avg_round_time": 0.0,
                "std_round_time": 0.0,
                "avg_thinking_time": None,
                "std_thinking_time": None,
                "reasoning_mode": None,
                "min_thinking_time": None,
                "max_thinking_time": None,
                "min_thinking_tokens": None,
                "max_thinking_tokens": None,
                "quadrant_distribution": {},
            }


        # Save detailed results including the raw data
        results_data = {
            'metrics': metrics,
            'raw_results': all_results
        }
        if self.log_stats:
            with open(log_filename, 'w') as f:
                json.dump(results_data, f, indent=2)
            print(f"Benchmark complete! Results saved to logs/")
        self.activation_saving()

        return metrics


    def analyze_simulations(self, results, n_rounds, num_quadrants, total_time=0) -> Dict:
        """Analyze simulation results and generate metrics."""
        # Collect all time metrics across simulations and trials
        all_trial_times = []
        all_round_times = []
        all_thinking_times = []
        all_success_rates = []
        success_rate = 0

        # Collect quadrant distribution data
        quadrant_labels = [chr(ord("A") + i) for i in range(num_quadrants)]
        quadrant_distribution = {
            label: {'times_chosen': 0, 'times_correct': 0, 'success_count': 0}
            for label in quadrant_labels
        }
        # Process all simulations
        for sim in results:
            for trial in sim['trials']:
                all_trial_times.append(trial.get('trial_time', 0))
                all_success_rates.append(1 if trial.get('success', False) else 0)

                # Process each round in the trial
                for round_data in trial.get('rounds', []):
                    all_round_times.append(round_data.get('round_time', 0))
                    thinking_time = round_data.get('thinking_time', 0)
                    if thinking_time > 0:
                        all_thinking_times.append(thinking_time)

                # Update quadrant distribution
                choice = trial.get('final_choice', '').upper()
                correct = trial.get('correct_quadrant', '')
                if int(correct):
                    correct = chr(ord("A") + int(correct) - 1)
                correct = correct.upper()

                if choice in quadrant_distribution:
                    quadrant_distribution[choice]['times_chosen'] += 1
                    if choice == correct:
                        quadrant_distribution[choice]['success_count'] += 1

                if correct in quadrant_distribution:
                    quadrant_distribution[correct]['times_correct'] += 1
                        # Calculate success rate
                    success_rate = np.mean(all_success_rates) if all_success_rates else 0

        # Calculate time metrics
        avg_trial_time = np.mean(all_trial_times) if all_trial_times else 0
        std_trial_time = np.std(all_trial_times) if all_trial_times else 0
        avg_round_time = np.mean(all_round_times) if all_round_times else 0
        std_round_time = np.std(all_round_times) if all_round_times else 0

        # Calculate thinking time metrics for reasoning models
        avg_thinking_time = np.mean(all_thinking_times) if all_thinking_times and self.is_reasoning_model else None
        std_thinking_time = np.std(all_thinking_times) if all_thinking_times and self.is_reasoning_model else None

        # Calculate accuracy for each quadrant
        for q in quadrant_distribution:
            times_chosen = quadrant_distribution[q]['times_chosen']
            if times_chosen > 0:
                quadrant_distribution[q]['accuracy_when_chosen'] = float(
                    quadrant_distribution[q]['success_count'] / times_chosen)
            else:
                quadrant_distribution[q]['accuracy_when_chosen'] = 0.0

        # Reorganized metrics according to specified order
        metrics = {
            'agent': self.current_agent,
            'timestamp': self.timestamp,
            'is_reasoning_model': self.is_reasoning_model,
            'n_simulations': int(self.n_simulations),
            'n_trials': int(self.n_trials),
            'n_rounds': int(n_rounds),
            'n_quadrants': int(num_quadrants),
            'n_cues': int(self.num_cues),
            'success_rate': float(success_rate),
            'total_time': float(total_time),
            'avg_trial_time': float(avg_trial_time),
            'std_trial_time': float(std_trial_time),
            'avg_round_time': float(avg_round_time),
            'std_round_time': float(std_round_time),
        }

        # Add reasoning model specific metrics
        if self.is_reasoning_model:
            metrics.update({
                'avg_thinking_time': float(avg_thinking_time) if avg_thinking_time is not None else None,
                'std_thinking_time': float(std_thinking_time) if std_thinking_time is not None else None,
                'reasoning_mode': self.reasoning_params['reasoning_mode'],
                'min_thinking_time': self.reasoning_params['min_thinking_time'],
                'max_thinking_time': self.reasoning_params['max_thinking_time'],
                'min_thinking_tokens': self.reasoning_params['min_thinking_tokens'],
                'max_thinking_tokens': self.reasoning_params['max_thinking_tokens']
            })
        else:
            metrics.update({
                'avg_thinking_time': None,
                'std_thinking_time': None,
                'reasoning_mode': None,
                'min_thinking_time': None,
                'max_thinking_time': None,
                'min_thinking_tokens': None,
                'max_thinking_tokens': None
            })

        # Add quadrant distribution
        metrics['quadrant_distribution'] = {f'quadrant_{q}': {
            'times_chosen': int(quadrant_distribution[q]['times_chosen']),
            'times_correct': int(quadrant_distribution[q]['times_correct']),
            'accuracy_when_chosen': float(quadrant_distribution[q]['accuracy_when_chosen'])
        } for q in quadrant_distribution}

        return metrics

    def single_benchmark(self, agent_name):
        """
        Run benchmarks for a single agent over all round and quadrant configurations.

        Parameters:
        -----------
        agent_name : str
            Name of the agent model to benchmark

        Returns:
        --------
        dict
            Dictionary containing benchmark results for this agent
        """
        # Initialize the LLM agent
        self.initialize_agent(agent_name)

        # Dictionary to store results for this agent
        agent_results = {}

        for n_rounds in self.rounds:
            rounds_label = f"{n_rounds} rounds"
            agent_results[rounds_label] = {}

            for num_quadrants in self.quadrants:
                quadrant_label = f"{num_quadrants} quadrant"

                if self.verbose:
                    print(
                        f"Running benchmark for {agent_name}, {n_rounds} rounds, {num_quadrants} quadrants, {self.n_trials} trials")

                # Run simulations for this configuration
                metrics = self.run_simulations(n_rounds, num_quadrants)

                # Still have to fix metrics
                
                # Store metrics directly
                agent_results[rounds_label][quadrant_label] = {
                    "success_rate": metrics.get('success_rate', 0),
                    "trial_time": metrics.get('avg_trial_time', 0),
                    "round_time": metrics.get('avg_round_time', 0),
                    "thinking_time": metrics.get('avg_thinking_time', 0) if self.is_reasoning_model else 0
                }

        return agent_results

    def multiple_benchmarks(self):
        """
        Run benchmarks for all agents and store the results.

        Returns:
        --------
        dict
            Dictionary containing benchmark results for all agents
        """
        start_time = time.time()

        for agent_name in self.agents:
            print(f"Running benchmark for agent: {agent_name}")
            self.results[agent_name] = self.single_benchmark(agent_name)

        elapsed_time = time.time() - start_time
        print(f"Total elapsed time: {elapsed_time:.2f} seconds")

        return self.results

    def results_to_dataframe(self):
        """
        Transform the results dictionary into a pandas DataFrame.

        Returns:
        --------
        pandas.DataFrame
            DataFrame containing benchmark results
        """
        rows = []

        # Collect all time metrics across all agents/configs
        all_agent_trial_times = {}
        all_agent_round_times = {}
        all_agent_thinking_times = {}

        # First pass: collect all time metrics for each agent
        for model, rounds_dict in self.results.items():
            all_agent_trial_times[model] = []
            all_agent_round_times[model] = []
            all_agent_thinking_times[model] = []

            # For each config, collect the time metrics
            for agent_dir in os.listdir(self.logs_dir):
                if agent_dir != model:
                    continue

                agent_path = os.path.join(self.logs_dir, agent_dir)
                for filename in os.listdir(agent_path):
                    if not filename.endswith('.json'):
                        continue

                    file_path = os.path.join(agent_path, filename)
                    try:
                        with open(file_path, 'r') as f:
                            data = json.load(f)

                            # Extract time metrics from all simulations and trials
                            for sim in data.get('raw_results', []):
                                for trial in sim.get('trials', []):
                                    # Add trial time
                                    trial_time = trial.get('trial_time', 0)
                                    if trial_time > 0:
                                        all_agent_trial_times[model].append(trial_time)

                                    # Process each round in the trial
                                    for round_data in trial.get('rounds', []):
                                        round_time = round_data.get('round_time', 0)
                                        if round_time > 0:
                                            all_agent_round_times[model].append(round_time)

                                        # If reasoning model, collect thinking times
                                        thinking_time = round_data.get('thinking_time', 0)
                                        if thinking_time > 0:
                                            all_agent_thinking_times[model].append(thinking_time)
                    except Exception as e:
                        if self.verbose:
                            print(f"Error reading {file_path}: {e}")

        # Second pass: create the dataframe rows
        for model, rounds_dict in self.results.items():
            for rounds_label, quadrants_dict in rounds_dict.items():
                for quadrant_label, metrics in quadrants_dict.items():
                    # Get metrics directly
                    performance = metrics.get("success_rate", 0)

                    # Use collected time metrics for calculations
                    trial_time_mean = np.mean(all_agent_trial_times[model]) if all_agent_trial_times[model] else 0
                    trial_time_std = np.std(all_agent_trial_times[model]) if all_agent_trial_times[model] else 0

                    round_time_mean = np.mean(all_agent_round_times[model]) if all_agent_round_times[model] else 0
                    round_time_std = np.std(all_agent_round_times[model]) if all_agent_round_times[model] else 0

                    thinking_time_mean = np.mean(all_agent_thinking_times[model]) if all_agent_thinking_times[
                        model] else 0
                    thinking_time_std = np.std(all_agent_thinking_times[model]) if all_agent_thinking_times[
                        model] else 0

                    row = {
                        "Model": model,
                        "Rounds": rounds_label,
                        "Quadrants": quadrant_label,
                        "Trials": self.n_trials,
                        "Performance": performance,
                        "Trial_Time": trial_time_mean,
                        "Trial_Time_Std": trial_time_std,
                        "Round_Time": round_time_mean,
                        "Round_Time_Std": round_time_std,
                        "Thinking_Time": thinking_time_mean,
                        "Thinking_Time_Std": thinking_time_std,
                        "raw_metrics": metrics  # Store raw metrics for later
                    }
                    rows.append(row)

        df = pd.DataFrame(rows)
        return df

    def save_results(self):
        """
        Return the DataFrame containing benchmark results.

        Returns:
        --------
        pandas.DataFrame
            DataFrame containing the results
        """
        # Just return the DataFrame for analysis
        df = self.results_to_dataframe()
        return df

    def plot_results(self, output_path=None):
        """
        Create a horizontal bar chart of aggregated benchmark results per model.

        Parameters:
        -----------
        output_path : str, optional
            Path to save the plot image, defaults to "benchmark_plot_TIMESTAMP.png"

        Returns:
        --------
        pandas.DataFrame
            DataFrame containing the results that were plotted
        """
        # Convert stored results to a DataFrame
        df = self.results_to_dataframe()


        # Aggregate the success rates per model
        aggregated_data = {}

        for model, group in df.groupby("Model"):
            # Extract success_rate directly from raw_metrics
            all_success_rates = [metrics.get("success_rate", 0) for metrics in group["raw_metrics"]]
            aggregated_data[model] = {"mean": np.mean(all_success_rates), "std": np.std(all_success_rates)}

        # Create arrays for models, means, and stds
        model_names = list(aggregated_data.keys())
        model_means = np.array([aggregated_data[model]["mean"] for model in model_names])
        model_stds = np.array([aggregated_data[model]["std"] for model in model_names])

        # Sort the data from largest mean to smallest mean
        sorted_indices = np.argsort(model_means)[::-1]  # descending order
        sorted_means = model_means[sorted_indices]
        sorted_stds = model_stds[sorted_indices]
        sorted_names = [model_names[i] for i in sorted_indices]

        # Categorize each model for coloring
        model_category = {
            "Deepseek_R1_1.5B_Qwen": "Open Source (<= 1.5B)",
            "Deepseek_R1_7B_Qwen": "Open Source (> 1.5B)",
            "Deepseek_R1_8B_Llama": "Open Source (> 1.5B)",
            "Deepseek_R1_14B_Qwen": "Open Source (> 1.5B)",
            "Deepseek_R1_32B_Qwen": "Open Source (> 1.5B)",
            "Qwen_0.5B": "Open Source (<= 1.5B)",
            "Qwen_1.5B": "Open Source (<= 1.5B)",
            "Qwen_3B": "Open Source (> 1.5B)",
            "Qwen_7B": "Open Source (> 1.5B)",
            "Qwen_14B": "Open Source (> 1.5B)",
            "Qwen_32B": "Open Source (> 1.5B)",
            "Qwen_0.5B_Instruct": "Open Source (<= 1.5B)",
            "Qwen_1.5B_Instruct": "Open Source (<= 1.5B)",
            "Qwen_3B_Instruct": "Open Source (> 1.5B)",
            "Qwen_7B_Instruct": "Open Source (> 1.5B)",
            "Qwen_14B_Instruct": "Open Source (> 1.5B)",
            "Qwen_32B_Instruct": "Open Source (> 1.5B)",
            "Centaur_8B": "Open Source (> 1.5B)",
            "Mistral_7B_Instruct": "Open Source (> 1.5B)",
            "Mistral_7B": "Open Source (> 1.5B)",
            "Phi_4_8B": "Open Source (> 1.5B)",
            "Phi_3.5_mini_Instruct": "Open Source (> 1.5B)",
            "Phi_3_mini_Instruct": "Open Source (> 1.5B)",
            "Phi_3.5_medium_Instruct": "Open Source (> 1.5B)",
            "Gemma_2B": "Open Source (> 1.5B)",
            "Gemma_9B": "Open Source (> 1.5B)",
            "Gemma_27B": "Open Source (> 1.5B)",
            "Gemma_2B_Instruct": "Open Source (> 1.5B)",
            "Gemma_9B_Instruct": "Open Source (> 1.5B)",
            "Gemma_27B_Instruct": "Open Source (> 1.5B)",
            "gpt4o": "API",
            "gpt4o-mini": "API",
            "o1-mini": "API",
            "sonnet": "API",
            "haiku": "API"
        }

        category_colors = {
            "API": "royalblue",
            "Open Source (> 1.5B)": "darkred",
            "Open Source (<= 1.5B)": "darkgreen"
        }

        # Build list of colors based on category
        bar_colors = [category_colors.get(model_category.get(name, "API"), "gray") for name in sorted_names]

        # Prepare asymmetric error bars
        lower_errors = np.where(sorted_means - sorted_stds < 0, sorted_means, sorted_stds)
        upper_errors = sorted_stds  # full std on the right

        # Plot the horizontal bar chart
        plt.figure(figsize=(7, 5))
        y_positions = np.arange(len(sorted_means))  # positions for each bar

        plt.barh(
            y_positions,
            sorted_means,
            height=0.5,  # slimmer bars
            xerr=[lower_errors, upper_errors],
            color=bar_colors,
            edgecolor='black',
            capsize=0
        )

        plt.yticks(ticks=y_positions, labels=sorted_names)
        plt.gca().invert_yaxis()  # best model on top
        plt.subplots_adjust(left=0.32)
        plt.xlabel("Aggregated Mean Success Rate")

        # Include parameters in title for clarity
        title_params = f"r={self.rounds[0] if len(self.rounds) == 1 else self.rounds}, " \
                       f"q={self.quadrants[0] if len(self.quadrants) == 1 else self.quadrants}, " \
                       f"t={self.n_trials}"
        plt.title(f"G1Bbon benchmark ({title_params})")

        # Create a legend for the categories
        legend_handles = [
            Patch(facecolor='royalblue', label='API', edgecolor='black'),
            Patch(facecolor='darkred', label='Open Source (> 1.5B)', edgecolor='black'),
            Patch(facecolor='darkgreen', label='Open Source (<= 1.5B)', edgecolor='black'),
        ]
        plt.legend(handles=legend_handles, loc="lower right")

        plt.tight_layout()

        # Save the plot in the benchmarks_plots directory
        if output_path is None:
            config_str = f"_r{self.rounds[0]}_q{self.quadrants[0]}_t{self.n_trials}"
            output_path = f"{self.benchmarks_plots_dir}/benchmark_plot{config_str}_{self.timestamp}.png"
        else:
            output_path = f"{self.benchmarks_plots_dir}/{output_path}"

        plt.savefig(output_path, dpi=300)
        print(f"Benchmark plot saved to {output_path}")

        # Set a flag to indicate plot has been generated
        self.plot_generated = True

        return df
    
    def activation_saving(self):
        """Save all collected activations and clean up the hook."""
        if hasattr(self, "hook") and self.hook is not None:
            print(f"Saving {len(self.hook.activations)} activations...")
            self.hook.save_all()
            self.hook.remove()
            self.hook = None  # Clean up reference
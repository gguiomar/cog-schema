import os
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any, Union
from tqdm import tqdm

from tasks.VSTtask import VSTtask

class TaskManager:
    def __init__(self, agents=None, rounds=None, quadrants=None, n_simulations=10, 
                 n_runs=5, num_cues=1, device="cuda:0", verbose=False,
                 output_dir="simulation_results", openai_api_key=None, 
                 anthropic_api_key=None, use_unsloth=True):
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
        n_runs : int
            Number of runs for each configuration
        num_cues : int
            Number of cues per quadrant
        device : str
            Device to use for model inference
        verbose : bool
            Whether to print detailed output
        output_dir : str
            Directory to save results
        openai_api_key : str
            OpenAI API key (if applicable)
        anthropic_api_key : str
            Anthropic API key (if applicable)
        use_unsloth : bool
            Whether to use Unsloth optimization
        """
        from agents.LLMagent import LLMagent  # Import here to avoid circular imports
        
        # Convert single values to lists for consistent processing
        self.agents = [agents] if isinstance(agents, str) else agents
        self.rounds = [rounds] if isinstance(rounds, int) else rounds
        self.quadrants = [quadrants] if isinstance(quadrants, int) else quadrants
        
        self.n_simulations = n_simulations
        self.n_runs = n_runs
        self.num_cues = num_cues
        self.device = device
        self.verbose = verbose
        self.output_dir = output_dir
        self.openai_api_key = openai_api_key
        self.anthropic_api_key = anthropic_api_key
        self.use_unsloth = use_unsloth
        
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.conversation_history = ""
        self.current_agent = None
        self.is_reasoning_model = False
        self.thinking_times = []
        
        # Create results directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize results storage
        self.results = {}
        
    def initialize_agent(self, agent_name):
        """Initialize an LLM agent with the specified model."""
        from agents.LLMagent import LLMagent  # Import here to avoid circular imports
        
        self.current_agent = agent_name
        
        # Initialize the LLM agent
        self.agent = LLMagent(
            model_name=agent_name,
            use_unsloth=self.use_unsloth,
            device_map=self.device,
            openai_api_key=self.openai_api_key,
            anthropic_api_key=self.anthropic_api_key
        )
        
        # Check if this is a reasoning model
        # (Based on LLMagent code, we know reasoning models are specific ones)
        reasoning_models = ["Deepseek_R1_1B_Qwen", "Deepseek_R1_7B_Qwen", "Deepseek_R1_8B_Llama"]
        self.is_reasoning_model = agent_name in reasoning_models
        
        return self.agent
    
    def build_prompt(self, available_cues: str, round_num: int) -> str:
        """Build prompt including conversation history."""
        # Add the current round information
        current_prompt = (
            f"Round {round_num + 1}: Available cues {available_cues}. "
            f"Based on previous observations, choose one cue by responding with just the letter. You press <<"
        )
        
        return self.conversation_history + current_prompt
        
    def update_history(self, cues: str, choice: str, result: Optional[str], round_num: int) -> None:
        """Update conversation history with round results."""
        result_text = result if result is not None else "Invalid choice"
        round_text = (
            f"Round {round_num + 1}: Available cues {cues}. "
            f"You chose {choice} and saw {result_text}.\n"
        )
        self.conversation_history += round_text
        
    def get_final_prompt(self, num_quadrants) -> str:
        """Build final prompt including full conversation history."""
        prompt = (
            self.conversation_history +
            "Based on all observed colors, which quadrant (1"
            f"{', ' + ', '.join(str(i) for i in range(2, num_quadrants + 1))}"
            ") do you think had the highest ratio of RED? "
            "Respond with just the number. You choose <<"
        )
        return prompt
    
    def run_single_task(self, n_rounds: int, num_quadrants: int) -> Dict:
        """
        Run a single VST task and return results.
        
        Parameters:
        -----------
        n_rounds : int
            Number of rounds for this task
        num_quadrants : int
            Number of quadrants for this task
        
        Returns:
        --------
        dict
            Dictionary containing task results and statistics
        """
        self.task = VSTtask(n_rounds, num_quadrants, self.num_cues)
        self.conversation_history = ""
        self.agent.reset_history()
        self.thinking_times = []  # Reset thinking times for this task

        stats = {
            'rounds': [],
            'final_choice': None,
            'correct_quadrant': self.task.biased_quadrant + 1,
            'success': False,
            'full_conversation': [],
            'agent': self.current_agent,
            'thinking_times': [],
            'avg_thinking_time': 0
        }

        # Initialize conversation with task description
        task_description = self.task.get_task_description()
        self.conversation_history = task_description + "\n\n"

        if self.verbose:
            print("\n=== Task Description ===")
            print(task_description)
            print("\n=== Beginning Rounds ===")

        stats['full_conversation'].append(("INITIAL_DESCRIPTION", task_description))

        # Run all rounds
        total_time_start = time.time()
        for round_num in range(self.task.n_rounds):
            if self.verbose:
                print(f"\n--- Round {round_num + 1} ---")

            round_data = self.task.get_round_data(round_num)
            available_cues = [q['name'] for q in round_data]

            # Build and show prompt with accumulated history
            prompt = self.build_prompt(', '.join(available_cues), round_num)

            if self.verbose:
                print("\nAccumulated prompt shown to LLM:")
                print("--------------------")
                print(prompt)
                print("--------------------")

            # Get agent's choice
            round_start_time = time.time()
            choice = self.agent.get_response(prompt)
            round_time = time.time() - round_start_time
            
            # Extract thinking time if this is a reasoning model
            thinking_time = 0
            if self.is_reasoning_model and hasattr(self.agent, 'thinking_time'):
                thinking_time = self.agent.thinking_time
                self.thinking_times.append(thinking_time)
            
            if self.verbose:
                print(f"\nLLM chose: {choice}")
                if self.is_reasoning_model:
                    print(f"Thinking time: {thinking_time:.2f} seconds")

            # Process choice
            result = self.task.process_choice(choice, round_data)
            quadrant = None
            
            if self.verbose:
                if result:
                    print(f"Result: {result}")
                    # Show quadrant info in verbose mode
                    for q in round_data:
                        if q['name'] == choice:
                            quadrant = q['quadrant'] + 1
                            print(f"(cue {choice} was from Quadrant {quadrant})")
                else:
                    print("Invalid choice!")

            # Update conversation history
            self.update_history(', '.join(available_cues), choice, result, round_num)
                    
            stats['rounds'].append({
                'available_cues': available_cues,
                'choice': choice,
                'result': result,
                'full_prompt': prompt,
                'round_time': round_time,
                'thinking_time': thinking_time,
                'quadrant': quadrant
            })
            
            stats['thinking_times'].append(thinking_time)

            # Record the conversation details
            stats['full_conversation'].extend([
                ("ACCUMULATED_PROMPT", prompt),
                ("LLM_CHOICE", choice),
                ("RESULT", result if result is not None else "Invalid choice")
            ])

        # Get final answer
        if self.verbose:
            print("\n=== Final Decision ===")

        final_prompt = self.get_final_prompt(num_quadrants)

        if self.verbose:
            print("\nFinal accumulated prompt shown to LLM:")
            print("-------------------------")
            print(final_prompt)
            print("-------------------------")

        final_choice = self.agent.get_response(final_prompt)
        total_time = time.time() - total_time_start
        time_per_round = total_time / n_rounds

        if self.verbose:
            print(f"\nLLM's final choice: Quadrant {final_choice}")
            print("\n=== Game Results ===")
            print(f"Correct quadrant: {self.task.biased_quadrant + 1}")
            print(f"LLM chose: {final_choice}")
            print(f"Success: {str(self.task.biased_quadrant + 1) == final_choice}")

        stats['final_choice'] = final_choice
        stats['success'] = str(self.task.biased_quadrant + 1) == final_choice
        stats['total_time'] = total_time
        stats['time_per_round'] = time_per_round
        stats['avg_thinking_time'] = sum(stats['thinking_times']) / len(stats['thinking_times']) if stats['thinking_times'] else 0

        stats['full_conversation'].extend([
            ("FINAL_ACCUMULATED_PROMPT", final_prompt),
            ("FINAL_CHOICE", final_choice)
        ])

        return stats
    
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
        rounds_label = f"{n_rounds} rounds"
        quadrant_label = f"{num_quadrants} quadrant"
        
        # Create a unique log filename based on agent, rounds, quadrants
        log_filename = f"{self.output_dir}/{self.current_agent}_{n_rounds}r_{num_quadrants}q_{self.timestamp}.json"
        
        for sim_num in tqdm(range(self.n_simulations), disable=self.verbose):
            if self.verbose:
                print(f"\nSimulation {sim_num + 1}/{self.n_simulations}")
                print("=" * 50)
            
            stats = self.run_single_task(n_rounds, num_quadrants)
            all_results.append(stats)
        
        # Calculate metrics for this configuration
        metrics = self.analyze_simulations(all_results, n_rounds, num_quadrants)
        
        # Save detailed results including the raw data
        results_data = {
            'metrics': metrics,
            'raw_results': all_results,
            'config': {
                'agent': self.current_agent,
                'rounds': n_rounds,
                'quadrants': num_quadrants,
                'n_simulations': self.n_simulations,
                'timestamp': self.timestamp
            }
        }
        
        with open(log_filename, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        return metrics
    
    def analyze_simulations(self, results, n_rounds, num_quadrants) -> Dict:
        """Analyze simulation results and generate metrics."""
        metrics = {
            'agent': self.current_agent,
            'is_reasoning_model': self.is_reasoning_model,
            'n_rounds': int(n_rounds),
            'n_quadrants': int(num_quadrants),
            'n_cues': int(self.num_cues),
            'n_simulations': int(self.n_simulations),
            'success_rate': float(np.mean([r['success'] for r in results])),
            'avg_time_per_round': float(np.mean([r['time_per_round'] for r in results])),
            'std_time_per_round': float(np.std([r['time_per_round'] for r in results])),
            'quadrant_distribution': {},
            'timestamp': self.timestamp
        }
        
        # Add thinking time metrics for reasoning models
        if self.is_reasoning_model:
            all_thinking_times = [time for r in results for time in r['thinking_times'] if time > 0]
            metrics['avg_thinking_time'] = float(np.mean(all_thinking_times)) if all_thinking_times else 0
            metrics['std_thinking_time'] = float(np.std(all_thinking_times)) if all_thinking_times else 0
            
        # Calculate quadrant selection frequencies
        for q in range(1, num_quadrants + 1):
            times_chosen = len([r for r in results if r['final_choice'] == str(q)])
            times_correct = len([r for r in results if int(r['correct_quadrant']) == q])
            
            metrics['quadrant_distribution'][f'quadrant_{q}'] = {
                'times_chosen': int(times_chosen),
                'times_correct': int(times_correct),
                'accuracy_when_chosen': float(sum([1 for r in results 
                                               if r['final_choice'] == str(q) and r['success']]) / times_chosen) 
                                               if times_chosen > 0 else 0
            }
        
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
                # Initialize an empty list for the runs
                agent_results[rounds_label][quadrant_label] = []
                
                for run in range(self.n_runs):
                    if self.verbose:
                        print(f"Running benchmark for {agent_name}, {n_rounds} rounds, {num_quadrants} quadrants, run {run+1}/{self.n_runs}")
                    
                    # Run simulations for this configuration
                    metrics = self.run_simulations(n_rounds, num_quadrants)
                    
                    # Store metrics for this run
                    run_metrics = {
                        "success_rate": metrics.get('success_rate', 0),
                        "time_per_round": metrics.get('avg_time_per_round', 0),
                        "thinking_time": metrics.get('avg_thinking_time', 0) if self.is_reasoning_model else 0
                    }
                    agent_results[rounds_label][quadrant_label].append(run_metrics)
        
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
            
            # Save intermediate results after each agent
            self.save_results(f"bench_{agent_name}_{self.timestamp}.json")
        
        elapsed_time = time.time() - start_time
        print(f"Total elapsed time: {elapsed_time:.2f} seconds")
        
        # Save the complete results
        self.save_results(f"bench_all_{self.timestamp}.json")
        
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
        
        for model, rounds_dict in self.results.items():
            for rounds_label, quadrants_dict in rounds_dict.items():
                for quadrant_label, runs in quadrants_dict.items():
                    # Extract metrics from the run dictionaries
                    success_rates = [r["success_rate"] for r in runs]
                    time_per_rounds = [r["time_per_round"] for r in runs]
                    thinking_times = [r.get("thinking_time", 0) for r in runs]
                    
                    # Calculate averages and standard deviations
                    performance = np.mean(success_rates)
                    std = np.std(success_rates)
                    time_mean = np.mean(time_per_rounds)
                    time_std = np.std(time_per_rounds)
                    thinking_mean = np.mean(thinking_times)
                    thinking_std = np.std(thinking_times)
                    
                    row = {
                        "Model": model,
                        "Rounds": rounds_label,
                        "Quadrants": quadrant_label,
                        "Performance": performance,
                        "Std": std,
                        "Time_Per_Round": time_mean,
                        "Time_Std": time_std,
                        "Thinking_Time": thinking_mean,
                        "Thinking_Time_Std": thinking_std,
                        "raw": runs  # store raw run values for later aggregation
                    }
                    rows.append(row)
        
        df = pd.DataFrame(rows)
        return df
    
    def save_results(self, filepath=None):
        """
        Save benchmark results to a JSON file.
        
        Parameters:
        -----------
        filepath : str, optional
            Path to save the results file, defaults to "bench_TIMESTAMP.json"
        """
        if filepath is None:
            filepath = f"{self.output_dir}/bench_{self.timestamp}.json"
        else:
            filepath = f"{self.output_dir}/{filepath}"
        
        # Convert results to a serializable format
        serializable_results = {}
        
        for model, rounds_dict in self.results.items():
            serializable_results[model] = {}
            for rounds_label, quadrants_dict in rounds_dict.items():
                serializable_results[model][rounds_label] = {}
                for quadrant_label, runs in quadrants_dict.items():
                    # Ensure all values are Python native types for JSON serialization
                    serializable_runs = []
                    for run in runs:
                        serializable_run = {
                            "success_rate": float(run["success_rate"]),
                            "time_per_round": float(run["time_per_round"])
                        }
                        if "thinking_time" in run:
                            serializable_run["thinking_time"] = float(run["thinking_time"])
                        serializable_runs.append(serializable_run)
                    
                    serializable_results[model][rounds_label][quadrant_label] = serializable_runs
        
        # Add metadata
        benchmark_data = {
            "results": serializable_results,
            "metadata": {
                "timestamp": self.timestamp,
                "agents": self.agents,
                "rounds": self.rounds,
                "quadrants": self.quadrants,
                "n_simulations": self.n_simulations,
                "n_runs": self.n_runs,
                "num_cues": self.num_cues,
                "device": self.device,
                "use_unsloth": self.use_unsloth
            }
        }
        
        with open(filepath, "w") as f:
            json.dump(benchmark_data, f, indent=2)
        
        print(f"Results saved to {filepath}")
        
        # Also save as DataFrame for easier analysis
        df = self.results_to_dataframe()
        df_path = filepath.replace(".json", ".csv")
        df.drop(columns=["raw"]).to_csv(df_path, index=False)
        
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
        
        # Aggregate the raw success_rate values per model
        aggregated_data = {}
        for model, group in df.groupby("Model"):
            # Extract success_rate from each run
            all_success_rates = [run["success_rate"] for runs in group["raw"].values for run in runs]
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
            "Deepseek_R1_1B_Qwen":     "Open Source (<= 1.5B)",
            "Deepseek_R1_7B_Qwen":     "Open Source (> 1.5B)",
            "Deepseek_R1_8B_Llama":    "Open Source (> 1.5B)",
            "Qwen_0.5B":               "Open Source (<= 1.5B)",
            "Qwen_1.5B":               "Open Source (<= 1.5B)",
            "Qwen_3B":                 "Open Source (> 1.5B)",
            "Qwen_7B":                 "Open Source (> 1.5B)",
            "Qwen_0.5B_Instruct":      "Open Source (<= 1.5B)",
            "Qwen_1.5B_Instruct":      "Open Source (<= 1.5B)",
            "Qwen_3B_Instruct":        "Open Source (> 1.5B)",
            "Qwen_7B_Instruct":        "Open Source (> 1.5B)",
            "Centaur_8B":              "Open Source (> 1.5B)",
            "Mistral_7B_Instruct":     "Open Source (> 1.5B)",
            "Mistral_7B":              "Open Source (> 1.5B)",
            "Phi_mini_2B_Instruct":    "Open Source (> 1.5B)",
            "Gemma_2B_Instruct":       "Open Source (> 1.5B)",
            "Gemma_2B":                "Open Source (> 1.5B)",
            "gpt4o":                   "API",
            "gpt4o-mini":              "API",
            "o1-mini":                 "API",
            "sonnet":                  "API",
            "haiku":                   "API"
        }
        
        category_colors = {
            "API":                     "royalblue",
            "Open Source (> 1.5B)":    "darkred",
            "Open Source (<= 1.5B)":   "darkgreen"
        }
        
        # Build list of colors based on category
        bar_colors = [category_colors.get(model_category.get(name, "API"), "gray") for name in sorted_names]
        
        # Prepare asymmetric error bars
        lower_errors = np.where(sorted_means - sorted_stds < 0, sorted_means, sorted_stds)
        upper_errors = sorted_stds  # full std on the right
        
        # Plot the horizontal bar chart
        plt.figure(figsize=(6, 5))
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
        plt.xlabel("Aggregated Mean Score")
        plt.title("G1Bbon benchmark")
        
        # Create a legend for the categories
        legend_handles = [
            Patch(facecolor='royalblue',  label='API',                      edgecolor='black'),
            Patch(facecolor='darkred',    label='Open Source (> 1.5B)',     edgecolor='black'),
            Patch(facecolor='darkgreen',  label='Open Source (<= 1.5B)',    edgecolor='black'),
        ]
        plt.legend(handles=legend_handles, loc="lower right")
        
        plt.tight_layout()
        
        # Save the plot if requested
        if output_path is None:
            output_path = f"{self.output_dir}/benchmark_plot_{self.timestamp}.png"
        else:
            output_path = f"{self.output_dir}/{output_path}"
            
        plt.savefig(output_path, dpi=300)
        plt.show()
        
        return df

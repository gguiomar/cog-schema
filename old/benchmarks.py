import time
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from agents.LLMagent import LLMagent
from manager.TaskManager import TaskManager

class BenchmarkRunner():
    def __init__(self, agents, rounds, quadrants, n_simulations=10, n_runs=5, 
                 num_cues=1, device="cuda:0", verbose=False,
                 openai_api_key=None, anthropic_api_key=None, use_unsloth=True):
        """
        Initialize the BenchmarkRunner.
        """
        self.agents = agents
        self.rounds = rounds
        self.quadrants = quadrants
        self.n_simulations = n_simulations
        self.n_runs = n_runs
        self.num_cues = num_cues
        self.device = device
        self.verbose = verbose
        self.openai_api_key = openai_api_key
        self.anthropic_api_key = anthropic_api_key
        self.use_unsloth = use_unsloth
        
        # Use a dictionary to store results.
        self.results = {}

    def single_benchmark(self, agent_name):
        """
        Run benchmarks for a single agent over all round and quadrant configurations
        and return the results as a nested dictionary.
        """
        # Initialize the LLM agent.
        pipe = LLMagent(
            model_name=agent_name,
            use_unsloth=self.use_unsloth,
            device_map=self.device,
            openai_api_key=self.openai_api_key,
            anthropic_api_key=self.anthropic_api_key
        )
        
        # Dictionary to store results for this agent.
        agent_results = {}
        
        for n_rounds in self.rounds:
            rounds_label = f"{n_rounds} rounds"
            agent_results[rounds_label] = {}
            for num_quadrants in self.quadrants:
                quadrant_label = f"{num_quadrants} quadrant"
                # Initialize an empty list for the runs.
                agent_results[rounds_label][quadrant_label] = []
                for run in range(self.n_runs):
                    # Create a new TaskManager instance for each run.
                    manager = TaskManager(
                        n_simulations=self.n_simulations,
                        n_rounds=n_rounds,
                        num_quadrants=num_quadrants,
                        num_cues=self.num_cues,
                        pipe=pipe,
                        verbose=self.verbose
                    )
                    # Start timer before running simulations.
                    start_run = time.time()
                    metrics = manager.run_simulations()
                    elapsed = time.time() - start_run
                    # Compute time per round
                    time_per_round = elapsed / (n_rounds)
                    # Store both success_rate and time_per_round as a dictionary.
                    run_metrics = {
                        "success_rate": metrics.get('success_rate', 0),
                        "time_per_round": time_per_round
                    }
                    agent_results[rounds_label][quadrant_label].append(run_metrics)
        return agent_results

    def multiple_benchmarks(self):
        """
        Run benchmarks for all agents and store the results in a dictionary.
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
        Each row contains Model, Rounds, Quadrants, Performance (mean success rate over runs),
        Std (std of success rates), Time_Per_Round (mean time per round), Time_Std (std of time per round),
        and also stores the raw run dictionaries.
        """
        rows = []
        for model, rounds_dict in self.results.items():
            for rounds_label, quadrants_dict in rounds_dict.items():
                for quadrant_label, runs in quadrants_dict.items():
                    # Extract success rates and time metrics from the run dictionaries.
                    success_rates = [r["success_rate"] for r in runs]
                    time_per_rounds = [r["time_per_round"] for r in runs]
                    performance = np.mean(success_rates)
                    std = np.std(success_rates)
                    time_mean = np.mean(time_per_rounds)
                    time_std = np.std(time_per_rounds)
                    row = {
                        "Model": model,
                        "Rounds": rounds_label,
                        "Quadrants": quadrant_label,
                        "Performance": performance,
                        "Std": std,
                        "Time_Per_Round": time_mean,
                        "Time_Std": time_std,
                        "raw": runs  # store raw run values for later aggregation
                    }
                    rows.append(row)
        df = pd.DataFrame(rows)
        return df

    def save_results(self, filepath="bench.pkl"):
        """
        Convert the results dictionary to a DataFrame and save it to a pickle file.
        """
        df = self.results_to_dataframe()
        with open(filepath, "wb") as f:
            pickle.dump(df, f)
        print(f"Results saved to {filepath}")

    def plot_results(self):
        """
        Create a horizontal bar chart of aggregated benchmark results per model.
        Aggregates the raw run values for each model across rounds and quadrants.
        """
        # Convert stored results to a DataFrame.
        df = self.results_to_dataframe()
        
        # Aggregate the raw success_rate values per model.
        aggregated_data = {}
        for model, group in df.groupby("Model"):
            # Extract success_rate from each run in group["raw"]
            all_success_rates = [run["success_rate"] for runs in group["raw"].values for run in runs]
            aggregated_data[model] = {"mean": np.mean(all_success_rates), "std": np.std(all_success_rates)}
        
        # Create arrays for models, means, and stds.
        model_names = list(aggregated_data.keys())
        model_means = np.array([aggregated_data[model]["mean"] for model in model_names])
        model_stds  = np.array([aggregated_data[model]["std"] for model in model_names])
        
        # ----------------------------------------------------
        # 1) Sort the data from largest mean to smallest mean
        # ----------------------------------------------------
        sorted_indices = np.argsort(model_means)[::-1]  # descending order
        sorted_means = model_means[sorted_indices]
        sorted_stds  = model_stds[sorted_indices]
        sorted_names = [model_names[i] for i in sorted_indices]
        
        # ----------------------------------------------------
        # 2) Categorize each model so we can color them
        # ----------------------------------------------------
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
            "gpt4o":                   "API",
            "gpt4o_mini":              "API",
            "Mistral_7B_Instruct":     "Open Source (> 1.5B)",
            "Mistral_7B":              "Open Source (> 1.5B)",
            "Phi_mini_2B_Instruct":    "Open Source (> 1.5B)",
            "Gemma_2B_Instruct":       "Open Source (> 1.5B)",
            "Gemma_2B":                "Open Source (> 1.5B)",

        }
        
        category_colors = {
            "API":                   "royalblue",
            "Open Source (> 1.5B)":    "darkred",
            "Open Source (<= 1.5B)":   "darkgreen"
        }
        
        # Build list of colors based on category.
        bar_colors = [category_colors.get(model_category.get(name, "API"), "gray")
                      for name in sorted_names]
        
        # ----------------------------------------------------
        # 3) Prepare asymmetric error bars so that the left error
        #    does not push below 0.
        # ----------------------------------------------------
        lower_errors = np.where(sorted_means - sorted_stds < 0, sorted_means, sorted_stds)
        upper_errors = sorted_stds  # full std on the right
        
        # ----------------------------------------------------
        # 4) Plot the horizontal bar chart with slimmer bars (height=0.5)
        #    and the asymmetric error bars.
        # ----------------------------------------------------
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
        
        # ----------------------------------------------------
        # 5) Create a legend for the categories.
        # ----------------------------------------------------
        legend_handles = [
            Patch(facecolor='royalblue',  label='API',                   edgecolor='black'),
            Patch(facecolor='darkred',     label='Open Source (> 1.5B)',    edgecolor='black'),
            Patch(facecolor='darkgreen',   label='Open Source (<= 1.5B)',   edgecolor='black'),
        ]
        plt.legend(handles=legend_handles, loc="lower right")
        
        plt.tight_layout()
        plt.show()

        return df

# Example usage:
if __name__ == "__main__":
    agents = [
        "Deepseek_R1_1B_Qwen", "Deepseek_R1_7B_Qwen", "Deepseek_R1_8B_Llama", 
        "Qwen_1B", "Qwen_3B", "Qwen_7B", "Qwen_1B_Instruct", "Qwen_3B_Instruct", 
        "Qwen_7B_Instruct", "Centaur_8B", "gpt4o", "gpt40-mini"
    ]
    
    rounds = [i for i in range(2, 16)]  # or use any specific rounds you want
    quadrants = [2, 3, 4]

    # Optionally set your API keys.
    openai_api_key = None  
    anthropic_api_key = None  

    # Create an instance of the BenchmarkRunner.
    benchmark = BenchmarkRunner(
        agents=agents,
        rounds=rounds,
        quadrants=quadrants,
        n_simulations=10,
        n_runs=5,
        num_cues=1,
        device="cuda:0",
        verbose=False,
        openai_api_key=openai_api_key,
        anthropic_api_key=anthropic_api_key,
        use_unsloth=True
    )
    
    # Run all benchmarks.
    benchmark.multiple_benchmarks()
    # Save the DataFrame of results to a pickle file.
    benchmark.save_results("bench.pkl")
    # Plot the aggregated results.
    benchmark.plot_results()

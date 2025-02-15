import time
import numpy as np
import pickle
import pandas as pd
from agents.LLMagent import LLMagent
from manager.TaskManager import TaskManager

class BenchmarkRunner:
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
                    metrics = manager.run_simulations()
                    # Append the success rate (or 0 if not found) to the list.
                    agent_results[rounds_label][quadrant_label].append(metrics.get('success_rate', 0))
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
        Each row contains Model, Rounds, Quadrants, Performance (mean) and Std (std).
        """
        rows = []
        for model, rounds_dict in self.results.items():
            for rounds_label, quadrants_dict in rounds_dict.items():
                for quadrant_label, runs in quadrants_dict.items():
                    performance = np.mean(runs)
                    std = np.std(runs)
                    row = {
                        "Model": model,
                        "Rounds": rounds_label,
                        "Quadrants": quadrant_label,
                        "Performance": performance,
                        "Std": std
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
    # Transform results into a DataFrame and save to bench.pkl.
    benchmark.save_results("bench.pkl")

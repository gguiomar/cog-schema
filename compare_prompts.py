# compare_prompt_versions.py
from manager.TaskManager import TaskManager
from tasks.TaskSelector import TaskSelector
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def main():
    # Configuration
    model_name = "Qwen_0.5B_Instruct"  
    rounds = range(2,14)
    quadrants = [4]
    n_simulations = 10
    n_trials = 10
    device = "cuda:0"  # Change as needed
    
    # Prepare to collect results across versions
    all_results = {}
    all_dataframes = []
    
    # Run each prompt version separately
    for version in range(5):  # Testing versions 0-4
        print(f"\n===== Testing Prompt Version {version} =====\n")
        # Create a TaskManager for this version
        print(f"Initializing TaskManager for prompt version {version}")
        manager = TaskManager(
            agents=[model_name],
            rounds=rounds,
            quadrants=quadrants,
            n_simulations=n_simulations,
            n_trials=n_trials,
            device=device,
            verbose=False,
            task_type=TaskSelector.BIAS_DETECTION
        )
        
        # Run benchmarks for this version
        print(f"Running benchmarks for prompt version {version}")
        version_results = manager.multiple_benchmarks()
        all_results[f"version_{version}"] = version_results
        
        #Save the results
        df = manager.save_results()
        # Add version info to DataFrame
        df["Prompt_Version"] = version
        all_dataframes.append(df)
        
        #Generate version-specific plot
        manager.plot_results(output_path=f"prompt_v{version}_comparison.png")
    
    # Combine all dataframes for comparison
    combined_df = pd.concat(all_dataframes)
    combined_df.to_csv("prompt_version_comparison.csv", index=False)
    
    # Print summary
    print("\n===== Summary of Prompt Version Performance =====\n")
    summary_data = []
    for version in range(5):
        version_df = combined_df[combined_df["Prompt_Version"] == version]
        avg_performance = version_df["Performance"].mean()
        summary_data.append({
            "Version": version,
            "Average Performance": avg_performance
        })
        print(f"Prompt Version {version}: Average Success Rate = {avg_performance:.4f}")
    
    # Create summary plot
    create_summary_plot(summary_data)
    
    print("\nComplete! Check 'prompt_version_comparison.csv' and 'prompt_version_summary.png'")

def create_summary_plot(summary_data):
    versions = [item["Version"] for item in summary_data]
    performances = [item["Average Performance"] for item in summary_data]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(versions, performances, color='skyblue', edgecolor='black')
    
    # Add value labels on top of bars
    for bar, perf in zip(bars, performances):
        plt.text(bar.get_x() + bar.get_width()/2, 
                 bar.get_height() + 0.01, 
                 f'{perf:.3f}', 
                 ha='center', va='bottom')
    
    plt.xlabel('Prompt Version')
    plt.ylabel('Average Success Rate')
    plt.title('Comparison of Prompt Version Performance')
    plt.xticks(versions)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.ylim(0, max(performances) * 1.15) 
    
    plt.tight_layout()
    plt.savefig("prompt_version_summary.png", dpi=300)

if __name__ == "__main__":
    main()
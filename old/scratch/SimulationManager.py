import json
import os
from datetime import datetime
from typing import Dict, List
import pandas as pd
import numpy as np
from tqdm import tqdm
import transformers
from agents.LLMTaskRunner import DetailedLLMTaskRunner

class SimulationManager:
    def __init__(self, n_simulations: int, nrounds: int, num_quadrants: int, 
                 pipe: transformers.Pipeline, output_dir: str = "simulation_results"):
        self.n_simulations = n_simulations
        self.nrounds = nrounds
        self.num_quadrants = num_quadrants
        self.pipe = pipe
        self.output_dir = output_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
    def run_simulations(self) -> Dict:
        """Run multiple simulations and save results"""
        all_results = []
        
        # Create log file
        log_filename = f"{self.output_dir}/simulation_log_{self.timestamp}.txt"
        results_filename = f"{self.output_dir}/simulation_results_{self.timestamp}.json"
        
        with open(log_filename, 'w') as log_file:
            log_file.write(f"Starting {self.n_simulations} simulations\n")
            log_file.write(f"Rounds per simulation: {self.nrounds}\n")
            log_file.write(f"Number of quadrants: {self.num_quadrants}\n\n")
            
            # Run simulations with progress bar
            for sim_num in tqdm(range(self.n_simulations)):
                runner = DetailedLLMTaskRunner(
                    nrounds=self.nrounds,
                    num_quadrants=self.num_quadrants,
                    pipe=self.pipe
                )
                
                # Run simulation and get results
                stats = runner.run_with_output()
                all_results.append(stats)
                
                # Log detailed results for this simulation
                log_file.write(f"\n=== Simulation {sim_num + 1} ===\n")
                log_file.write(f"Correct quadrant: {stats['correct_quadrant']}\n")
                log_file.write(f"LLM choice: {stats['final_choice']}\n")
                log_file.write(f"Success: {stats['success']}\n")
                log_file.write("Round by round:\n")
                
                for round_num, round_data in enumerate(stats['rounds'], 1):
                    log_file.write(f"Round {round_num}:\n")
                    log_file.write(f"  Squares shown: {round_data['squares_shown']}\n")
                    log_file.write(f"  Choice: {round_data['choice']}\n")
                    log_file.write(f"  Result: {round_data['result']}\n")
                log_file.write("\n")
                
        # Save all raw results to JSON
        with open(results_filename, 'w') as f:
            json.dump(all_results, f, indent=2)
            
        return self.analyze_results(all_results)
    
    def analyze_results(self, results: List[Dict]) -> Dict:
        """Generate metrics from simulation results"""
        metrics = {
            'n_simulations': self.n_simulations,
            'success_rate': np.mean([r['success'] for r in results]),
            'quadrant_distribution': {},
            'color_distributions': {},
            'round_statistics': {},
            'timestamp': self.timestamp
        }
        
        # Calculate quadrant selection frequencies
        llm_choices = [r['final_choice'] for r in results]
        correct_quadrants = [r['correct_quadrant'] for r in results]
        
        for q in range(1, self.num_quadrants + 1):
            metrics['quadrant_distribution'][f'quadrant_{q}'] = {
                'times_chosen': llm_choices.count(str(q)),
                'times_correct': correct_quadrants.count(q)
            }
        
        # Analyze round-by-round behavior
        for round_num in range(self.nrounds):
            round_stats = {
                'red_count': 0,
                'green_count': 0,
                'invalid_choices': 0
            }
            
            for result in results:
                if round_num < len(result['rounds']):
                    round_data = result['rounds'][round_num]
                    if round_data['result'] == 'RED':
                        round_stats['red_count'] += 1
                    elif round_data['result'] == 'GREEN':
                        round_stats['green_count'] += 1
                    elif round_data['result'] is None:
                        round_stats['invalid_choices'] += 1
                        
            metrics['round_statistics'][f'round_{round_num + 1}'] = round_stats
        
        # Save metrics to file
        metrics_filename = f"{self.output_dir}/metrics_{self.timestamp}.json"
        with open(metrics_filename, 'w') as f:
            json.dump(metrics, f, indent=2)
            
        # Generate and save summary plots
        self.generate_plots(metrics)
            
        return metrics
    
    def generate_plots(self, metrics: Dict) -> None:
        """Generate visualization of results using matplotlib"""
        import matplotlib.pyplot as plt
        
        # Success rate over time plot
        plt.figure(figsize=(10, 6))
        plt.bar(['Success Rate'], [metrics['success_rate']], color='green')
        plt.title('Overall Success Rate')
        plt.ylim(0, 1)
        plt.savefig(f"{self.output_dir}/success_rate_{self.timestamp}.png")
        plt.close()
        
        # Quadrant selection distribution
        plt.figure(figsize=(10, 6))
        quadrants = list(metrics['quadrant_distribution'].keys())
        times_chosen = [d['times_chosen'] for d in metrics['quadrant_distribution'].values()]
        times_correct = [d['times_correct'] for d in metrics['quadrant_distribution'].values()]
        
        x = np.arange(len(quadrants))
        width = 0.35
        
        plt.bar(x - width/2, times_chosen, width, label='Times Chosen')
        plt.bar(x + width/2, times_correct, width, label='Times Correct')
        plt.xlabel('Quadrant')
        plt.ylabel('Count')
        plt.title('Quadrant Selection Distribution')
        plt.xticks(x, quadrants)
        plt.legend()
        plt.savefig(f"{self.output_dir}/quadrant_distribution_{self.timestamp}.png")
        plt.close()

import os
import json
import numpy as np
from datetime import datetime
from typing import Dict, List
from tqdm import tqdm
from agents.LLMAgent import LLMAgent
from quad.VSTtask import VSTtask


class TaskManager:
    def __init__(self, n_simulations: int, nrounds: int, num_quadrants: int, 
                 pipe: LLMAgent, output_dir: str = "simulation_results"):
        """Initialize task manager with simulation parameters."""
        self.n_simulations = n_simulations
        self.nrounds = nrounds
        self.num_quadrants = num_quadrants
        self.agent = pipe
        self.output_dir = output_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        os.makedirs(output_dir, exist_ok=True)
        
    def run_single_task(self) -> Dict:
        """Run a single VST task and return results."""
        task = VSTtask(self.nrounds, self.num_quadrants)
        self.agent.reset_history()
        
        stats = {
            'rounds': [],
            'final_choice': None,
            'correct_quadrant': task.biased_quadrant + 1,
            'success': False
        }
        
        # Initialize conversation with task description
        self.agent.update_history(task.get_task_description())
        
        # Run all rounds
        for round_num in range(task.n_rounds):
            round_data = task.get_round_data(round_num)
            available_queues = [q['name'] for q in round_data]
            
            # Get agent's choice
            prompt = (
                f"Round {round_num + 1}: Available queues {', '.join(available_queues)}. "
                "Choose one queue by responding with just the letter. You press <<"
            )
            choice = self.agent.get_response(prompt)
            
            # Process choice
            result = task.process_choice(choice, round_data)
            
            # Update history and stats
            if result:
                round_text = (
                    f"Round {round_num + 1}: Available queues {', '.join(available_queues)}. "
                    f"You chose {choice} and saw {result}.\n"
                )
                self.agent.update_history(round_text)
                
            stats['rounds'].append({
                'available_queues': available_queues,
                'choice': choice,
                'result': result
            })
        
        # Get final answer
        final_prompt = (
            f"Based on all observations, which quadrant (1-{task.n_quadrants}) "
            "had the highest ratio of RED? Respond with just the number. You choose <<"
        )
        final_choice = self.agent.get_response(final_prompt)
        
        stats['final_choice'] = final_choice
        stats['success'] = str(task.biased_quadrant + 1) == final_choice
        
        return stats
    
    def run_simulations(self) -> Dict:
        """Run multiple simulations and analyze results."""
        all_results = []
        log_filename = f"{self.output_dir}/simulation_log_{self.timestamp}.txt"
        results_filename = f"{self.output_dir}/simulation_results_{self.timestamp}.json"
        
        with open(log_filename, 'w') as log_file:
            log_file.write(f"Starting {self.n_simulations} simulations\n")
            log_file.write(f"Rounds per simulation: {self.nrounds}\n")
            log_file.write(f"Number of quadrants: {self.num_quadrants}\n\n")
            
            for sim_num in tqdm(range(self.n_simulations)):
                stats = self.run_single_task()
                all_results.append(stats)
                
                # Log detailed results
                log_file.write(f"\n=== Simulation {sim_num + 1} ===\n")
                log_file.write(f"Correct quadrant: {stats['correct_quadrant']}\n")
                log_file.write(f"LLM choice: {stats['final_choice']}\n")
                log_file.write(f"Success: {stats['success']}\n")
                log_file.write("Round by round:\n")
                
                for round_num, round_data in enumerate(stats['rounds'], 1):
                    log_file.write(f"Round {round_num}:\n")
                    log_file.write(f"  Queues shown: {round_data['available_queues']}\n")
                    log_file.write(f"  Choice: {round_data['choice']}\n")
                    log_file.write(f"  Result: {round_data['result']}\n")
                    
        # Save raw results
        with open(results_filename, 'w') as f:
            json.dump(all_results, f, indent=2)
            
        return self.analyze_results(all_results)
    
    def analyze_results(self, results: List[Dict]) -> Dict:
        """Analyze simulation results and generate metrics."""
        metrics = {
            'n_simulations': self.n_simulations,
            'success_rate': np.mean([r['success'] for r in results]),
            'quadrant_distribution': {},
            'timestamp': self.timestamp
        }
        
        # Calculate quadrant selection frequencies
        for q in range(1, self.num_quadrants + 1):
            times_chosen = len([r for r in results if r['final_choice'] == str(q)])
            times_correct = len([r for r in results if r['correct_quadrant'] == q])
            
            metrics['quadrant_distribution'][f'quadrant_{q}'] = {
                'times_chosen': times_chosen,
                'times_correct': times_correct
            }
        
        # Save metrics
        metrics_filename = f"{self.output_dir}/metrics_{self.timestamp}.json"
        with open(metrics_filename, 'w') as f:
            json.dump(metrics, f, indent=2)
            
        return metrics
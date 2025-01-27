import os
from datetime import datetime
import json
from typing import List, Dict, Tuple
import numpy as np
from tqdm import tqdm
from tasks.VSTtask import VSTtask
from agents.LLMagent import LLMagent

import os
from datetime import datetime
import json
from typing import List, Dict, Tuple
import numpy as np
from tqdm import tqdm
from tasks.VSTtask import VSTtask
from agents.LLMagent import LLMagent

class TaskManager:
    def __init__(self, n_simulations: int, nrounds: int, num_quadrants: int, 
                 pipe: LLMagent, output_dir: str = "simulation_results", verbose: bool = False):
        """Initialize task manager with simulation parameters."""
        self.n_simulations = n_simulations
        self.nrounds = nrounds
        self.num_quadrants = num_quadrants
        self.agent = pipe
        self.output_dir = output_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.verbose = verbose
        self.conversation_history = ""
        
        os.makedirs(output_dir, exist_ok=True)
    
    def build_prompt(self, available_queues: str, round_num: int) -> str:
        """Build prompt including conversation history."""
        # Add the current round information
        current_prompt = (
            f"Round {round_num + 1}: Available queues {available_queues}. "
            f"Based on previous observations, choose one queue by responding with just the letter. You press <<"
        )
        
        return self.conversation_history + current_prompt
        
    def update_history(self, queues: str, choice: str, result: str, round_num: int) -> None:
        """Update conversation history with round results."""
        round_text = (
            f"Round {round_num + 1}: Available queues {queues}. "
            f"You chose {choice} and saw {result}.\n"
        )
        self.conversation_history += round_text
        
    def get_final_prompt(self) -> str:
        """Build final prompt including full conversation history."""
        prompt = (
            self.conversation_history +
            "Based on all observed colors, which quadrant (1"
            f"{', ' + ', '.join(str(i) for i in range(2, self.num_quadrants + 1))}"
            ") do you think had the highest ratio of RED? "
            "Respond with just the number. You choose <<"
        )
        return prompt
        
    def run_single_task(self) -> Dict:
        """Run a single VST task and return results."""
        self.task = VSTtask(self.nrounds, self.num_quadrants)
        self.conversation_history = ""
        self.agent.reset_history()
        
        stats = {
            'rounds': [],
            'final_choice': None,
            'correct_quadrant': self.task.biased_quadrant + 1,
            'success': False,
            'full_conversation': []
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
        for round_num in range(self.task.n_rounds):
            if self.verbose:
                print(f"\n--- Round {round_num + 1} ---")
            
            round_data = self.task.get_round_data(round_num)
            available_queues = [q['name'] for q in round_data]
            
            # Build and show prompt with accumulated history
            prompt = self.build_prompt(', '.join(available_queues), round_num)
            
            if self.verbose:
                print("\nAccumulated prompt shown to LLM:")
                print("--------------------")
                print(prompt)
                print("--------------------")
            
            # Get agent's choice
            choice = self.agent.get_response(prompt)
            
            if self.verbose:
                print(f"\nLLM chose: {choice}")
            
            # Process choice
            result = self.task.process_choice(choice, round_data)
            
            if self.verbose:
                if result:
                    print(f"Result: {result}")
                    # Show quadrant info in verbose mode
                    for q in round_data:
                        if q['name'] == choice:
                            print(f"(Queue {choice} was from Quadrant {q['quadrant'] + 1})")
                else:
                    print("Invalid choice!")
            
            # Update history and stats
            if result:
                self.update_history(', '.join(available_queues), choice, result, round_num)
                
            stats['rounds'].append({
                'available_queues': available_queues,
                'choice': choice,
                'result': result,
                'full_prompt': prompt  # Store the full accumulated prompt
            })
            
            # Store the sequential nature of the conversation
            stats['full_conversation'].extend([
                ("ACCUMULATED_PROMPT", prompt),
                ("LLM_CHOICE", choice),
                ("RESULT", result)
            ])
        
        # Get final answer
        if self.verbose:
            print("\n=== Final Decision ===")
            
        final_prompt = self.get_final_prompt()
        
        if self.verbose:
            print("\nFinal accumulated prompt shown to LLM:")
            print("-------------------------")
            print(final_prompt)
            print("-------------------------")
        
        final_choice = self.agent.get_response(final_prompt)
        
        if self.verbose:
            print(f"\nLLM's final choice: Quadrant {final_choice}")
            print("\n=== Game Results ===")
            print(f"Correct quadrant: {self.task.biased_quadrant + 1}")
            print(f"LLM chose: {final_choice}")
            print(f"Success: {str(self.task.biased_quadrant + 1) == final_choice}")
        
        stats['final_choice'] = final_choice
        stats['success'] = str(self.task.biased_quadrant + 1) == final_choice
        
        stats['full_conversation'].extend([
            ("FINAL_ACCUMULATED_PROMPT", final_prompt),
            ("FINAL_CHOICE", final_choice)
        ])
        
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
            
            for sim_num in tqdm(range(self.n_simulations), disable=self.verbose):
                if self.verbose:
                    print(f"\nSimulation {sim_num + 1}/{self.n_simulations}")
                    print("=" * 50)
                
                stats = self.run_single_task()
                all_results.append(stats)
                
                # Log detailed results including full conversation history
                log_file.write(f"\n=== Simulation {sim_num + 1} ===\n")
                log_file.write(f"Correct quadrant: {stats['correct_quadrant']}\n")
                log_file.write(f"LLM choice: {stats['final_choice']}\n")
                log_file.write(f"Success: {stats['success']}\n\n")
                
                # Log the complete conversation showing prompt accumulation
                log_file.write("=== Complete Conversation Log ===\n")
                for msg_type, content in stats['full_conversation']:
                    log_file.write(f"\n{msg_type}:\n")
                    log_file.write("-" * 40 + "\n")
                    log_file.write(f"{content}\n")
                    log_file.write("-" * 40 + "\n")
                log_file.write("\n" + "=" * 50 + "\n")
                    
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

# class TaskManager:
#     def __init__(self, n_simulations: int, nrounds: int, num_quadrants: int, 
#                  pipe: LLMagent, output_dir: str = "simulation_results", verbose: bool = False):
#         """Initialize task manager with simulation parameters."""
#         self.n_simulations = n_simulations
#         self.nrounds = nrounds
#         self.num_quadrants = num_quadrants
#         self.agent = pipe
#         self.output_dir = output_dir
#         self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         self.verbose = verbose
        
#         os.makedirs(output_dir, exist_ok=True)
        
#     def run_single_task(self) -> Dict:
#         """Run a single VST task and return results."""
#         task = VSTtask(self.nrounds, self.num_quadrants)
#         self.agent.reset_history()
        
#         stats = {
#             'rounds': [],
#             'final_choice': None,
#             'correct_quadrant': task.biased_quadrant + 1,
#             'success': False,
#             'full_conversation': []
#         }
        
#         # Initialize conversation with task description
#         task_description = task.get_task_description()
#         self.agent.update_history(task_description)
        
#         if self.verbose:
#             print("\n=== Task Description ===")
#             print(task_description)
#             print("\n=== Beginning Rounds ===")
        
#         stats['full_conversation'].append(("SYSTEM", task_description))
        
#         # Run all rounds
#         for round_num in range(task.n_rounds):
#             if self.verbose:
#                 print(f"\n--- Round {round_num + 1} ---")
            
#             round_data = task.get_round_data(round_num)
#             available_queues = [q['name'] for q in round_data]
            
#             # Build and show prompt
#             prompt = (
#                 f"Round {round_num + 1}: Available queues {', '.join(available_queues)}. "
#                 "Choose one queue by responding with just the letter. You press <<"
#             )
            
#             if self.verbose:
#                 print("\nPrompt shown to LLM:")
#                 print("--------------------")
#                 print(prompt)
#                 print("--------------------")
            
#             # Get agent's choice
#             choice = self.agent.get_response(prompt)
            
#             if self.verbose:
#                 print(f"\nLLM chose: {choice}")
            
#             # Process choice
#             result = task.process_choice(choice, round_data)
            
#             if self.verbose:
#                 if result:
#                     print(f"Result: {result}")
#                     # Show quadrant info in verbose mode
#                     for q in round_data:
#                         if q['name'] == choice:
#                             print(f"(Queue {choice} was from Quadrant {q['quadrant'] + 1})")
#                 else:
#                     print("Invalid choice!")
            
#             # Update history and stats
#             if result:
#                 round_text = (
#                     f"Round {round_num + 1}: Available queues {', '.join(available_queues)}. "
#                     f"You chose {choice} and saw {result}.\n"
#                 )
#                 self.agent.update_history(round_text)
                
#             stats['rounds'].append({
#                 'available_queues': available_queues,
#                 'choice': choice,
#                 'result': result,
#                 'prompt': prompt
#             })
            
#             stats['full_conversation'].extend([
#                 ("PROMPT", prompt),
#                 ("LLM", choice),
#                 ("RESULT", result)
#             ])
        
#         # Get final answer
#         if self.verbose:
#             print("\n=== Final Decision ===")
            
#         final_prompt = (
#             f"Based on all observations, which quadrant (1-{task.n_quadrants}) "
#             "had the highest ratio of RED? Respond with just the number. You choose <<"
#         )
        
#         if self.verbose:
#             print("\nFinal prompt shown to LLM:")
#             print("-------------------------")
#             print(final_prompt)
#             print("-------------------------")
        
#         final_choice = self.agent.get_response(final_prompt)
        
#         if self.verbose:
#             print(f"\nLLM's final choice: Quadrant {final_choice}")
#             print("\n=== Game Results ===")
#             print(f"Correct quadrant: {task.biased_quadrant + 1}")
#             print(f"LLM chose: {final_choice}")
#             print(f"Success: {str(task.biased_quadrant + 1) == final_choice}")
        
#         stats['final_choice'] = final_choice
#         stats['success'] = str(task.biased_quadrant + 1) == final_choice
        
#         stats['full_conversation'].extend([
#             ("FINAL_PROMPT", final_prompt),
#             ("FINAL_CHOICE", final_choice)
#         ])
        
#         return stats
    
#     def run_simulations(self) -> Dict:
#         """Run multiple simulations and analyze results."""
#         all_results = []
#         log_filename = f"{self.output_dir}/simulation_log_{self.timestamp}.txt"
#         results_filename = f"{self.output_dir}/simulation_results_{self.timestamp}.json"
        
#         with open(log_filename, 'w') as log_file:
#             log_file.write(f"Starting {self.n_simulations} simulations\n")
#             log_file.write(f"Rounds per simulation: {self.nrounds}\n")
#             log_file.write(f"Number of quadrants: {self.num_quadrants}\n\n")
            
#             for sim_num in tqdm(range(self.n_simulations), disable=self.verbose):
#                 if self.verbose:
#                     print(f"\nSimulation {sim_num + 1}/{self.n_simulations}")
#                     print("=" * 50)
                
#                 stats = self.run_single_task()
#                 all_results.append(stats)
                
#                 # Log detailed results including full conversation
#                 log_file.write(f"\n=== Simulation {sim_num + 1} ===\n")
#                 log_file.write(f"Correct quadrant: {stats['correct_quadrant']}\n")
#                 log_file.write(f"LLM choice: {stats['final_choice']}\n")
#                 log_file.write(f"Success: {stats['success']}\n")
#                 log_file.write("\nFull conversation:\n")
                
#                 for msg_type, content in stats['full_conversation']:
#                     log_file.write(f"{msg_type}: {content}\n")
                    
#         # Save raw results
#         with open(results_filename, 'w') as f:
#             json.dump(all_results, f, indent=2)
            
#         return self.analyze_results(all_results)
    
#     def analyze_results(self, results: List[Dict]) -> Dict:
#         """Analyze simulation results and generate metrics."""
#         metrics = {
#             'n_simulations': self.n_simulations,
#             'success_rate': np.mean([r['success'] for r in results]),
#             'quadrant_distribution': {},
#             'timestamp': self.timestamp
#         }
        
#         # Calculate quadrant selection frequencies
#         for q in range(1, self.num_quadrants + 1):
#             times_chosen = len([r for r in results if r['final_choice'] == str(q)])
#             times_correct = len([r for r in results if r['correct_quadrant'] == q])
            
#             metrics['quadrant_distribution'][f'quadrant_{q}'] = {
#                 'times_chosen': times_chosen,
#                 'times_correct': times_correct
#             }
        
#         # Save metrics
#         metrics_filename = f"{self.output_dir}/metrics_{self.timestamp}.json"
#         with open(metrics_filename, 'w') as f:
#             json.dump(metrics, f, indent=2)
            
#         return metrics

# class TaskManager:
#     def __init__(self, n_simulations: int, nrounds: int, num_quadrants: int, 
#                  pipe: LLMagent, output_dir: str = "simulation_results"):
#         """Initialize task manager with simulation parameters."""
#         self.n_simulations = n_simulations
#         self.nrounds = nrounds
#         self.num_quadrants = num_quadrants
#         self.agent = pipe
#         self.output_dir = output_dir
#         self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
#         os.makedirs(output_dir, exist_ok=True)
        
#     def run_single_task(self) -> Dict:
#         """Run a single VST task and return results."""
#         task = VSTtask(self.nrounds, self.num_quadrants)
#         self.agent.reset_history()
        
#         stats = {
#             'rounds': [],
#             'final_choice': None,
#             'correct_quadrant': task.biased_quadrant + 1,
#             'success': False
#         }
        
#         # Initialize conversation with task description
#         self.agent.update_history(task.get_task_description())
        
#         # Run all rounds
#         for round_num in range(task.n_rounds):
#             round_data = task.get_round_data(round_num)
#             available_queues = [q['name'] for q in round_data]
            
#             # Get agent's choice
#             prompt = (
#                 f"Round {round_num + 1}: Available queues {', '.join(available_queues)}. "
#                 "Choose one queue by responding with just the letter. You press <<"
#             )
#             choice = self.agent.get_response(prompt)
            
#             # Process choice
#             result = task.process_choice(choice, round_data)
            
#             # Update history and stats
#             if result:
#                 round_text = (
#                     f"Round {round_num + 1}: Available queues {', '.join(available_queues)}. "
#                     f"You chose {choice} and saw {result}.\n"
#                 )
#                 self.agent.update_history(round_text)
                
#             stats['rounds'].append({
#                 'available_queues': available_queues,
#                 'choice': choice,
#                 'result': result
#             })
        
#         # Get final answer
#         final_prompt = (
#             f"Based on all observations, which quadrant (1-{task.n_quadrants}) "
#             "had the highest ratio of RED? Respond with just the number. You choose <<"
#         )
#         final_choice = self.agent.get_response(final_prompt)
        
#         stats['final_choice'] = final_choice
#         stats['success'] = str(task.biased_quadrant + 1) == final_choice
        
#         return stats
    
#     def run_simulations(self) -> Dict:
#         """Run multiple simulations and analyze results."""
#         all_results = []
#         log_filename = f"{self.output_dir}/simulation_log_{self.timestamp}.txt"
#         results_filename = f"{self.output_dir}/simulation_results_{self.timestamp}.json"
        
#         with open(log_filename, 'w') as log_file:
#             log_file.write(f"Starting {self.n_simulations} simulations\n")
#             log_file.write(f"Rounds per simulation: {self.nrounds}\n")
#             log_file.write(f"Number of quadrants: {self.num_quadrants}\n\n")
            
#             for sim_num in tqdm(range(self.n_simulations)):
#                 stats = self.run_single_task()
#                 all_results.append(stats)
                
#                 # Log detailed results
#                 log_file.write(f"\n=== Simulation {sim_num + 1} ===\n")
#                 log_file.write(f"Correct quadrant: {stats['correct_quadrant']}\n")
#                 log_file.write(f"LLM choice: {stats['final_choice']}\n")
#                 log_file.write(f"Success: {stats['success']}\n")
#                 log_file.write("Round by round:\n")
                
#                 for round_num, round_data in enumerate(stats['rounds'], 1):
#                     log_file.write(f"Round {round_num}:\n")
#                     log_file.write(f"  Queues shown: {round_data['available_queues']}\n")
#                     log_file.write(f"  Choice: {round_data['choice']}\n")
#                     log_file.write(f"  Result: {round_data['result']}\n")
                    
#         # Save raw results
#         with open(results_filename, 'w') as f:
#             json.dump(all_results, f, indent=2)
            
#         return self.analyze_results(all_results)
    
#     def analyze_results(self, results: List[Dict]) -> Dict:
#         """Analyze simulation results and generate metrics."""
#         metrics = {
#             'n_simulations': self.n_simulations,
#             'success_rate': np.mean([r['success'] for r in results]),
#             'quadrant_distribution': {},
#             'timestamp': self.timestamp
#         }
        
#         # Calculate quadrant selection frequencies
#         for q in range(1, self.num_quadrants + 1):
#             times_chosen = len([r for r in results if r['final_choice'] == str(q)])
#             times_correct = len([r for r in results if r['correct_quadrant'] == q])
            
#             metrics['quadrant_distribution'][f'quadrant_{q}'] = {
#                 'times_chosen': times_chosen,
#                 'times_correct': times_correct
#             }
        
#         # Save metrics
#         metrics_filename = f"{self.output_dir}/metrics_{self.timestamp}.json"
#         with open(metrics_filename, 'w') as f:
#             json.dump(metrics, f, indent=2)
            
#         return metrics
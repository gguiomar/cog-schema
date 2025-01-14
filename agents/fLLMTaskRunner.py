import random
from typing import List, Tuple
import transformers
from quad.quadtext_min import ModifiedTask

class DetailedLLMTaskRunner:
    def __init__(self, nrounds: int, num_quadrants: int, pipe: transformers.Pipeline):
        self.task = ModifiedTask(nrounds=nrounds, num_quadrants=num_quadrants)
        self.pipe = pipe
        self.conversation_history = ""
        self.round_history: List[Tuple[str, str, str]] = []
        
    def build_prompt(self, available_squares: str) -> str:
        # Start with the task description
        if not self.conversation_history:
            self.conversation_history = self.task.generate_task_description() + "\n\n"
            
        # Add the current round information
        current_prompt = (
            f"Round {self.task.current_round}: You see black squares {available_squares}. "
            f"Based on previous observations, choose one square to sample by responding with just the letter. You press <<"
        )
        
        return self.conversation_history + current_prompt
    
    def update_history(self, squares_shown: str, choice: str, result: str) -> None:
        # Add the round result to the conversation history
        round_text = (
            f"Round {self.task.current_round}: You see black squares {squares_shown}. "
            f"You press <<{choice}>> and see the color {result}.\n"
        )
        self.conversation_history += round_text
        self.round_history.append((squares_shown, choice, result))
    
    def get_final_prompt(self) -> str:
        prompt = (
            self.conversation_history +
            "Based on all observed colors, which quadrant (1, 2"
            f"{', 3' if len(self.task.quadrants) > 2 else ''}"
            f"{', 4' if len(self.task.quadrants) > 3 else ''}"
            ") do you think had the highest ratio difference between RED and GREEN? "
            "Respond with just the number. You choose <<"
        )
        return prompt
    
    def run_with_output(self) -> dict:
        """Run the task with detailed output of all interactions"""
        stats = {
            'rounds': [],
            'final_choice': None,
            'correct_quadrant': None,
            'success': False,
            'full_conversation': []
        }
        
        # Print initial task description
        initial_description = self.task.generate_task_description()
        print("=== Task Description ===")
        print(initial_description)
        print("\n=== Beginning Rounds ===")
        stats['full_conversation'].append(("SYSTEM", initial_description))
        
        # Run all rounds
        for current_round in range(self.task.nrounds):
            print(f"\n--- Round {current_round + 1} ---")
            
            # Generate and show available squares
            self.task.play_round()
            available_squares = ', '.join(square['name'] for square in self.task.black_squares)
            
            # Build and show prompt
            prompt = self.build_prompt(available_squares)
            print("\nPrompt shown to LLM:")
            print("--------------------")
            print(prompt)
            print("--------------------")
            
            # Get and show LLM's choice
            choice = self.pipe(prompt)[0]['generated_text'][len(prompt):].strip()
            print(f"\nLLM chose: {choice}")
            
            # Get and show result
            result = self.task.get_task_output(choice)
            if result:
                print(f"Result: {result}")
                self.update_history(available_squares, choice, result)
            else:
                print("Invalid choice!")
                
            stats['rounds'].append({
                'squares_shown': available_squares,
                'choice': choice,
                'result': result,
                'prompt': prompt
            })
            stats['full_conversation'].append(("PROMPT", prompt))
            stats['full_conversation'].append(("LLM", choice))
            stats['full_conversation'].append(("RESULT", result))
        
        # Final decision
        print("\n=== Final Decision ===")
        final_prompt = self.get_final_prompt()
        print("\nFinal prompt shown to LLM:")
        print("-------------------------")
        print(final_prompt)
        print("-------------------------")
        
        final_choice = self.pipe(final_prompt)[0]['generated_text'][len(final_prompt):].strip()
        print(f"\nLLM's final choice: Quadrant {final_choice}")
        
        # Record results
        stats['final_choice'] = final_choice
        stats['correct_quadrant'] = self.task.biased_quadrant + 1
        stats['success'] = str(self.task.biased_quadrant + 1) == final_choice
        
        # Print final results
        print("\n=== Game Results ===")
        print(f"Correct quadrant: {stats['correct_quadrant']}")
        print(f"LLM chose: {stats['final_choice']}")
        print(f"Success: {stats['success']}")
        
        stats['full_conversation'].append(("FINAL_PROMPT", final_prompt))
        stats['full_conversation'].append(("FINAL_CHOICE", final_choice))
        
        return stats

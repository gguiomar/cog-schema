import random
from typing import List, Tuple
import transformers
from quad.quadtext_min import ModifiedTask

class LLMTaskRunner:
    def __init__(self, nrounds: int, num_quadrants: int, pipe: transformers.Pipeline):
        self.task = ModifiedTask(nrounds=nrounds, num_quadrants=num_quadrants)
        self.pipe = pipe
        self.conversation_history = ""
        self.round_history: List[Tuple[str, str, str]] = []  # (squares shown, choice, result)
        
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
    
    def run(self) -> dict:
        """Run the full task with the LLM and return statistics"""
        stats = {
            'rounds': [],
            'final_choice': None,
            'correct_quadrant': None,
            'success': False
        }
        
        # Run all rounds
        for _ in range(self.task.nrounds):
            self.task.play_round()
            available_squares = ', '.join(square['name'] for square in self.task.black_squares)
            
            # Get LLM's choice
            prompt = self.build_prompt(available_squares)
            choice = self.pipe(prompt)[0]['generated_text'][len(prompt):].strip()
            
            # Get result and update history
            result = self.task.get_task_output(choice)
            if result:
                self.update_history(available_squares, choice, result)
            stats['rounds'].append({
                'squares_shown': available_squares,
                'choice': choice,
                'result': result
            })
        
        # Get final answer
        final_prompt = self.get_final_prompt()
        final_choice = self.pipe(final_prompt)[0]['generated_text'][len(final_prompt):].strip()
        
        # Record results
        stats['final_choice'] = final_choice
        stats['correct_quadrant'] = self.task.biased_quadrant + 1
        stats['success'] = str(self.task.biased_quadrant + 1) == final_choice
        
        return stats
    
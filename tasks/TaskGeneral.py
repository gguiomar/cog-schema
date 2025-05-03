from typing import Dict, Optional
from util.util import *

class TaskGeneral:
    def __init__(self, n_rounds: int = 1, n_quadrants: int = 4, n_cues: int = 1):
        self.n_rounds = n_rounds
        self.n_quadrants = n_quadrants
        self.quadrants = list(range(self.n_quadrants))
        self.n_cues = n_cues
        self.current_round = 0
        self.current_trial = 0

        self.correct_answer = None
        self.current_answer = None

        # Setup quadrants and cues
        self.letters = [chr(65 + i) for i in range(self.n_quadrants * self.n_cues)]
        self.cue_map = {
            q: self.letters[q*self.n_cues:(q+1)*self.n_cues]
            for q in range(self.n_quadrants)
        }
    

    def get_trial_separator(self) -> Optional[str]:
        try:
            prompt = load_prompt_from_xml(self.strings, 'trial_separator')
            return prompt.format(trial_num = self.current_trial + 1)
        except AttributeError:
            raise ValueError("Trial separator not defined for this task.")

    def get_initial_prompt(self):
        try:
            prompt = load_prompt_from_xml(self.strings, 'initial_prompt')
            return prompt.format(n_rounds=self.n_rounds)
        except AttributeError:
            raise ValueError("Initial prompt not defined for this task.")

    def get_intermediate_prompt(self) -> str:
        try:
            round_data = self.get_round_data(self.current_round)
            available_cues = [q['name'] for q in round_data]

            # Build and show prompt with accumulated history
            self.available_cues = ', '.join(available_cues)
            print("-------------------------------------")
            print(self.strings)

            prompt = load_prompt_from_xml(self.strings, 'intermediate_prompt')
            print(prompt)
            return prompt.format(
                current_trial=self.current_trial + 1,
                current_round=self.current_round + 1,
                available_cues=self.available_cues
            )
        except AttributeError:
            raise ValueError("Intermediate prompt not defined for this task.")

    def get_final_prompt(self) -> str:
        try:
            prompt = load_prompt_from_xml(self.strings, 'final_prompt')
            return prompt.format(
                current_trial=self.current_trial + 1,
                letters=self.letters
            )
        except AttributeError:
            raise ValueError("Final prompt not defined for this task.")
        
    def give_final_feedback(self) -> Optional[str]:
        pass

    def process_choice(self) -> Optional[str]:
        pass

    def process_final_choice(self) -> Optional[str]:
        pass

    def print_final_log(self):
        pass

    def update_answer(self, answer: str):
        self.current_answer = answer

    def get_correct_answer(self) -> str:
        return self.correct_answer

    def give_feedback(self):
        pass

    def update_round(self, round_num):
        self.current_round = round_num

    def update_trial(self, trial_num):
        self.current_trial = trial_num

    def update_result(self, result):
        self.current_result = result

    def get_round_data(self, round_num: int):
        """Get data for specific round."""
        if round_num < 0 or round_num >= self.n_rounds:
            raise ValueError(f"Round number must be between 0 and {self.n_rounds - 1}")
        return self.rounds[round_num]
    
    

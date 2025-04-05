import random
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
        pass

    def get_initial_prompt(self):
        try:
            return self.initial_prompt()
        except AttributeError:
            raise ValueError("Initial prompt not defined for this task.")

    def get_intermediate_prompt(self) -> str:
        try:
            return self.intermediate_prompt()
        except AttributeError:
            raise ValueError("Intermediate prompt not defined for this task.")

    def get_final_prompt(self) -> str:
        try:
            return self.final_prompt()
        except AttributeError:
            raise ValueError("Final prompt not defined for this task.")

    def process_choice(self) -> Optional[str]:
        pass

    def create_round_stats(self) -> Dict:
        # Create round stats
        return {
            'available_cues': self.available_cues,
            'choice': self.current_answer,
            'quadrant': self.quadrant,
            'result': self.current_result,
            'round_time': self.round_time,
            'thinking_time': self.thinking_time
        }


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

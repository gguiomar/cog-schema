import random
from typing import List, Dict, Optional
from tqdm import tqdm
import xml.etree.ElementTree as ET
from tasks.TaskGeneral import TaskGeneral
from util.util import *

class ClassicalConditioningTask(TaskGeneral):
    def __init__(self, n_rounds: int = 1, n_quadrants: int = 4, n_cues: int = 1):
        super().__init__(n_rounds, n_quadrants, n_cues)
        self.current_result = None
        self.available_cues = ["A"]
        # self.available_queues = ["A"]
        self.strings = ET.parse('tasks/ClassicalConditioningTask.xml')
        # self.reward_quadrant = random.choice(self.quadrants)
        self.correct_answer = "A"
        self.conditioned_stimulus = "A"
        self.received_reward = True
        self.rounds = self._generate_rounds()

    def _generate_rounds(self) -> List[List[Dict]]:
        """Generate rounds for classical conditioning task.
           Each round shows a single cue (the conditioned stimulus) always in RED."""
        rounds = []
        for r in range(self.n_rounds):
            if r < self.n_rounds - 1:
                round_str = [{
                    'name': self.conditioned_stimulus,
                    'color': 'RED',
                    'quadrant': 0  # since there's only one cue, assign quadrant 0
                }]
            else:
                # Last round: show reward or no reward
                if self.received_reward:
                    reward_str = "REWARD +100 POINTS"
                    round_str = [{
                    'name': reward_str,
                    'color': 'WHITE',
                    'quadrant': 0  # since there's only one cue, assign quadrant 0
                }]
                else:
                    reward_str = "PUNISHMENT -1000 POINTS"
                    round_str = [{
                    'name': reward_str,
                    'color': 'PURPLE',
                    'quadrant': 0  # since there's only one cue, assign quadrant 0
                }]
            rounds.append(round_str)
        return rounds
    
    def give_feedback(self) -> str:
        """Return round results including trial number."""
        result_text = self.current_result if self.current_result is not None else "Invalid choice"
        prompt = load_prompt_from_xml(self.strings, 'feedback_prompt')
        return prompt.format(
            current_trial=self.current_trial + 1,
            current_round=self.current_round + 1,
            available_cues=self.available_cues,
            current_answer=self.current_answer,
            result_text=result_text
        )
    
    def create_round_stats(self) -> Dict:
        # Create round stats
        return {
            'available_cues': self.available_cues,
            'choice': self.current_answer,
            'result': self.current_result,
            'round_time': self.round_time,
            'thinking_time': self.thinking_time
        }
    
    def get_final_prompt(self) -> str:
        try:
            prompt = load_prompt_from_xml(self.strings, 'final_prompt')
            return prompt.format(
                current_trial=self.current_trial + 1,
                current_round=self.current_round + 1
            )
        except AttributeError:
            raise ValueError("Final prompt not defined for this task.")

    def process_choice(self):
        round_data = self.get_round_data(self.current_round)
        result = None
        for cue in round_data:
            if cue['name'] == self.current_answer:
                result = cue['color']
                break
        # round_data = self.get_round_data(self.current_round)

        return result
    
    def get_round_data(self, round_num: int) -> List[Dict]:
        """Get data for specific round."""
        if round_num < 0 or round_num >= self.n_rounds:
            raise ValueError(f"Round number must be between 0 and {self.n_rounds - 1}")
        return self.rounds[round_num]

    def give_feedback(self) -> str:
        """Return round results including trial number."""
        result_text = self.current_result if self.current_result is not None else "Invalid choice"
        prompt = load_prompt_from_xml(self.strings, 'feedback_prompt')
        return prompt.format(
            current_trial=self.current_trial + 1,
            current_round=self.current_round + 1,
            available_cues=self.available_cues,
            current_answer=self.current_answer,
            result_text=result_text
        )
    
    def process_final_choice(self) -> bool:
        """Process the final choice and check if it's correct."""
        if self.current_answer == "A":
            self.received_reward = True
            return True
        return False
    
    def print_final_log(self):
        tqdm.write(f"LLM's final choice: {self.current_answer}")
        if self.current_answer == "A":
            tqdm.write(f"Result: REWARD +100 POINTS")
        else:
            tqdm.write(f"Result: INVALID")

    def update_result(self, result):
        self.current_result = result

    def give_final_feedback(self) -> str:
        """Return round results including trial number."""
        result_text = self.current_result if self.current_result is not None else "Invalid choice"
        prompt = load_prompt_from_xml(self.strings, 'feedback_prompt')
        return prompt.format(
            current_trial=self.current_trial + 1,
            current_round=self.current_round + 2,
            available_cues=self.available_cues,
            current_answer=self.current_answer,
            result_text=result_text
        )
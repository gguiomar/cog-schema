import random
from typing import List, Dict, Optional
from tqdm import tqdm
import xml.etree.ElementTree as ET
from tasks.NewTask import TaskGeneral
from util.util import *

class ClassicConditioning(TaskGeneral):
    def __init__(self, n_rounds: int = 1, n_quadrants: int = 4, n_cues: int = 1):
        super().__init__(n_rounds, n_quadrants, n_cues)

        self.reward_quadrant = random.choice(self.quadrants)
        self.correct_answer = self.reward_quadrant

        self.rounds = self._generate_rounds()

        self.available_queues = None

    def initial_prompt(self):
        prompt = (
            f"""
            Task:
            You will play a game with {self.n_rounds} rounds.
            In each round you'll see active cues (choosable):
            One cue has 90%% one color / 10%% other, others  have 50/50 color distribution.
            Possible cues are: A, B, C, D.
            Active cues disappear after random duration, at least one cue is active per round.
            Your task is to pick one of the available cues every round, by responding with just the letter and nothing else. Don't use markup or punctuation.
            Answer like this: "You choose: <letter>".
            After {self.n_rounds} rounds, identify the biased cue.
            Correct: +100 points, Wrong: -100 points.
            """
        )
        return prompt

    def intermediate_prompt(self) -> str:
        current_prompt = (
            f"""
            Trial {self.current_trial + 1}, Round {self.current_round + 1}: Available cues {self.available_cues}.
            Based on previous observations, choose one cue by responding with just the letter and nothing else.
            """
        )

        return current_prompt

    def final_prompt(self) -> str:
        """Build final prompt for a trial including conversation history."""
        prompt = (
                f"Trial {self.current_trial + 1}: Based on all observed colors, which quadrant (1"
                f"{', ' + ', '.join(str(i) for i in range(2, self.current_trial + 1))}"
                ") do you think had the highest ratio of RED? "
                "You choose:"
        )
        return prompt

import random
from typing import List, Dict, Optional
from tqdm import tqdm
import xml.etree.ElementTree as ET
from tasks.TaskGeneral import TaskGeneral
from util.util import *

class ClassicalConditioningTask(TaskGeneral):
    def __init__(self, n_rounds: int = 1, n_quadrants: int = 4, n_cues: int = 1):
        super().__init__(n_rounds, n_quadrants, n_cues)

        self.reward_quadrant = random.choice(self.quadrants)
        self.correct_answer = self.reward_quadrant

        self.rounds = self._generate_rounds()

        self.available_queues = None

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

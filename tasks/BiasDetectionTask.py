import random
from typing import List, Dict, Optional
from tqdm import tqdm
import xml.etree.ElementTree as ET
from tasks.TaskGeneral import TaskGeneral
from util.util import *

class BiasDetectionTask(TaskGeneral):
    def __init__(self, n_rounds: int = 1, n_quadrants: int = 4, n_cues: int = 1):
        super().__init__(n_rounds, n_quadrants, n_cues)

        self.biased_quadrant = random.choice(self.quadrants)
        self.correct_answer = self.biased_quadrant
        self.rounds = self._generate_rounds()

        self.available_cues = None

        self.strings = ET.parse('tasks/BiasDetectionTask.xml')

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

    def process_choice(self):
        round_data = self.get_round_data(self.current_round)
        result = None
        for cue in round_data:
            if cue['name'] == self.current_answer:
                result = cue['color']
                break

        self.quadrant = None
        round_data = self.get_round_data(self.current_round)

        # Find which quadrant the chosen cue belongs to
        if result is not None:  # Only try to identify quadrant if choice was valid
            for q in round_data:
                if q['name'].lower() == self.current_answer.lower():
                    self.quadrant = q['quadrant'] + 1  # +1 because quadrants are 0-indexed in code
                    break

        if self.verbose:
            if result:
                tqdm.write(f"Result: {result}")
                if self.quadrant:
                    tqdm.write(f"(cue {self.current_answer} was from Quadrant {self.quadrant})")
            else:
                tqdm.write("Invalid choice!")

        return result
    
    def give_final_feedback(self) -> str:
        """Return final feedback after all rounds."""
        prompt = load_prompt_from_xml(self.strings, 'final_feedback_prompt')
        feedback_text = ""
        if self.current_answer == self.letters[self.correct_answer]:
            feedback_text = load_prompt_from_xml(self.strings, 'feedback_correct')
        elif self.current_answer in self.letters:
            feedback_text = load_prompt_from_xml(self.strings, 'feedback_incorrect').format(biased_quadrant=self.letters[self.correct_answer])
        else:
            feedback_text = load_prompt_from_xml(self.strings, 'feedback_invalid')

        return prompt.format(
            current_trial=self.current_trial + 1,
            letters=self.letters,
            current_answer=self.current_answer,
            feedback=feedback_text,
            score=100 if self.current_answer == self.letters[self.correct_answer] else -100
        )
    
    def process_final_choice(self) -> bool:
        """Process the final choice and check if it's correct."""
        if self.current_answer == self.letters[self.correct_answer]:
            return True
        
        return False
    
    def print_final_log(self):
        tqdm.write(f"LLM's final choice: {self.current_answer}")
        tqdm.write(f"Correct quadrant: {self.letters[self.correct_answer]}")
        if self.current_answer == self.letters[self.correct_answer]:
            tqdm.write(f"Result: Correct")
        else:
            tqdm.write(f"Result: Incorrect")

    def _generate_rounds(self) -> List[Dict]:
        """Generate rounds for bias detection task."""
        while True:
            rounds = []
            for _ in range(self.n_rounds):
                round_cues = []
                for q in self.quadrants:
                    for cue in self.cue_map[q]:
                        if random.random() < 0.5:  # 50% chance cue is available
                            round_cues.append({
                                'name': cue,
                                'color': self._get_color_for_bias_detection(q),
                                'quadrant': q
                            })

                # Ensure at least one cue per round
                if not round_cues:
                    q = random.choice(self.quadrants)
                    cue = random.choice(self.cue_map[q])
                    round_cues.append({
                        'name': cue,
                        'color': self._get_color_for_bias_detection(q),
                        'quadrant': q
                    })
                rounds.append(round_cues)

            if self._validate_bias_detection_rounds(rounds):
                return rounds

    def _get_color_for_bias_detection(self, quadrant: int) -> str:
        """Determine color based on quadrant probability for bias detection."""
        if quadrant == self.biased_quadrant:
            return 'RED' if random.random() < 0.9 else 'GREEN'
        return random.choice(['RED', 'GREEN'])  # 50/50 distribution

    def _validate_bias_detection_rounds(self, rounds: List[List[Dict]]) -> bool:
        """Validate that the generated rounds create a solvable bias detection game."""
        color_counts = {q: {'RED': 0, 'GREEN': 0} for q in self.quadrants}

        # Count colors for each quadrant
        for round_data in rounds:
            for cue in round_data:
                q = cue['quadrant']
                color = cue['color']
                color_counts[q][color] += 1

        # Check ratios and conditions
        for q in self.quadrants:
            total = color_counts[q]['RED'] + color_counts[q]['GREEN']
            if total == 0:
                return False

            red_ratio = color_counts[q]['RED'] / total
            if q == self.biased_quadrant:
                if red_ratio < 0.8:  # Ensure biased quadrant has clear bias
                    return False
            elif not (0.35 <= red_ratio <= 0.65):  # Ensure other quadrants are balanced
                return False

        return True
    
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
    
    def get_final_prompt(self) -> str:
        try:
            prompt = load_prompt_from_xml(self.strings, 'final_prompt')
            return prompt.format(
                current_trial=self.current_trial + 1,
                letters=self.letters
            )
        except AttributeError:
            raise ValueError("Final prompt not defined for this task.")


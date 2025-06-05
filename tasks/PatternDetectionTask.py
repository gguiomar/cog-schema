import random
from typing import List, Dict, Optional
from tqdm import tqdm
import xml.etree.ElementTree as ET
from tasks.TaskGeneral import TaskGeneral
from util.util import *

class PatternDetectionTask(TaskGeneral):
    def __init__(self, n_rounds: int = 1, n_quadrants: int = 4, n_cues: int = 1, prompt_version: int = None):
        super().__init__(n_rounds, n_quadrants, n_cues, prompt_version)

        self.biased_quadrant = random.choice(self.quadrants)
        self.correct_answer = self.biased_quadrant
        self.pattern_quadrant = self.biased_quadrant
        self.pattern_cue = self.cue_map[self.pattern_quadrant][0]  # Use first cue from pattern quadrant
        self.rounds = self._generate_pattern_detection_rounds()

        self.available_cues = None
        self.strings = ET.parse('tasks/PatternDetectionTask.xml')

    def get_round_data(self, round_num: int) -> List[Dict]:
        """Get data for specific round."""
        if round_num < 0 or round_num >= self.n_rounds:
            raise ValueError(f"Round number must be between 0 and {self.n_rounds - 1}")
        return self.rounds[round_num]

    def give_feedback(self) -> str:
        """Return round results including trial number."""
        result_text = self.current_result if self.current_result is not None else "Invalid choice"
        prompt = load_prompt_from_xml(self.strings, 'feedback_prompt', self.prompt_version)
        return prompt.format(
            current_trial=self.current_trial + 1,
            current_round=self.current_round + 1,
            n_rounds=self.n_rounds,
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
        prompt = load_prompt_from_xml(self.strings, 'final_feedback_prompt', self.prompt_version)
        feedback_text = ""
        if self.current_answer == self.letters[self.correct_answer]:
            feedback_text = load_prompt_from_xml(self.strings, 'feedback_correct', self.prompt_version)
        elif self.current_answer in self.letters:
            feedback_text = load_prompt_from_xml(self.strings, 'feedback_incorrect', self.prompt_version).format(pattern_quadrant=self.letters[self.correct_answer])
        else:
            feedback_text = load_prompt_from_xml(self.strings, 'feedback_invalid', self.prompt_version)

        return prompt.format(
            current_trial=self.current_trial + 1,
            n_rounds=self.n_rounds,
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
            prompt = load_prompt_from_xml(self.strings, 'final_prompt', self.prompt_version)
            return prompt.format(
                current_trial=self.current_trial + 1,
                n_rounds=self.n_rounds,
                letters=self.letters
            )
        except AttributeError:
            raise ValueError("Final prompt not defined for this task.")

    def _generate_pattern_detection_rounds(self) -> List[List[Dict]]:
        """Generate rounds for pattern detection task."""
        rounds = []
        pattern_available = [random.random() < 0.7 for _ in range(self.n_rounds)]  # Randomly determine when pattern cue is available
        
        # Ensure pattern cue is available in enough rounds to be detectable
        min_pattern_rounds = max(3, self.n_rounds // 2)  # At least 3 rounds or half, whichever is greater
        if sum(pattern_available) < min_pattern_rounds:
            # Force pattern cue to be available in random rounds until we meet minimum
            unavailable_indices = [i for i, available in enumerate(pattern_available) if not available]
            to_add = min_pattern_rounds - sum(pattern_available)
            for i in random.sample(unavailable_indices, min(to_add, len(unavailable_indices))):
                pattern_available[i] = True
        
        # Prepare the alternating pattern
        start_color = random.choice(['RED', 'GREEN'])
        pattern_colors = []
        for i in range(self.n_rounds):
            if i % 2 == 0:
                pattern_colors.append(start_color)
            else:
                pattern_colors.append('GREEN' if start_color == 'RED' else 'RED')
        
        for round_idx in range(self.n_rounds):
            round_cues = []
            
            # Add pattern cue if available this round
            if pattern_available[round_idx]:
                round_cues.append({
                    'name': self.pattern_cue,
                    'color': pattern_colors[round_idx],
                    'quadrant': self.pattern_quadrant,
                    'is_pattern': True
                })
            
            # Add other random cues
            for q in self.quadrants:
                for cue in self.cue_map[q]:
                    if cue != self.pattern_cue and random.random() < 0.5:  # 50% chance other cues are available
                        round_cues.append({
                            'name': cue,
                            'color': random.choice(['RED', 'GREEN']),
                            'quadrant': q,
                            'is_pattern': False
                        })
            
            # Ensure at least one cue per round
            if not round_cues:
                q = random.choice(self.quadrants)
                available_cues = [c for c in self.cue_map[q] if c != self.pattern_cue]
                
                if available_cues:
                    cue = random.choice(available_cues)
                    round_cues.append({
                        'name': cue,
                        'color': random.choice(['RED', 'GREEN']),
                        'quadrant': q,
                        'is_pattern': False
                    })
                else:
                    # Force pattern cue if no other cues available
                    round_cues.append({
                        'name': self.pattern_cue,
                        'color': pattern_colors[round_idx],
                        'quadrant': self.pattern_quadrant,  # Now using the stored quadrant
                        'is_pattern': True
                    })
            
            rounds.append(round_cues)
        
        # Validate the pattern is detectable
        if self._validate_pattern_detection_rounds(rounds):
            return rounds
        else:
            # Try again if validation fails
            return self._generate_pattern_detection_rounds()

    def _validate_pattern_detection_rounds(self, rounds: List[List[Dict]]) -> bool:
        """Validate that the pattern is detectable in the generated rounds."""
        pattern_appearances = []
        
        # Collect all appearances of pattern cue
        for round_data in rounds:
            for cue in round_data:
                if cue['name'] == self.pattern_cue:
                    pattern_appearances.append(cue['color'])
        
        # Ensure pattern cue appears at least 3 times to be detectable
        if len(pattern_appearances) < 3:
            return False
        
        # Check if the pattern alternates as intended
        for i in range(1, len(pattern_appearances)):
            if pattern_appearances[i] == pattern_appearances[i-1]:
                # Pattern doesn't alternate
                return False
        return True
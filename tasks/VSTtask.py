# import random
# from typing import List, Dict, Optional, Tuple

# class VSTtask:
#     def __init__(self, n_rounds: int, n_quadrants: int = 2, n_cues: int = 1):
#         if not 2 <= n_quadrants <= 4:
#             raise ValueError("Number of quadrants must be between 2 and 4")
#         if n_cues < 1:
#             raise ValueError("Number of cues per quadrant must be at least 1")
            
#         self.n_rounds = n_rounds
#         self.n_quadrants = n_quadrants
#         self.n_cues = n_cues
#         self.current_round = 0
        
#         # Generate unique cue letters and quadrant mapping
#         self.letters = [chr(65 + i) for i in range(n_quadrants * n_cues)]
#         self.cue_map = {
#             q: self.letters[q*n_cues:(q+1)*n_cues]
#             for q in range(n_quadrants)
#         }
#         self.cue_to_quadrant = {
#             cue: q 
#             for q in self.cue_map 
#             for cue in self.cue_map[q]
#         }
        
#         self.quadrants = list(range(n_quadrants))
#         self.biased_quadrant = random.choice(self.quadrants)
#         self.biased_color = 'RED'  # Can be randomized to 'GREEN' if needed
#         self.pregen_rounds: List[List[Dict]] = []
        
#         # Generate and validate task until we get a solvable configuration
#         while True:
#             self._pregenerate_rounds()
#             if self._validate_task():
#                 break

#     def _pregenerate_rounds(self):
#         """Generate all rounds upfront for validation"""
#         self.pregen_rounds = []
#         for _ in range(self.n_rounds):
#             round_cues = []
            
#             # Generate active cues for this round
#             for q in self.quadrants:
#                 for cue in self.cue_map[q]:
#                     if random.random() < 0.5:
#                         color = self._get_cue_color(q)
#                         round_cues.append({
#                             'name': cue,
#                             'color': color,
#                             'quadrant': q,
#                             'rounds_remaining': random.randint(1, self.n_rounds)
#                         })
            
#             # Ensure at least one cue per round
#             if not round_cues:
#                 q = random.choice(self.quadrants)
#                 cue = random.choice(self.cue_map[q])
#                 round_cues.append({
#                     'name': cue,
#                     'color': self._get_cue_color(q),
#                     'quadrant': q,
#                     'rounds_remaining': random.randint(1, self.n_rounds)
#                 })
            
#             self.pregen_rounds.append(round_cues)

#     def _validate_task(self) -> bool:
#         """Check if the generated task is solvable through potential observations"""
#         color_counts = {q: {'RED': 0, 'GREEN': 0} for q in self.quadrants}
        
#         # Simulate perfect sampling of all cues in all rounds
#         for round_data in self.pregen_rounds:
#             for cue in round_data:
#                 q = cue['quadrant']
#                 color = cue['color']
#                 color_counts[q][color] += 1

#         # Calculate color ratios for each quadrant
#         ratios = {}
#         for q in self.quadrants:
#             total = color_counts[q]['RED'] + color_counts[q]['GREEN']
#             if total == 0:
#                 return False  # No samples from this quadrant
#             ratios[q] = {
#                 'RED': color_counts[q]['RED'] / total,
#                 'GREEN': color_counts[q]['GREEN'] / total
#             }

#         # Check biased quadrant meets 90/10 threshold
#         biased_ratio = ratios[self.biased_quadrant][self.biased_color]
#         if biased_ratio < 0.8:  # Allow some tolerance for small sample sizes
#             return False

#         # Check other quadrants are within 50/50 ± 15%
#         for q in self.quadrants:
#             if q == self.biased_quadrant:
#                 continue
#             red_ratio = ratios[q]['RED']
#             if not (0.35 <= red_ratio <= 0.65):
#                 return False

#         # Check biased quadrant has clearly highest target color count
#         biased_count = color_counts[self.biased_quadrant][self.biased_color]
#         other_counts = [
#             color_counts[q][self.biased_color] 
#             for q in self.quadrants if q != self.biased_quadrant
#         ]
#         return biased_count > max(other_counts, default=0)

#     def _get_cue_color(self, quadrant: int) -> str:
#         """Generate color based on quadrant probability"""
#         if quadrant == self.biased_quadrant:
#             return self.biased_color if random.random() < 0.9 else \
#                   'GREEN' if self.biased_color == 'RED' else 'RED'
#         return random.choice(['RED', 'GREEN'])

#     def generate_task_description(self) -> str:
#         quadrant_descs = [
#             f"Quadrant {q+1} with cues {', '.join(self.cue_map[q])}"
#             for q in self.quadrants
#         ]
#         return (
#             f"Visual Sampling Task ({self.n_quadrants} quadrants, {self.n_cues} cues/quadrant)\n"
#             f"You will play a game with {self.n_rounds} rounds.\n"
#             "One quadrant has a 90% bias to one color while the  other quadrants have 50/50 distribution\n"
#             "There's at least one active cue per round\n"
#             "Active cues disappear after random duration\n\n"
#             f"After {self.n_rounds} rounds, identify the biased quadrant by pressing {np.arange(1, self.n_cues+1, 1)}.\n"
#             "Correct: +100 points, Wrong: -100 points."
#         )

#     def play_round(self) -> None:
#         if self.current_round >= self.n_rounds:
#             raise ValueError("All rounds completed")

#         current_data = self.pregen_rounds[self.current_round]
#         available = ', '.join(q['name'] for q in current_data)
#         print(f"\nRound {self.current_round+1}/{self.n_rounds}")
#         print(f"Active cues: {available}")
#         print("You press <<")

#     def process_choice(self, choice: str) -> Optional[str]:
#         current_data = self.pregen_rounds[self.current_round]
#         for cue in current_data:
#             if cue['name'] == choice.upper():
#                 print(f"{choice.upper()} >> shows {cue['color']}")
#                 return cue['color']
#         print(f"Invalid choice: {choice}")
#         return None

#     def final_question(self) -> None:
#         valid = [str(q+1) for q in self.quadrants]
#         answer = input(f"\nWhich quadrant had the 90/10 distribution? ({'/'.join(valid)}) << ")
        
#         if answer in valid:
#             correct = str(self.biased_quadrant + 1)
#             if answer == correct:
#                 print(f"Correct! Quadrant {correct} (+100 points)")
#             else:
#                 print(f"Incorrect. Right answer was Quadrant {correct} (-100 points)")
#         else:
#             print("Invalid quadrant selection")

#     def run_game(self):
#         print(self.generate_task_description())
#         for self.current_round in range(self.n_rounds):
#             self.play_round()
#             while True:
#                 choice = input("Choose cue: ").strip().upper()
#                 if any(q['name'] == choice for q in self.pregen_rounds[self.current_round]):
#                     break
#                 print("Invalid choice, try again")
#             self.process_choice(choice)
#         self.final_question()


# if __name__ == "__main__":
#     # Example validation test
#     valid = False
#     while not valid:
#         task = VSTtask(n_rounds=10, n_quadrants=2, n_cues=1)
#         valid = task._validate_task()
    
#     task.run_game()

import random
from typing import List, Dict, Optional
import json
import os
from datetime import datetime
import numpy as np
import transformers
from tqdm import tqdm

class VSTtask:
    def __init__(self, n_rounds: int, n_quadrants: int = 2, n_cues: int = 1):
        """Initialize VST task with specified parameters."""
        if not 2 <= n_quadrants <= 4:
            raise ValueError("Number of quadrants must be between 2 and 4")
        if n_cues < 1:
            raise ValueError("Number of cues per quadrant must be at least 1")
            
        self.n_rounds = n_rounds
        self.n_quadrants = n_quadrants
        self.n_cues = n_cues
        self.current_round = 0
        
        # Setup quadrants and cues
        self.letters = [chr(65 + i) for i in range(n_quadrants * n_cues)]
        self.cue_map = {
            q: self.letters[q*n_cues:(q+1)*n_cues]
            for q in range(n_quadrants)
        }
        
        # Select biased quadrant and generate rounds
        self.quadrants = list(range(n_quadrants))
        self.biased_quadrant = random.choice(self.quadrants)
        self.rounds = self._generate_rounds()
        
    def _get_color(self, quadrant: int) -> str:
        """Determine color based on quadrant probability."""
        if quadrant == self.biased_quadrant:
            return 'RED' if random.random() < 0.9 else 'GREEN'
        return random.choice(['RED', 'GREEN'])
        
    def _generate_rounds(self) -> List[List[Dict]]:
        """Generate all rounds with cue colors and validations."""
        while True:
            rounds = []
            for _ in range(self.n_rounds):
                round_cues = []
                for q in self.quadrants:
                    for cue in self.cue_map[q]:
                        if random.random() < 0.5:
                            round_cues.append({
                                'name': cue,
                                'color': self._get_color(q),
                                'quadrant': q
                            })
                
                # Ensure at least one cue per round
                if not round_cues:
                    q = random.choice(self.quadrants)
                    cue = random.choice(self.cue_map[q])
                    round_cues.append({
                        'name': cue,
                        'color': self._get_color(q),
                        'quadrant': q
                    })
                rounds.append(round_cues)
                
            if self._validate_rounds(rounds):
                return rounds
    
    def _validate_rounds(self, rounds: List[List[Dict]]) -> bool:
        """Validate that the generated rounds create a solvable game."""
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
                if red_ratio < 0.8:
                    return False
            elif not (0.35 <= red_ratio <= 0.65):
                return False
                
        return True
    
    def get_round_data(self, round_num: int) -> List[Dict]:
        """Get data for specific round."""
        return self.rounds[round_num]
    
    def get_task_description(self) -> str:
        """Generate task description."""
        return (
            f"You will play a game with {self.n_rounds} rounds.\n"
            "In each round you'll see active cues (chooseable):\n" +
            "One quadrant has 90% one color/10% other\n"
            "Other quadrants have 50/50 color distribution\n"
            "At least one cue active per round\n"
            "Active cues disappear after random duration\n\n"
            f"After {self.n_rounds} rounds, identify the biased quadrant.\n"
            "Correct: +100 points, Wrong: -100 points."
        )
    
    def process_choice(self, choice: str, round_data: List[Dict]) -> Optional[str]:
        """Process choice and return color if valid."""
        for cue in round_data:
            if cue['name'] == choice:
                return cue['color']
        return None
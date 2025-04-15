import random
from typing import List, Dict, Optional
from tqdm import tqdm
import xml.etree.ElementTree as ET
from tasks.TaskGeneral import TaskGeneral
from util.util import *

class PatternDetectionTask(TaskGeneral):
    def __init__(self, n_rounds: int = 1, n_quadrants: int = 4, n_cues: int = 1):
        super().__init__(n_rounds, n_quadrants, n_cues)

        self.biased_quadrant = random.choice(self.quadrants)
        self.correct_answer = self.biased_quadrant

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
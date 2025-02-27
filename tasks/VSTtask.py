import random
from typing import List, Dict, Optional
import numpy as np

class VSTtask:
    # Task type constants
    TASK_BIAS_DETECTION = "bias_detection"
    TASK_PATTERN_DETECTION = "pattern_detection"
    TASK_CONDITIONAL_PROBABILITY = "conditional_probability"
    
    def __init__(self, n_rounds: int, n_quadrants: int = 2, n_cues: int = 1, 
                 task_type: str = TASK_BIAS_DETECTION):
        """Initialize VST task with specified parameters."""
        if not 2 <= n_quadrants <= 4:
            raise ValueError("Number of quadrants must be between 2 and 4")
        if n_cues < 1:
            raise ValueError("Number of cues per quadrant must be at least 1")
        
        # Validate task type
        valid_tasks = [self.TASK_BIAS_DETECTION, self.TASK_PATTERN_DETECTION, 
                      self.TASK_CONDITIONAL_PROBABILITY]
        if task_type not in valid_tasks:
            raise ValueError(f"Invalid task type. Must be one of: {', '.join(valid_tasks)}")
            
        self.task_type = task_type
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
        
        # Default values that will be set differently based on task type
        self.quadrants = list(range(n_quadrants))
        self.biased_quadrant = None
        self.pattern_cue = None
        self.pattern_quadrant = None  # Store the quadrant of the pattern cue
        self.conditional_cue_trigger = None
        self.conditional_cue_response = None
        
        # Task-specific setup
        if task_type == self.TASK_BIAS_DETECTION:
            self.biased_quadrant = random.choice(self.quadrants)
        elif task_type == self.TASK_PATTERN_DETECTION:
            # Choose a random cue to show the pattern
            self.pattern_cue = random.choice(self.letters)
            # Find which quadrant the pattern cue belongs to
            for q, cues in self.cue_map.items():
                if self.pattern_cue in cues:
                    self.pattern_quadrant = q
                    break
        elif task_type == self.TASK_CONDITIONAL_PROBABILITY:
            # Choose two random different cues for the conditional relationship
            cues = random.sample(self.letters, 2)
            self.conditional_cue_trigger = cues[0]  # This cue triggers
            self.conditional_cue_response = cues[1]  # This cue responds
        
        self.rounds = self._generate_rounds()

    def _generate_rounds(self) -> List[List[Dict]]:
        """Generate all rounds with cue colors based on task type."""
        if self.task_type == self.TASK_BIAS_DETECTION:
            return self._generate_bias_detection_rounds()
        elif self.task_type == self.TASK_PATTERN_DETECTION:
            return self._generate_pattern_detection_rounds()
        elif self.task_type == self.TASK_CONDITIONAL_PROBABILITY:
            return self._generate_conditional_probability_rounds()
        else:
            raise ValueError(f"Unsupported task type: {self.task_type}")

    def _generate_bias_detection_rounds(self) -> List[List[Dict]]:
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

    def _generate_conditional_probability_rounds(self) -> List[List[Dict]]:
        """Generate rounds for conditional probability task."""
        rounds = []
        previous_trigger_color = None
        
        # Find quadrants for trigger and response cues
        trigger_quadrant = None
        response_quadrant = None
        for q, cues in self.cue_map.items():
            if self.conditional_cue_trigger in cues:
                trigger_quadrant = q
            if self.conditional_cue_response in cues:
                response_quadrant = q
        
        for _ in range(self.n_rounds):
            round_cues = []
            
            # Determine trigger cue first
            trigger_available = random.random() < 0.7  # 70% chance trigger cue is available
            
            if trigger_available:
                trigger_color = random.choice(['RED', 'BLUE'])
                round_cues.append({
                    'name': self.conditional_cue_trigger,
                    'color': trigger_color,
                    'quadrant': trigger_quadrant,
                    'is_trigger': True
                })
                previous_trigger_color = trigger_color
            
            # Determine response cue
            response_available = random.random() < 0.7  # 70% chance response cue is available
            
            if response_available:
                # If previous trigger was BLUE, high chance of RED
                if previous_trigger_color == 'BLUE':
                    response_color = 'RED' if random.random() < 0.8 else 'GREEN'
                else:
                    # Otherwise random color
                    response_color = random.choice(['RED', 'GREEN'])
                
                round_cues.append({
                    'name': self.conditional_cue_response,
                    'color': response_color,
                    'quadrant': response_quadrant,
                    'is_response': True
                })
            
            # Add other random cues
            for q in self.quadrants:
                for cue in self.cue_map[q]:
                    if (cue != self.conditional_cue_trigger and 
                        cue != self.conditional_cue_response and 
                        random.random() < 0.5):  # 50% chance other cues are available
                        round_cues.append({
                            'name': cue,
                            'color': random.choice(['RED', 'GREEN', 'BLUE']),
                            'quadrant': q,
                            'is_trigger': False,
                            'is_response': False
                        })
            
            # Ensure at least one cue per round
            if not round_cues:
                q = random.choice(self.quadrants)
                available_cues = [c for c in self.cue_map[q] 
                                if c != self.conditional_cue_trigger 
                                and c != self.conditional_cue_response]
                if available_cues:
                    cue = random.choice(available_cues)
                    round_cues.append({
                        'name': cue,
                        'color': random.choice(['RED', 'GREEN', 'BLUE']),
                        'quadrant': q,
                        'is_trigger': False,
                        'is_response': False
                    })
                else:
                    # If no other cues, force one of the special cues
                    cue = random.choice([self.conditional_cue_trigger, self.conditional_cue_response])
                    is_trigger = cue == self.conditional_cue_trigger
                    if is_trigger:
                        color = random.choice(['RED', 'BLUE'])
                        previous_trigger_color = color
                        quadrant = trigger_quadrant
                    else:
                        if previous_trigger_color == 'BLUE':
                            color = 'RED' if random.random() < 0.8 else 'GREEN'
                        else:
                            color = random.choice(['RED', 'GREEN'])
                        quadrant = response_quadrant
                    
                    round_cues.append({
                        'name': cue,
                        'color': color,
                        'quadrant': quadrant,
                        'is_trigger': is_trigger,
                        'is_response': not is_trigger
                    })
            
            rounds.append(round_cues)
        
        # Validate conditional probability is detectable
        if self._validate_conditional_probability_rounds(rounds):
            return rounds
        else:
            # Try again if validation fails
            return self._generate_conditional_probability_rounds()

    def _validate_conditional_probability_rounds(self, rounds: List[List[Dict]]) -> bool:
        """Validate that the conditional probability is detectable in the generated rounds."""
        # Track sequence of trigger followed by response
        trigger_blue_count = 0
        response_after_blue_count = 0
        response_red_after_blue_count = 0
        
        # Also track non-conditional appearance of red in response cue
        total_response_appearances = 0
        total_response_red = 0
        
        for i in range(len(rounds) - 1):  # Look at pairs of rounds
            # Check if trigger was blue in current round
            trigger_was_blue = False
            for cue in rounds[i]:
                if cue['name'] == self.conditional_cue_trigger and cue['color'] == 'BLUE':
                    trigger_was_blue = True
                    trigger_blue_count += 1
                    break
            
            # Check response color in next round
            if trigger_was_blue:
                for cue in rounds[i+1]:
                    if cue['name'] == self.conditional_cue_response:
                        response_after_blue_count += 1
                        if cue['color'] == 'RED':
                            response_red_after_blue_count += 1
                        break
            
            # Count all response cue appearances and colors
            for cue in rounds[i]:
                if cue['name'] == self.conditional_cue_response:
                    total_response_appearances += 1
                    if cue['color'] == 'RED':
                        total_response_red += 1
        
        # Check final round for statistics
        for cue in rounds[-1]:
            if cue['name'] == self.conditional_cue_response:
                total_response_appearances += 1
                if cue['color'] == 'RED':
                    total_response_red += 1
        
        # Validation criteria:
        # 1. Trigger must show blue at least 3 times
        if trigger_blue_count < 3:
            return False
        
        # 2. Response must appear after blue trigger at least 3 times
        if response_after_blue_count < 3:
            return False
        
        # 3. Response must be red after blue trigger at least 70% of the time
        if response_after_blue_count > 0:
            red_after_blue_ratio = response_red_after_blue_count / response_after_blue_count
            if red_after_blue_ratio < 0.7:
                return False
        
        # 4. Overall red ratio in response should be lower than red-after-blue ratio
        if total_response_appearances > 0 and response_after_blue_count > 0:
            overall_red_ratio = total_response_red / total_response_appearances
            red_after_blue_ratio = response_red_after_blue_count / response_after_blue_count
            if overall_red_ratio >= red_after_blue_ratio:
                return False
        
        return True

    def get_round_data(self, round_num: int) -> List[Dict]:
        """Get data for specific round."""
        if round_num < 0 or round_num >= self.n_rounds:
            raise ValueError(f"Round number must be between 0 and {self.n_rounds - 1}")
        return self.rounds[round_num]
    
    def get_task_description(self) -> str:
        """Generate task description based on task type."""
        if self.task_type == self.TASK_BIAS_DETECTION:
            return self._get_bias_detection_description()
        elif self.task_type == self.TASK_PATTERN_DETECTION:
            return self._get_pattern_detection_description()
        elif self.task_type == self.TASK_CONDITIONAL_PROBABILITY:
            return self._get_conditional_probability_description()
        else:
            raise ValueError(f"Unsupported task type: {self.task_type}")

    def _get_bias_detection_description(self) -> str:
        """Get description for bias detection task."""
        return (
            f"You will play a game with {self.n_rounds} rounds.\n"
            "In each round you'll see active cues (letters):\n" +
            "One quadrant has 90% RED / 10% GREEN\n"
            "Other quadrants have 50% RED / 50% GREEN distribution\n"
            "At least one cue active per round\n"
            "Active cues appear and disappear randomly\n\n"
            f"After {self.n_rounds} rounds, identify the quadrant with the highest ratio of RED.\n"
            "Correct: +100 points, Wrong: -100 points."
        )

    def _get_pattern_detection_description(self) -> str:
        """Get description for pattern detection task."""
        return (
            f"You will play a game with {self.n_rounds} rounds.\n"
            "In each round you'll see active cues (letters):\n" +
            "One cue shows an alternating color pattern (RED, GREEN, RED, GREEN...)\n"
            "Other cues show random colors\n"
            "Cues appear and disappear randomly each round\n"
            "At least one cue active per round\n\n"
            f"After {self.n_rounds} rounds, identify which cue showed the alternating pattern.\n"
            "Correct: +100 points, Wrong: -100 points."
        )

    def _get_conditional_probability_description(self) -> str:
        """Get description for conditional probability task."""
        return (
            f"You will play a game with {self.n_rounds} rounds.\n"
            "In each round you'll see active cues (letters):\n" +
            "There's a relationship between cues' colors across rounds\n"
            "One cue is more likely to be RED after another cue shows BLUE\n"
            "Cues can show RED, GREEN, or BLUE colors\n"
            "Cues appear and disappear randomly each round\n"
            "At least one cue active per round\n\n"
            f"After {self.n_rounds} rounds, identify which cue was most likely to be RED after another cue showed BLUE.\n"
            "Correct: +100 points, Wrong: -100 points."
        )
    
    def process_choice(self, choice: str, round_data: List[Dict]) -> Optional[str]:
        """Process choice and return color if valid."""
        for cue in round_data:
            if cue['name'] == choice:
                return cue['color']
        return None
    
    def get_correct_answer(self) -> str:
        """Get the correct answer based on task type."""
        if self.task_type == self.TASK_BIAS_DETECTION:
            return str(self.biased_quadrant + 1)  # +1 because quadrants are 0-indexed in code
        elif self.task_type == self.TASK_PATTERN_DETECTION:
            return self.pattern_cue
        elif self.task_type == self.TASK_CONDITIONAL_PROBABILITY:
            return self.conditional_cue_response
        else:
            raise ValueError(f"Unsupported task type: {self.task_type}")
    
    def get_final_question(self) -> str:
        """Get the final question based on task type."""
        if self.task_type == self.TASK_BIAS_DETECTION:
            return (f"Which quadrant (1-{self.n_quadrants}) had the highest ratio of RED?")
        elif self.task_type == self.TASK_PATTERN_DETECTION:
            return ("Which cue showed the alternating pattern?")
        elif self.task_type == self.TASK_CONDITIONAL_PROBABILITY:
            return ("Which cue was most likely to be RED after another cue showed BLUE?")
        else:
            raise ValueError(f"Unsupported task type: {self.task_type}")
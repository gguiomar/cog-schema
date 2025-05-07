import random
from typing import List, Dict, Optional
import numpy as np

class VSTtask:
    # Task type constants
    TASK_BIAS_DETECTION = "bias_detection"
    TASK_PATTERN_DETECTION = "pattern_detection"
    TASK_CONDITIONAL_PROBABILITY = "conditional_probability"
    TASK_CLASSICAL_CONDITIONING = "classical_conditioning"
    
    def __init__(self, n_rounds: int, n_quadrants: int = 2, n_cues: int = 1, 
                 task_type: str = TASK_BIAS_DETECTION):
        """Initialize VST task with specified parameters."""
        if task_type == self.TASK_CLASSICAL_CONDITIONING:
            if n_quadrants != 1:
                raise ValueError("For classical conditioning, number of quadrants must be 1")
            n_cues = 1  # force one cue as well
        else:
            if not 2 <= n_quadrants <= 4:
                raise ValueError("Number of quadrants must be between 2 and 4")
        
        # Validate task type
        valid_tasks = [self.TASK_BIAS_DETECTION, self.TASK_PATTERN_DETECTION, 
                      self.TASK_CONDITIONAL_PROBABILITY, self.TASK_CLASSICAL_CONDITIONING]

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
        self.received_reward = None
        
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
            cues_special = random.sample(self.letters, 2)
            self.conditional_cue_uniform = cues_special[0]  # Uniform: displays colors uniformly
            self.conditional_cue_biased = cues_special[1]   # Biased: if previous color was RED, likely BLUE; otherwise uniform
            
            for q, cues in self.cue_map.items():
                if self.conditional_cue_uniform in cues:
                    self.conditional_quadrant_uniform = q
                if self.conditional_cue_biased in cues:
                    self.conditional_quadrant_biased = q

        elif task_type == self.TASK_CLASSICAL_CONDITIONING:
            self.n_cues = 1
            self.conditioned_stimulus = self.letters[0]
            self.reward_probability = 0.9
            self.rounds = self._generate_classical_conditioning_rounds()
            self.received_reward = random.random() < self.reward_probability
        else:
            raise ValueError(f"Invalid task type. Must be one of: {self.TASK_BIAS_DETECTION}, "
                             f"{self.TASK_PATTERN_DETECTION}, {self.TASK_CONDITIONAL_PROBABILITY}, "
                             f"{self.TASK_CLASSICAL_CONDITIONING}")        

        
        self.rounds = self._generate_rounds()

    def _generate_rounds(self) -> List[List[Dict]]:
        """Generate all rounds with cue colors based on task type."""
        if self.task_type == self.TASK_BIAS_DETECTION:
            return self._generate_bias_detection_rounds()
        elif self.task_type == self.TASK_PATTERN_DETECTION:
            return self._generate_pattern_detection_rounds()
        elif self.task_type == self.TASK_CONDITIONAL_PROBABILITY:
            return self._generate_conditional_probability_rounds()
        elif self.task_type == self.TASK_CLASSICAL_CONDITIONING:
            return self._generate_classical_conditioning_rounds()
        else:
            raise ValueError(f"Unsupported task type: {self.task_type}")

    def _generate_classical_conditioning_rounds(self) -> List[List[Dict]]:
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
        """Generate rounds for conditional probability task with new structure."""
        rounds = []
        # Determine quadrants for special cues
        uniform_quadrant = None
        biased_quadrant = None
        for q, cues in self.cue_map.items():
            if self.conditional_cue_uniform in cues:
                uniform_quadrant = q
            if self.conditional_cue_biased in cues:
                biased_quadrant = q
        
        prev_biased = None
        for _ in range(self.n_rounds):
            round_cues = []
            # Always add the special cues.
            # Uniform cue: always uniform.
            color_uniform = random.choice(['RED', 'GREEN', 'BLUE'])
            round_cues.append({
                'name': self.conditional_cue_uniform,
                'color': color_uniform,
                'quadrant': uniform_quadrant,
                'is_special': True,
                'special_type': 'uniform'
            })
            # Biased cue: if previous color was RED, bias toward BLUE.
            if prev_biased == 'RED':
                r = random.random()
                if r < 0.8:
                    color_biased = 'BLUE'
                elif r < 0.9:
                    color_biased = 'RED'
                else:
                    color_biased = 'GREEN'
            else:
                color_biased = random.choice(['RED', 'GREEN', 'BLUE'])
            round_cues.append({
                'name': self.conditional_cue_biased,
                'color': color_biased,
                'quadrant': biased_quadrant,
                'is_special': True,
                'special_type': 'biased'
            })
            prev_biased = color_biased
            
            # Add extra (blocked) cues if n_cues > 2.
            if self.n_cues > 2:
                for q in self.quadrants:
                    for cue in self.cue_map[q]:
                        if cue not in [self.conditional_cue_uniform, self.conditional_cue_biased]:
                            if random.random() < 0.5:
                                round_cues.append({
                                    'name': cue,
                                    'color': random.choice(['RED', 'GREEN', 'BLUE']),
                                    'quadrant': q,
                                    'is_special': False
                                })
            rounds.append(round_cues)
        
        if self._validate_conditional_probability_rounds(rounds):
            return rounds
        else:
            return self._generate_conditional_probability_rounds()


    def _validate_conditional_probability_rounds(self, rounds: List[List[Dict]]) -> bool:
        """Validate that the biased cue shows a high chance of BLUE following RED."""
        biased_blue_after_red = 0
        biased_after_red_count = 0
        for i in range(1, len(rounds)):
            prev_biased_color = None
            curr_biased_color = None
            for cue in rounds[i-1]:
                if cue.get('is_special') and cue.get('special_type') == 'biased':
                    prev_biased_color = cue['color']
            for cue in rounds[i]:
                if cue.get('is_special') and cue.get('special_type') == 'biased':
                    curr_biased_color = cue['color']
            if prev_biased_color == 'RED':
                biased_after_red_count += 1
                if curr_biased_color == 'BLUE':
                    biased_blue_after_red += 1
        if biased_after_red_count > 0:
            ratio = biased_blue_after_red / biased_after_red_count
            if ratio < 0.7:
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
        elif self.task_type == self.TASK_CLASSICAL_CONDITIONING:
            return self._get_classical_conditioning_description()
        else:
            raise ValueError(f"Unsupported task type: {self.task_type}")

    def _get_classical_conditioning_description(self) -> str:
        """Get description for classical conditioning task."""
        return (
            f"You will play a classical conditioning task with {self.n_rounds} rounds.\n"
            "In each round a single cue (the conditioned stimulus) will appear in RED.\n"
            "After all rounds, a reward will be delivered with a 90% chance (and no reward with a 10% chance).\n"
            "Your task is to report whether you received a reward.\n"
            "Correct: +100 points if correct, -100 points if incorrect."
        )


    def _get_bias_detection_description(self) -> str:
        """Get description for bias detection task."""
        return (
            # f"""
            # Task:
            # You will play a game of {self.n_rounds} rounds.
            # In each round you'll see active cues drawn from {{A, B, C, D}}.  
            # One cue is biased (90% one color, 10% the other); all others are 50/50.  
            # Active cues disappear after a random time; at least one is always present.  

            # Your job:  
            # 1. In each round, pick *one* available cue by responding with exactly one uppercase letter (A, B, C or D), and *nothing else*.  
            # 2. After 4 rounds, identify the biased cue by again returning exactly its single uppercase letter.  

            # Let's begin.
            # """
            # f"You will play a game with {self.n_rounds} rounds.\n"
            # "In each round you'll see active cues (chooseable):\n" +
            # "One quadrant has 90% one color/10% other\n"
            # "Other quadrants have 50/50 color distribution\n"
            # "At least one cue active per round\n"
            # f"After {self.n_rounds} rounds, identify the biased quadrant.\n"
            # "Correct: +100 points, Wrong: -100 points."
            f"You will play a game with {self.n_rounds} rounds."
            "Each round, you will see a set of active cues labeled A, B, C, or D."
            "One of the four quadrants is biased: it shows 90% of one color and 10% of the other."
            "The other quadrants are unbiased: they show a 50/50 color split."
            "At least one cue is active per round."
            "Your task: figure out which quadrant is biased."
            "At the end of 6 rounds, you must make a single guess."
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
        """Get description for conditional probability task with new structure."""
        return (
            f"You will play a game with {self.n_rounds} rounds.\n"
            "In each round you'll see active cues (letters):\n"
            "Two special cues are always present:\n"
            " - One cue displays a color chosen uniformly at random from RED, GREEN, or BLUE.\n"
            " - The other cue is biased: if its previous color was RED, it is highly likely (80%) to be BLUE next; otherwise, its color is chosen uniformly.\n"
            "Additional blocked cues may appear if more than 2 cues per quadrant are configured.\n\n"
            f"After {self.n_rounds} rounds, identify the cue that exhibited the biased behavior.\n"
            "Correct: +100 points, Wrong: -100 points."
        )

    
    def process_choice(self, choice: str, round_data: List[Dict]) -> Optional[str]:
        """Process choice and return color if valid."""
        for cue in round_data:
            if cue['name'] == choice:
                return cue['color']
        return None
    
    def get_final_question(self) -> str:
        """Get the final question based on task type."""
        if self.task_type == self.TASK_BIAS_DETECTION:
            if self.n_cues == 1:
                return f"Which cue showed the bias in RED appearance?"
            else:
                return f"Which quadrant (1-{self.n_quadrants}) had the highest ratio of RED?"
        elif self.task_type == self.TASK_PATTERN_DETECTION:
            if self.n_cues == 1:
                return f"Which cue showed the alternating pattern?"
            else:
                return f"Which quadrant (1-{self.n_quadrants}) contained the cue with the alternating pattern?"
        elif self.task_type == self.TASK_CONDITIONAL_PROBABILITY:
            if self.n_cues == 1:
                return f"Which cue exhibited the biased behavior?"
            else:
                return f"Which quadrant (1-{self.n_quadrants}) contained the cue that was most likely to be RED after a cue showed BLUE?"
        elif self.task_type == self.TASK_CLASSICAL_CONDITIONING:
            return "Did you receive a reward? Respond with Y or N."
        else:
            raise ValueError(f"Unsupported task type: {self.task_type}")


    def get_correct_answer(self) -> str:
        """Get the correct answer based on task type."""
        if self.task_type == self.TASK_BIAS_DETECTION:
            if self.n_cues == 1:
                return self.biased_cue
            else:
                return str(self.biased_quadrant + 1)
        elif self.task_type == self.TASK_PATTERN_DETECTION:
            if self.n_cues == 1:
                return self.pattern_cue
            else:
                return str(self.pattern_quadrant + 1)
        elif self.task_type == self.TASK_CONDITIONAL_PROBABILITY:
            if self.n_cues == 1:
                return self.conditional_cue_biased
            else:
                return str(self.conditional_quadrant_biased + 1)
        elif self.task_type == self.TASK_CLASSICAL_CONDITIONING:
            return "Y" if self.received_reward else "N"
        else:
            raise ValueError(f"Unsupported task type: {self.task_type}")



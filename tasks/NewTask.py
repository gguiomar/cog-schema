import random
from typing import List, Dict, Optional
import numpy as np
from tqdm import tqdm


class VSTtask_general:
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

    def process_choice(self) -> Optional[str]: # This only holds for the bias task, fix!
        # Choice processing is task-specific
        self.task_choice_processing()

    def create_round_stats(self) -> Dict:
        # Create round stats
        round_stats = {
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
        # Updating of conversation history wth feedback is task-specific
        self.give_task_feedback()

    def update_round(self, round_num):
        self.current_round = round_num

    def update_trial(self, trial_num):
        self.current_trial = trial_num

    def update_result(self, result):
        self.current_result = result


class BiasDetectionTask(VSTtask_general):
    def __init__(self, n_rounds: int = 1, n_quadrants: int = 4, n_cues: int = 1):
        super().__init__(n_rounds, n_quadrants, n_cues)

        self.biased_quadrant = random.choice(self.quadrants)
        self.correct_answer = self.biased_quadrant
        self.rounds = self._generate_rounds()

        self.available_cues = None
        self.conversation_history = None

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
        round_data = self.get_round_data(self.current_round)
        available_cues = [q['name'] for q in round_data]

        # Build and show prompt with accumulated history
        self.available_cues = ', '.join(available_cues)

        current_prompt = (
            f"""
            Trial {self.current_trial + 1}, Round {self.current_round + 1}: Available cues {self.available_cues}.
            Based on previous observations, choose one cue by responding with just the letter and nothing else.
            """
        )

        return self.conversation_history + current_prompt

    def final_prompt(self) -> str:
        """Build final prompt for a trial including conversation history."""
        prompt = (
                self.conversation_history +
                f"Trial {self.current_trial + 1}: Based on all observed colors, which cue"
                f"{self.letters}"
                ") do you think had the highest ratio of RED? "
                "You choose:"
        )
        return prompt

    def get_round_data(self, round_num: int) -> List[Dict]:
        """Get data for specific round."""
        if round_num < 0 or round_num >= self.n_rounds:
            raise ValueError(f"Round number must be between 0 and {self.n_rounds - 1}")
        return self.rounds[round_num]

    def give_task_feedback(self):
        """Update conversation history with round results including trial number."""
        result_text = self.current_result if self.current_result is not None else "Invalid choice"
        round_text = (
            f"Trial {self.current_trial + 1}, Round {self.current_round + 1}: Available cues {self.available_cues}. "
            f"You chose {self.current_answer} and saw {result_text}.\n"
        )
        self.conversation_history += round_text

    def task_choice_processing(self):
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

class ClassicConditioning(VSTtask_general):
    def __init__(self, n_rounds: int = 1, n_quadrants: int = 4, n_cues: int = 1):
        super().__init__(n_rounds, n_quadrants, n_cues)

        self.reward_quadrant = random.choice(self.quadrants)
        self.correct_answer = self.reward_quadrant

        self.rounds = self._generate_rounds()

        self.available_queues = None
        self.conversation_history = None

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

        return self.conversation_history + current_prompt

    def final_prompt(self) -> str:
        """Build final prompt for a trial including conversation history."""
        prompt = (
                self.conversation_history +
                f"Trial {self.current_trial + 1}: Based on all observed colors, which quadrant (1"
                f"{', ' + ', '.join(str(i) for i in range(2, self.current_trial + 1))}"
                ") do you think had the highest ratio of RED? "
                "You choose:"
        )
        return prompt
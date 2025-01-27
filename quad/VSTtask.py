import random
from typing import List, Dict, Optional, Tuple

class VSTtask:
    def __init__(self, n_rounds: int, n_quadrants: int = 2, n_queues: int = 1):
        if not 2 <= n_quadrants <= 4:
            raise ValueError("Number of quadrants must be between 2 and 4")
        if n_queues < 1:
            raise ValueError("Number of queues per quadrant must be at least 1")
            
        self.n_rounds = n_rounds
        self.n_quadrants = n_quadrants
        self.n_queues = n_queues
        self.current_round = 0
        
        # Generate unique queue letters and quadrant mapping
        self.letters = [chr(65 + i) for i in range(n_quadrants * n_queues)]
        self.queue_map = {
            q: self.letters[q*n_queues:(q+1)*n_queues]
            for q in range(n_quadrants)
        }
        self.queue_to_quadrant = {
            queue: q 
            for q in self.queue_map 
            for queue in self.queue_map[q]
        }
        
        self.quadrants = list(range(n_quadrants))
        self.biased_quadrant = random.choice(self.quadrants)
        self.active_queues: List[Dict] = []

    def generate_task_description(self) -> str:
        quadrant_descs = [
            f"Quadrant {q+1} with queues {', '.join(self.queue_map[q])}"
            for q in self.quadrants
        ]
        return (
            f"Visual Sampling Task ({self.n_quadrants} quadrants, {self.n_queues} queues/quadrant)\n"
            f"You will play a game with {self.n_rounds} rounds.\n"
            "In each round you'll see active queues (chooseable) and inactive queues (black):\n" +
            '\n'.join(quadrant_descs) + "\n\n"
            "Key mechanics:\n"
            "1. One quadrant has 90% one color/10% other\n"
            "2. Other quadrants have 50/50 color distribution\n"
            "3. At least one queue active per round\n"
            "4. Active queues disappear after random duration\n\n"
            f"After {self.n_rounds} rounds, identify the biased quadrant.\n"
            "Correct: +100 points, Wrong: -100 points."
        )

    def _get_queue_color(self, quadrant: int) -> Tuple[str, str]:
        """Return (actual color, initial state) for a queue"""
        if quadrant == self.biased_quadrant:
            color = 'RED' if random.random() < 0.9 else 'GREEN'
        else:
            color = random.choice(['RED', 'GREEN'])
        return color, 'black'

    def _ensure_active_queues(self):
        """Guarantee at least one active queue per round"""
        if not self.active_queues:
            forced_queue = random.choice(self.letters)
            q = self.queue_to_quadrant[forced_queue]
            color, state = self._get_queue_color(q)
            self.active_queues.append({
                'name': forced_queue,
                'color': color,
                'state': state,
                'quadrant': q,
                'rounds_remaining': random.randint(1, self.n_rounds - self.current_round + 1)
            })

    def play_round(self) -> None:
        self.current_round += 1
        self.active_queues = []
        
        # Generate initial active queues
        for q in self.quadrants:
            for queue in self.queue_map[q]:
                if random.random() < 0.5:  # Base activation chance
                    color, state = self._get_queue_color(q)
                    self.active_queues.append({
                        'name': queue,
                        'color': color,
                        'state': state,
                        'quadrant': q,
                        'rounds_remaining': random.randint(1, self.n_rounds - self.current_round + 1)
                    })
        
        # Ensure minimum one active queue
        self._ensure_active_queues()
        
        # Display round information
        available = ', '.join(q['name'] for q in self.active_queues)
        print(f"\nRound {self.current_round}/{self.n_rounds}")
        print(f"Active queues: {available}")
        print("You press <<")

    def get_queue_state(self, choice: str) -> Optional[str]:
        for q in self.active_queues:
            if q['name'] == choice:
                q['state'] = q['color']  # Reveal actual color
                return q['state']
        return None

    def process_choice(self, choice: str) -> None:
        state = self.get_queue_state(choice.upper())
        if state:
            print(f"{choice.upper()} >> shows {state}")
        else:
            print(f"Invalid choice: {choice}")

    def final_question(self) -> None:
        valid = [str(q+1) for q in self.quadrants]
        answer = input(
            f"\nWhich quadrant had the 90/10 distribution? ({'/'.join(valid)}) << "
        )
        
        if answer in valid:
            correct = str(self.biased_quadrant + 1)
            if answer == correct:
                print(f"Correct! Quadrant {correct} (+100 points)")
            else:
                print(f"Incorrect. Right answer was Quadrant {correct} (-100 points)")
        else:
            print("Invalid quadrant selection")

    def run_game(self):
        print(self.generate_task_description())
        for _ in range(self.n_rounds):
            self.play_round()
            while True:
                choice = input("Choose queue: ").strip().upper()
                if choice in {q['name'] for q in self.active_queues}:
                    break
                print("Invalid choice, try again")
            self.process_choice(choice)
        self.final_question()


# Example usage
if __name__ == "__main__":
    # 2 quadrants, 1 queue each (original ModifiedTask behavior)
    task = VSTtask(n_rounds=10, n_quadrants=2, n_queues=2)
    task.run_game()
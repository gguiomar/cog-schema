import random
from typing import List, Dict, Optional

class ModifiedTask:
    def __init__(self, nrounds: int, num_quadrants: int = 2):
        if not 2 <= num_quadrants <= 4:
            raise ValueError("Number of quadrants must be between 2 and 4")
            
        self.nrounds = nrounds
        self.squares_per_quadrant = 2
        self.current_round = 0
        
        
        self.letter_pairs = {
            0: ['A', 'B'],  
            1: ['P', 'Q'],  
            2: ['X', 'Y'],  
            3: ['M', 'N']   
        }
        self.quadrants = list(range(num_quadrants))
        self.biased_quadrant = random.choice(self.quadrants)
        
        self.black_squares: List[Dict] = []  

    def generate_task_description(self) -> str:
        quadrant_letters = [f"Quadrant {i+1}" for i in self.quadrants]
        square_names = []
        for q in self.quadrants:
            square_names.extend(self.letter_pairs[q])
            
        description = (
            f"You will play a game with {self.nrounds} rounds.\n"
            "In each round you'll be shown a set of black squares distributed over "
            f"{len(self.quadrants)} quadrants of a rectangle.\n"
            f"The available quadrants are: {', '.join(quadrant_letters)}.\n"
            "Each quadrant contains 2 squares with specific letter names:\n" +
            '\n'.join([f"- Quadrant {i+1}: squares {' and '.join(self.letter_pairs[i])}" 
                      for i in self.quadrants]) + "\n"
            "Each black square will be accessible for a finite amount of rounds until it disappears.\n"
            "New black squares might or might not appear in each round.\n"
            "In each round you'll be able to choose one of the available black squares "
            "and learn its color (GREEN or RED) by selecting its letter.\n"
            f"Once the {self.nrounds} rounds are finished, you'll be asked to choose "
            "which quadrant had the highest ratio of one color to the other.\n"
            "If you guess correctly you gain 100 points, if incorrectly you lose 100 points."
        )
        return description

    def get_square_color(self, quadrant: int) -> str:
        """Determine color based on quadrant probability"""
        if quadrant == self.biased_quadrant:
            return 'RED' if random.random() < 0.9 else 'GREEN'
        return random.choice(['RED', 'GREEN'])

    def get_task_output(self, choice: str) -> Optional[str]:
        """Return the color of the chosen square if it exists"""
        for square in self.black_squares:
            if square['name'] == choice:
                return square['state']
        return None

    def play_round(self) -> None:
        self.current_round += 1
        self.black_squares = []
        
        
        for quadrant in self.quadrants:
            for letter in self.letter_pairs[quadrant]:
                if random.random() < 0.5:  
                    state = self.get_square_color(quadrant)
                    rounds_remaining = random.randint(1, self.nrounds - self.current_round + 1)
                    self.black_squares.append({
                        'name': letter,
                        'state': state,
                        'quadrant': quadrant,
                        'rounds_remaining': rounds_remaining
                    })

        
        available_squares = ', '.join(square['name'] for square in self.black_squares)
        print(f"Round {self.current_round}: You see black squares {available_squares}. You press <<")

    def complete_choice(self, choice: str) -> None:
        state = self.get_task_output(choice)
        if state:
            print(f"{choice} >> and see the color {state}.")
        else:
            print(f"Invalid choice: {choice}")

    def ask_final_question(self) -> None:
        print("Which quadrant has the highest ratio difference <<")
        valid_choices = [str(i+1) for i in self.quadrants]
        choice_quadrant = input(f"Enter your choice ({', '.join(valid_choices)}): ")
        
        if choice_quadrant.isdigit() and choice_quadrant in valid_choices:
            correct_answer = self.biased_quadrant + 1
            is_correct = int(choice_quadrant) == correct_answer
            points = 100 if is_correct else -100
            print(f"You chose quadrant {choice_quadrant}. "
                  f"The correct answer was quadrant {correct_answer}. "
                  f"You {'gain' if is_correct else 'lose'} {abs(points)} points!")
        else:
            print("Invalid choice!")


if __name__ == "__main__":
    task = ModifiedTask(nrounds=5, num_quadrants=2)
    print(task.generate_task_description())

    for _ in range(task.nrounds):
        task.play_round()
        user_choice = input("Enter your choice (letter): ")
        task.complete_choice(user_choice)

    task.ask_final_question()
#%%
import random

class Task:
    def __init__(self, nrounds, K):
        self.nrounds = nrounds
        self.K = K
        self.quadrants = ['A', 'B', 'C', 'D']
        self.black_squares = []  # Format: [{'name': 'A1', 'state': 'GREEN', 'rounds_remaining': 3}, ...]
        self.current_round = 0

    def generate_task_description(self):
        description = (
            f"You will play a game with {self.nrounds} rounds.\n"
            "In each round you'll be shown a set of black squares distributed over 4 quadrants of a rectangle.\n"
            "Each quadrant is named A, B, C, D.\n"
            f"There can be up to K = {self.K} black squares in each quadrant, each named after its number and quadrant: "
            f"{', '.join([f'{q}{i+1}' for q in self.quadrants for i in range(self.K)])}.\n"
            "Each black square will be accessible for a finite amount of rounds until it disappears.\n"
            "New black squares might or not appear in each round.\n"
            "In each round you'll be able to choose one of the available black squares and know its identity which can be either GREEN or RED by pressing the button corresponding to its name.\n"
            f"Once the {self.nrounds} rounds are finished you'll be asked to choose which quadrant had the highest ratio of one color to the other.\n"
            "If you guess correctly you gain 100 points, if incorrectly you lose 100 points."
        )
        return description

    def get_task_output(self, choice):
        # Find the chosen square
        for square in self.black_squares:
            if square['name'] == choice:
                return square['state']
        return None  # If the choice doesn't exist

    def play_round(self):
        self.current_round += 1
        # Generate available squares for the round
        self.black_squares = []
        for quadrant in self.quadrants:
            for i in range(self.K):
                if random.random() < 0.5:  # 50% chance of a square appearing
                    state = random.choice(['GREEN', 'RED'])
                    rounds_remaining = random.randint(1, self.nrounds - self.current_round + 1)
                    self.black_squares.append({
                        'name': f'{quadrant}{i+1}',
                        'state': state,
                        'rounds_remaining': rounds_remaining
                    })

        # Display available squares
        available_squares = ', '.join(square['name'] for square in self.black_squares)
        print(f"Round {self.current_round}: You see black squares {available_squares}. You press <<")

    def complete_choice(self, choice):
        state = self.get_task_output(choice)
        if state:
            print(f"{choice} >> and see the color {state}.")
        else:
            print(f"Invalid choice: {choice}")

    def ask_final_question(self):
        print("Which quadrant has the highest ratio difference <<")
        # This function would await an input of the user's choice of quadrant
        choice_quadrant = input("Enter your choice (A, B, C, D): ")
        print(f"You chose quadrant {choice_quadrant}.")

#%%
# Example usage

task = Task(nrounds=5, K=3)
print(task.generate_task_description())

for _ in range(task.nrounds):
    task.play_round()
    user_choice = input("Enter your choice (e.g., A1): ")
    task.complete_choice(user_choice)

task.ask_final_question()

# %%

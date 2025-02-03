import random
import json
import os
from datetime import datetime
from flask import Flask, render_template, request, session, redirect, url_for

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Replace with a secure key

# --- VSTtask Class ---

class VSTtask:
    def __init__(self, n_rounds: int, n_quadrants: int = 2, n_queues: int = 1):
        if not 2 <= n_quadrants <= 4:
            raise ValueError("Number of quadrants must be between 2 and 4")
        if n_queues < 1:
            raise ValueError("Number of queues per quadrant must be at least 1")
            
        self.n_rounds = n_rounds
        self.n_quadrants = n_quadrants
        self.n_queues = n_queues
        
        # Setup quadrants and queues
        self.letters = [chr(65 + i) for i in range(n_quadrants * n_queues)]
        self.queue_map = {
            q: self.letters[q*n_queues:(q+1)*n_queues]
            for q in range(n_quadrants)
        }
        
        self.quadrants = list(range(n_quadrants))
        self.biased_quadrant = random.choice(self.quadrants)
        self.rounds = self._generate_rounds()
        
    def _get_color(self, quadrant: int) -> str:
        if quadrant == self.biased_quadrant:
            return 'RED' if random.random() < 0.9 else 'GREEN'
        return random.choice(['RED', 'GREEN'])
        
    def _generate_rounds(self):
        while True:
            rounds = []
            for _ in range(self.n_rounds):
                round_queues = []
                for q in self.quadrants:
                    for queue in self.queue_map[q]:
                        if random.random() < 0.5:
                            round_queues.append({
                                'name': queue,
                                'color': self._get_color(q),
                                'quadrant': q
                            })
                if not round_queues:
                    q = random.choice(self.quadrants)
                    queue = random.choice(self.queue_map[q])
                    round_queues.append({
                        'name': queue,
                        'color': self._get_color(q),
                        'quadrant': q
                    })
                # Each round now includes a random duration (in milliseconds)
                duration = random.randint(2000, 5000)
                rounds.append({'queues': round_queues, 'duration': duration})
            if self._validate_rounds([r['queues'] for r in rounds]):
                return rounds
    
    def _validate_rounds(self, rounds):
        color_counts = {q: {'RED': 0, 'GREEN': 0} for q in self.quadrants}
        for round_queues in rounds:
            for queue in round_queues:
                q = queue['quadrant']
                color = queue['color']
                color_counts[q][color] += 1
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
    
    def get_round_data(self, round_num: int):
        return self.rounds[round_num]
    
    def get_task_description(self) -> str:
        return (
            f"You will play a game with {self.n_rounds} rounds.<br>"
            "In each round you'll see active queues (clickable buttons):<br>"
            "One quadrant has 90% one color / 10% the other<br>"
            "Other quadrants have a 50/50 color distribution<br>"
            "At least one queue is active per round<br>"
            "Active queues disappear after a short duration.<br><br>"
            f"After {self.n_rounds} rounds, identify the biased quadrant.<br>"
            "Correct: +100 points, Wrong: -100 points."
        )
    
    def process_choice(self, choice: str, round_queues) -> str:
        for queue in round_queues:
            if queue['name'] == choice:
                return queue['color']
        return None

# --- Logging Route ---

@app.route('/log_choice', methods=['POST'])
def log_choice():
    # Expecting JSON payload with { round: <int>, cue: {name: <str>, color: <str>} }
    data = request.get_json()
    if data is None:
        return "No data provided", 400
    
    # Add server-side timestamp
    data["timestamp"] = datetime.utcnow().isoformat() + "Z"
    
    # Use a session-specific log file. If not set, create one.
    log_filename = session.get("log_filename")
    if not log_filename:
        # Create a unique filename
        log_filename = f"logs/choice_{datetime.utcnow().timestamp()}_{random.randint(1000,9999)}.json"
        session["log_filename"] = log_filename
    os.makedirs("logs", exist_ok=True)
    
    # Append the log as a JSON line.
    with open(log_filename, "a") as f:
        f.write(json.dumps(data) + "\n")
    return "OK", 200

# --- Flask Routes ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start')
def start():
    # Set game parameters (adjust as desired):
    n_rounds = 5
    n_quadrants = 2
    n_queues = 1
    task = VSTtask(n_rounds=n_rounds, n_quadrants=n_quadrants, n_queues=n_queues)
    session['game'] = {
         'rounds': task.rounds,
         'biased_quadrant': task.biased_quadrant,
         'n_rounds': task.n_rounds,
         'n_quadrants': task.n_quadrants,
         'current_round': 0,
         'task_description': task.get_task_description()
    }
    # Initialize an empty log list (if you prefer in-session logging)
    session['choices_log'] = []
    return redirect(url_for('round_page', round_number=0))

@app.route('/round/<int:round_number>', methods=['GET'])
def round_page(round_number):
    game = session.get('game')
    if not game:
        return redirect(url_for('index'))
    if round_number >= game['n_rounds']:
        return redirect(url_for('final'))
    round_data = game['rounds'][round_number]
    return render_template('round.html', round_number=round_number, round_data=round_data)

@app.route('/final', methods=['GET', 'POST'])
def final():
    game = session.get('game')
    if not game:
        return redirect(url_for('index'))
    if request.method == 'POST':
        try:
            chosen = int(request.form.get('biased_quadrant'))
        except (ValueError, TypeError):
            chosen = -1
        correct = (chosen == game['biased_quadrant'])
        score = 100 if correct else -100
        return render_template('result.html', chosen=chosen, correct=correct, score=score,
                               biased=game['biased_quadrant'])
    return render_template('final.html', n_quadrants=game['n_quadrants'])

if __name__ == '__main__':
    app.run(debug=True)

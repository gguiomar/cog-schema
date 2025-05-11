import os
import json
import pandas as pd
import numpy as np
from typing import Tuple

def load_all_game_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Reads all JSON log files under ./filtered_logs (from cwd) and returns two DataFrames:

    - Choices: columns [game_id, round_nb, available_cues, chosen_quadrant, color,
                       round_time, A_logit, B_logit, C_logit, D_logit]
    - Games:   columns [game_id, model, nb_quadrants, nb_rounds,
                        success, chosen_quadrant, biased_quadrant,
                        trial_time, A_logit, B_logit, C_logit, D_logit, src_file]
    """
    CURRENT_DIR = os.getcwd()
    root_dir = os.path.join(CURRENT_DIR, "filtered_logs")
    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f"Could not find folder: {root_dir}")

    choices_records = []
    games_records   = []

    for dirpath, _, filenames in os.walk(root_dir):
        model_name = os.path.basename(dirpath)  # e.g. "Qwen_3B"

        for fname in filenames:
            if not fname.endswith('.json'):
                continue

            src_file = os.path.join(dirpath, fname)
            with open(src_file, 'r') as fh:
                data = json.load(fh)

            nb_quadrants = data['metrics']['n_quadrants']
            nb_rounds    = data['metrics']['n_rounds']

            for trial_idx, raw in enumerate(data['raw_results'], start=1):
                game_id = f"{model_name}_{nb_rounds}r_{nb_quadrants}q_trial{trial_idx}"

                # last trial
                trial = raw['trials'][-1]

                # logits per round + final choice logits
                logits_list = trial.get('logits', [])
                
                # get final logits (last in list)
                final_logits = logits_list[-1] if logits_list else {}

                games_records.append({
                    'game_id':         game_id,
                    'model':           model_name,
                    'nb_quadrants':    nb_quadrants,
                    'nb_rounds':       nb_rounds,
                    'success':         trial.get('success', False),
                    'chosen_quadrant': trial.get('final_choice'),
                    'biased_quadrant': trial.get('correct_quadrant'),
                    'trial_time':      trial.get('trial_time'),
                    'A_logit':         final_logits.get('A', 0),
                    'B_logit':         final_logits.get('B', 0),
                    'C_logit':         final_logits.get('C', 0),
                    'D_logit':         final_logits.get('D', 0),
                    'src_file':        fname
                })

                # now rounds
                for round_nb, rd in enumerate(trial['rounds'], start=1):
                    logit_dict = logits_list[round_nb-1] if round_nb-1 < len(logits_list) else {}
                    record = {
                        'game_id':         game_id,
                        'round_nb':        round_nb,
                        'available_cues':  rd.get('available_cues'),
                        'chosen_quadrant': rd.get('choice'),
                        'color':           rd.get('result'),
                        'round_time':      rd.get('round_time'),
                        'A_logit':         logit_dict.get('A', 0),
                        'B_logit':         logit_dict.get('B', 0),
                        'C_logit':         logit_dict.get('C', 0),
                        'D_logit':         logit_dict.get('D', 0),
                    }
                    choices_records.append(record)

    Choices = pd.DataFrame(choices_records)
    Games   = pd.DataFrame(games_records)
    return Choices, Games

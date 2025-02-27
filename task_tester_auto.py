#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import random
from typing import List, Dict, Optional

# Add parent directory to path to import VSTtask
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tasks.VSTtask import VSTtask

def test_bias_detection():
    """Test the bias detection task."""
    print("\n=== Testing Bias Detection Task ===")
    task = VSTtask(n_rounds=10, n_quadrants=2, n_cues=1, task_type=VSTtask.TASK_BIAS_DETECTION)
    
    print("Task Description:")
    print(task.get_task_description())
    
    print(f"\nCorrect Answer: Quadrant {task.get_correct_answer()}")
    
    print("\nSimulating rounds:")
    for round_num in range(task.n_rounds):
        round_data = task.get_round_data(round_num)
        available_cues = ', '.join(cue['name'] for cue in round_data)
        print(f"Round {round_num + 1}: Available cues: {available_cues}")
        
        # Randomly select a cue
        chosen_cue = random.choice([cue['name'] for cue in round_data])
        result = task.process_choice(chosen_cue, round_data)
        
        # Find which quadrant this cue belongs to
        quadrant = None
        for cue in round_data:
            if cue['name'] == chosen_cue:
                quadrant = cue['quadrant'] + 1  # +1 for 1-based indexing
                break
        
        print(f"  Selected: {chosen_cue} (Quadrant {quadrant}) - Color: {result}")
    
    print("\nFinal Question:")
    print(task.get_final_question())

def test_pattern_detection():
    """Test the pattern detection task."""
    print("\n=== Testing Pattern Detection Task ===")
    task = VSTtask(n_rounds=10, n_quadrants=2, n_cues=1, task_type=VSTtask.TASK_PATTERN_DETECTION)
    
    print("Task Description:")
    print(task.get_task_description())
    
    print(f"\nCorrect Answer: Cue {task.get_correct_answer()}")
    
    print("\nSimulating rounds:")
    pattern_appearances = []
    
    for round_num in range(task.n_rounds):
        round_data = task.get_round_data(round_num)
        available_cues = ', '.join(cue['name'] for cue in round_data)
        print(f"Round {round_num + 1}: Available cues: {available_cues}")
        
        # Randomly select a cue
        chosen_cue = random.choice([cue['name'] for cue in round_data])
        result = task.process_choice(chosen_cue, round_data)
        
        # Check if pattern cue was selected and track its color
        if chosen_cue == task.pattern_cue:
            pattern_appearances.append(result)
            print(f"  Selected: {chosen_cue} - Color: {result} (Pattern Cue)")
        else:
            print(f"  Selected: {chosen_cue} - Color: {result}")
    
    print("\nPattern cue observations:")
    if pattern_appearances:
        print(f"  Cue {task.pattern_cue} appeared {len(pattern_appearances)} times with colors: {', '.join(pattern_appearances)}")
    else:
        print(f"  Cue {task.pattern_cue} was never selected")
    
    print("\nFinal Question:")
    print(task.get_final_question())

def test_conditional_probability():
    """Test the conditional probability task."""
    print("\n=== Testing Conditional Probability Task ===")
    task = VSTtask(n_rounds=10, n_quadrants=2, n_cues=1, task_type=VSTtask.TASK_CONDITIONAL_PROBABILITY)
    
    print("Task Description:")
    print(task.get_task_description())
    
    print(f"\nCorrect Answer: Cue {task.get_correct_answer()}")
    print(f"Trigger Cue: {task.conditional_cue_trigger}, Response Cue: {task.conditional_cue_response}")
    
    print("\nSimulating rounds:")
    previous_trigger_color = None
    for round_num in range(task.n_rounds):
        round_data = task.get_round_data(round_num)
        available_cues = ', '.join(cue['name'] for cue in round_data)
        print(f"Round {round_num + 1}: Available cues: {available_cues}")
        
        # First check if trigger cue is available and if so, select it
        trigger_available = False
        for cue in round_data:
            if cue['name'] == task.conditional_cue_trigger:
                trigger_available = True
                break
        
        if trigger_available:
            chosen_cue = task.conditional_cue_trigger
        else:
            # Otherwise randomly choose one
            chosen_cue = random.choice([cue['name'] for cue in round_data])
        
        result = task.process_choice(chosen_cue, round_data)
        
        # Track trigger colors for analysis
        if chosen_cue == task.conditional_cue_trigger:
            previous_trigger_color = result
            print(f"  Selected: {chosen_cue} - Color: {result} (Trigger Cue)")
        elif chosen_cue == task.conditional_cue_response:
            if previous_trigger_color == 'BLUE':
                print(f"  Selected: {chosen_cue} - Color: {result} (Response Cue, after BLUE trigger)")
            else:
                print(f"  Selected: {chosen_cue} - Color: {result} (Response Cue)")
        else:
            print(f"  Selected: {chosen_cue} - Color: {result}")
    
    print("\nFinal Question:")
    print(task.get_final_question())

def summarize_statistics():
    """Run multiple trials and summarize statistics for each task type."""
    print("\n=== Task Statistics Summary ===")
    
    num_trials = 100
    rounds = 10
    quadrants = 2
    cues = 1
    
    for task_type, name in [
        (VSTtask.TASK_BIAS_DETECTION, "Bias Detection"),
        (VSTtask.TASK_PATTERN_DETECTION, "Pattern Detection"),
        (VSTtask.TASK_CONDITIONAL_PROBABILITY, "Conditional Probability")
    ]:
        print(f"\n--- {name} Task Statistics ({num_trials} trials) ---")
        
        # Track validation success rate
        validation_success = 0
        
        # Initialize task-specific metrics
        if task_type == VSTtask.TASK_BIAS_DETECTION:
            biased_quadrant_counts = {}
        elif task_type == VSTtask.TASK_PATTERN_DETECTION:
            pattern_visibility = []
        elif task_type == VSTtask.TASK_CONDITIONAL_PROBABILITY:
            blue_trigger_counts = []
            red_after_blue_ratios = []
        
        for _ in range(num_trials):
            try:
                task = VSTtask(rounds, quadrants, cues, task_type=task_type)
                validation_success += 1
                
                # Collect task-specific statistics
                if task_type == VSTtask.TASK_BIAS_DETECTION:
                    biased_quad = task.biased_quadrant + 1
                    biased_quadrant_counts[biased_quad] = biased_quadrant_counts.get(biased_quad, 0) + 1
                
                elif task_type == VSTtask.TASK_PATTERN_DETECTION:
                    # Count how many times pattern cue appears
                    pattern_appearances = 0
                    for round_data in task.rounds:
                        for cue in round_data:
                            if cue['name'] == task.pattern_cue:
                                pattern_appearances += 1
                    pattern_visibility.append(pattern_appearances)
                
                elif task_type == VSTtask.TASK_CONDITIONAL_PROBABILITY:
                    # Track blue trigger counts and red-after-blue ratios
                    trigger_blue_count = 0
                    response_after_blue_count = 0
                    response_red_after_blue_count = 0
                    
                    for i in range(len(task.rounds) - 1):
                        trigger_was_blue = False
                        for cue in task.rounds[i]:
                            if cue['name'] == task.conditional_cue_trigger and cue['color'] == 'BLUE':
                                trigger_was_blue = True
                                trigger_blue_count += 1
                                break
                        
                        if trigger_was_blue:
                            for cue in task.rounds[i+1]:
                                if cue['name'] == task.conditional_cue_response:
                                    response_after_blue_count += 1
                                    if cue['color'] == 'RED':
                                        response_red_after_blue_count += 1
                                    break
                    
                    blue_trigger_counts.append(trigger_blue_count)
                    if response_after_blue_count > 0:
                        red_after_blue_ratios.append(response_red_after_blue_count / response_after_blue_count)
            
            except Exception as e:
                print(f"Error in trial: {e}")
        
        # Print validation success rate
        print(f"Validation Success Rate: {validation_success}/{num_trials} ({validation_success/num_trials:.2%})")
        
        # Print task-specific statistics
        if task_type == VSTtask.TASK_BIAS_DETECTION:
            print("Biased Quadrant Distribution:")
            for quad, count in sorted(biased_quadrant_counts.items()):
                print(f"  Quadrant {quad}: {count}/{validation_success} ({count/validation_success:.2%})")
        
        elif task_type == VSTtask.TASK_PATTERN_DETECTION:
            avg_visibility = sum(pattern_visibility) / len(pattern_visibility) if pattern_visibility else 0
            print(f"Pattern Cue Visibility: {avg_visibility:.2f} rounds on average (out of {rounds})")
            
            # Count distribution of visibility
            visibility_dist = {}
            for v in pattern_visibility:
                visibility_dist[v] = visibility_dist.get(v, 0) + 1
            
            print("Pattern Cue Visibility Distribution:")
            for v, count in sorted(visibility_dist.items()):
                print(f"  {v} rounds: {count}/{validation_success} ({count/validation_success:.2%})")
        
        elif task_type == VSTtask.TASK_CONDITIONAL_PROBABILITY:
            avg_blue_triggers = sum(blue_trigger_counts) / len(blue_trigger_counts) if blue_trigger_counts else 0
            avg_red_after_blue = sum(red_after_blue_ratios) / len(red_after_blue_ratios) if red_after_blue_ratios else 0
            
            print(f"Blue Trigger Frequency: {avg_blue_triggers:.2f} rounds on average (out of {rounds})")
            print(f"Red After Blue Ratio: {avg_red_after_blue:.2%} on average")

def run_all_tests():
    """Run tests for all task types."""
    test_bias_detection()
    test_pattern_detection()
    test_conditional_probability()
    summarize_statistics()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        task_type = sys.argv[1].lower()
        if task_type == "bias":
            test_bias_detection()
        elif task_type == "pattern":
            test_pattern_detection()
        elif task_type == "conditional":
            test_conditional_probability()
        elif task_type == "stats":
            summarize_statistics()
        else:
            print(f"Unknown task type: {task_type}")
            print("Available options: bias, pattern, conditional, stats")
    else:
        run_all_tests()
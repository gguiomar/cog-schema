#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import random
import time
from typing import List, Dict, Optional

# Add parent directory to path to import VSTtask
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tasks.VSTtask import VSTtask

def clear_screen():
    """Clear the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header(title):
    """Print a formatted header."""
    width = 60
    print("\n" + "=" * width)
    print(title.center(width))
    print("=" * width + "\n")

def select_task_type():
    """Let the user select which task type to play."""
    print_header("SELECT TASK TYPE")
    print("1. Bias Detection - Find the quadrant with highest RED ratio")
    print("2. Pattern Detection - Find the cue with alternating pattern")
    print("3. Conditional Probability - Find the quadrant containing the cue most likely to be RED after a cue showed BLUE")
    print("4. Classical Conditioning - Report whether you received a reward")
    
    while True:
        choice = input("\nEnter your choice (1-4): ").strip()
        if choice == "1":
            return VSTtask.TASK_BIAS_DETECTION
        elif choice == "2":
            return VSTtask.TASK_PATTERN_DETECTION
        elif choice == "3":
            return VSTtask.TASK_CONDITIONAL_PROBABILITY
        elif choice == "4":
            return VSTtask.TASK_CLASSICAL_CONDITIONING
        else:
            print("Invalid choice. Please enter 1, 2, 3, or 4.")

def configure_task(task_type):
    """Configure task parameters."""
    print_header("CONFIGURE TASK")
    
    # Number of rounds is always configurable.
    while True:
        try:
            n_rounds = int(input("Number of rounds (3-20): ").strip())
            if 3 <= n_rounds <= 20:
                break
            else:
                print("Please enter a number between 3 and 20.")
        except ValueError:
            print("Please enter a valid number.")
    
    if task_type == VSTtask.TASK_CLASSICAL_CONDITIONING:
        n_quadrants = 1
        n_cues = 1
        print("For Classical Conditioning, n_quadrants and n_cues are set to 1.")
    else:
        while True:
            try:
                n_quadrants = int(input("Number of quadrants (2-4): ").strip())
                if 2 <= n_quadrants <= 4:
                    break
                else:
                    print("Please enter a number between 2 and 4.")
            except ValueError:
                print("Please enter a valid number.")
        
        while True:
            try:
                n_cues = int(input("Number of cues per quadrant (1-3): ").strip())
                if 1 <= n_cues <= 3:
                    break
                else:
                    print("Please enter a number between 1 and 3.")
            except ValueError:
                print("Please enter a valid number.")
    
    return n_rounds, n_quadrants, n_cues

def print_observations(observations):
    """Print the observation history."""
    if not observations:
        print("No observations yet.")
        return
    
    print("\n--- Observation History ---")
    for obs in observations:
        print(f"Round {obs['round']}: Selected {obs['cue']} - Saw {obs['color']}")
    print("---------------------------\n")

def play_task(task_type, n_rounds, n_quadrants, n_cues):
    """Play the selected task."""
    # Initialize task
    task = VSTtask(n_rounds=n_rounds, n_quadrants=n_quadrants, n_cues=n_cues, task_type=task_type)
    
    # Track observations (for tasks that involve multiple cues)
    observations = []
    
    # Print task description
    clear_screen()
    print_header("TASK DESCRIPTION")
    print(task.get_task_description())
    input("\nPress Enter to begin...")
    
    # Main game loop: for CLASSICAL_CONDITIONING each round shows one cue (always RED)
    for round_num in range(task.n_rounds):
        clear_screen()
        print_header(f"ROUND {round_num + 1}/{task.n_rounds}")
        
        # For tasks other than classical conditioning, display previous observations
        if task_type != VSTtask.TASK_CLASSICAL_CONDITIONING:
            print_observations(observations)
        
        # Get round data
        round_data = task.get_round_data(round_num)
        available_cues = [cue['name'] for cue in round_data]
        
        # Display available cues
        if task_type == VSTtask.TASK_CLASSICAL_CONDITIONING:
            print(f"Conditioned Stimulus: {available_cues[0]} (Color: {round_data[0]['color']})")
        else:
            print(f"Available cues: {', '.join(available_cues)}")
        
        # Get player's choice if applicable (for tasks where the subject must select a cue)
        if task_type in (VSTtask.TASK_BIAS_DETECTION, VSTtask.TASK_PATTERN_DETECTION, VSTtask.TASK_CONDITIONAL_PROBABILITY):
            while True:
                choice = input("\nSelect a cue: ").strip().upper()
                if choice in available_cues:
                    break
                else:
                    print(f"Invalid choice. Please select from: {', '.join(available_cues)}")
            
            # Process choice
            result = task.process_choice(choice, round_data)
            # For bias detection, also show quadrant info if available
            quadrant = None
            if task_type == VSTtask.TASK_BIAS_DETECTION:
                for cue in round_data:
                    if cue['name'] == choice:
                        quadrant = cue['quadrant'] + 1  # 1-indexed
                        break
                print(f"\nYou selected {choice} (Quadrant {quadrant}) and saw {result}")
            else:
                print(f"\nYou selected {choice} and saw {result}")
            
            observation = {
                'round': round_num + 1,
                'cue': choice,
                'color': result,
                'quadrant': quadrant
            }
            observations.append(observation)
        else:
            # For classical conditioning, no choice is needed; simply wait a moment.
            print("\nObserve the cue...")
            time.sleep(1)
        
        input("\nPress Enter for next round...")
    
    # Final decision stage
    clear_screen()
    print_header("FINAL DECISION")
    
    # Display observation history if applicable
    if task_type != VSTtask.TASK_CLASSICAL_CONDITIONING:
        print_observations(observations)
    
    # Get final question and input answer
    print(task.get_final_question())
    final_answer = input("\nYour answer: ").strip().upper()
    
    # Check if correct
    correct_answer = task.get_correct_answer()
    success = (final_answer == correct_answer)
    
    # Show result
    print_header("RESULT")
    print(f"Your answer: {final_answer}")
    print(f"Correct answer: {correct_answer}")
    if task_type == VSTtask.TASK_CLASSICAL_CONDITIONING:
        print(f"\n{'Success! You received a reward (+100 points)' if success else 'Incorrect. No reward (-100 points)'}")
        print(f"\nThe simulated reward outcome was: {correct_answer}")
    else:
        print(f"{'Success! +100 points' if success else 'Incorrect. -100 points'}")
    
    # Additional analysis for non-classical tasks
    if task_type == VSTtask.TASK_BIAS_DETECTION:
        quadrant_counts = {}
        for obs in observations:
            if obs['quadrant'] is not None:
                if obs['quadrant'] not in quadrant_counts:
                    quadrant_counts[obs['quadrant']] = {'RED': 0, 'GREEN': 0}
                quadrant_counts[obs['quadrant']][obs['color']] += 1
        
        print("\nColor counts per quadrant:")
        for q in sorted(quadrant_counts.keys()):
            red = quadrant_counts[q]['RED']
            green = quadrant_counts[q]['GREEN']
            total = red + green
            red_ratio = red / total if total > 0 else 0
            print(f"Quadrant {q}: RED={red}, GREEN={green}, RED ratio={red_ratio:.2f}")
    
    elif task_type == VSTtask.TASK_PATTERN_DETECTION:
        pattern_observations = [obs['color'] for obs in observations if obs['cue'] == task.pattern_cue]
        print(f"\nPattern cue ({task.pattern_cue}) observations: {', '.join(pattern_observations) if pattern_observations else 'None'}")
        if len(pattern_observations) >= 2:
            alternating = all(pattern_observations[i] != pattern_observations[i-1] for i in range(1, len(pattern_observations)))
            print(f"Pattern was {'detectable' if alternating else 'not clearly detectable'} from your observations.")
        else:
            print("Not enough observations of the pattern cue to detect the pattern.")
    
    elif task_type == VSTtask.TASK_CONDITIONAL_PROBABILITY:
        blue_trigger_count = 0
        response_after_blue_count = 0
        response_red_after_blue_count = 0
        for i in range(len(observations)-1):
            if observations[i]['cue'] == task.conditional_cue_uniform and observations[i]['color'] == 'BLUE':
                blue_trigger_count += 1
                if observations[i+1]['cue'] == task.conditional_cue_biased:
                    response_after_blue_count += 1
                    if observations[i+1]['color'] == 'RED':
                        response_red_after_blue_count += 1
        print(f"\nUniform cue showed BLUE: {blue_trigger_count} times")
        if response_after_blue_count > 0:
            red_ratio = response_red_after_blue_count / response_after_blue_count
            print(f"Biased cue after BLUE: {response_after_blue_count} times; RED after BLUE ratio: {red_ratio:.2f}")
        else:
            print("No biased cue observations following a BLUE uniform cue.")
    
    # Ask to play again
    print("\nWould you like to play again?")
    play_again = input("Enter 'y' to play again, any other key to quit: ").strip().lower()
    return play_again == 'y'

def main():
    """Main function to run the task tester."""
    clear_screen()
    print_header("VST TASK TESTER")
    print("Welcome to the Visual Sampling Task Tester!")
    print("This program allows you to play different VST tasks.\n")
    
    play_again = True
    while play_again:
        task_type = select_task_type()
        n_rounds, n_quadrants, n_cues = configure_task(task_type)
        play_again = play_task(task_type, n_rounds, n_quadrants, n_cues)
    
    print_header("THANK YOU FOR PLAYING!")
    print("Goodbye!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nGame interrupted. Exiting...")
    except Exception as e:
        print(f"\n\nAn error occurred: {str(e)}")

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
    print("3. Conditional Probability - Find the cue most likely to be RED after another shows BLUE")
    
    while True:
        choice = input("\nEnter your choice (1-3): ").strip()
        if choice == "1":
            return VSTtask.TASK_BIAS_DETECTION
        elif choice == "2":
            return VSTtask.TASK_PATTERN_DETECTION
        elif choice == "3":
            return VSTtask.TASK_CONDITIONAL_PROBABILITY
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")

def configure_task():
    """Configure task parameters."""
    print_header("CONFIGURE TASK")
    
    # Number of rounds
    while True:
        try:
            n_rounds = int(input("Number of rounds (3-20): ").strip())
            if 3 <= n_rounds <= 20:
                break
            else:
                print("Please enter a number between 3 and 20.")
        except ValueError:
            print("Please enter a valid number.")
    
    # Number of quadrants
    while True:
        try:
            n_quadrants = int(input("Number of quadrants (2-4): ").strip())
            if 2 <= n_quadrants <= 4:
                break
            else:
                print("Please enter a number between 2 and 4.")
        except ValueError:
            print("Please enter a valid number.")
    
    # Number of cues per quadrant
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
    
    # Track observations
    observations = []
    
    # Print task description
    clear_screen()
    print_header("TASK DESCRIPTION")
    print(task.get_task_description())
    input("\nPress Enter to begin...")
    
    # Main game loop
    for round_num in range(task.n_rounds):
        clear_screen()
        print_header(f"ROUND {round_num + 1}/{task.n_rounds}")
        
        # Display observation history
        print_observations(observations)
        
        # Get round data
        round_data = task.get_round_data(round_num)
        available_cues = [cue['name'] for cue in round_data]
        
        # Display available cues
        print(f"Available cues: {', '.join(available_cues)}")
        
        # Get player's choice
        while True:
            choice = input("\nSelect a cue: ").strip().upper()
            if choice in available_cues:
                break
            else:
                print(f"Invalid choice. Please select from: {', '.join(available_cues)}")
        
        # Process choice
        result = task.process_choice(choice, round_data)
        
        # Find quadrant for bias detection task
        quadrant = None
        if task_type == VSTtask.TASK_BIAS_DETECTION:
            for cue in round_data:
                if cue['name'] == choice:
                    quadrant = cue['quadrant'] + 1  # +1 because quadrants are 0-indexed in code
                    break
        
        # Show result
        print(f"\nYou selected {choice} and saw {result}")
        if quadrant:
            print(f"(Cue {choice} is from Quadrant {quadrant})")
        
        # Add to observations
        observation = {
            'round': round_num + 1,
            'cue': choice,
            'color': result,
            'quadrant': quadrant
        }
        observations.append(observation)
        
        # Wait before next round
        time.sleep(1)
        input("\nPress Enter for next round...")
    
    # Final decision
    clear_screen()
    print_header("FINAL DECISION")
    
    # Display observation history
    print_observations(observations)
    
    # Ask for final answer
    print(task.get_final_question())
    final_answer = input("\nYour answer: ").strip().upper()
    
    # Check if correct
    correct_answer = task.get_correct_answer()
    success = final_answer == correct_answer
    
    # Show result
    print_header("RESULT")
    print(f"Your answer: {final_answer}")
    print(f"Correct answer: {correct_answer}")
    print(f"{'Success! +100 points' if success else 'Incorrect. -100 points'}")
    
    # Show task-specific details
    if task_type == VSTtask.TASK_BIAS_DETECTION:
        print(f"\nQuadrant {correct_answer} had the highest ratio of RED.")
    elif task_type == VSTtask.TASK_PATTERN_DETECTION:
        print(f"\nCue {correct_answer} showed the alternating pattern.")
    elif task_type == VSTtask.TASK_CONDITIONAL_PROBABILITY:
        print(f"\nCue {correct_answer} was most likely to be RED after cue {task.conditional_cue_trigger} showed BLUE.")
    
    # Analyze observations
    print("\n--- Observation Analysis ---")
    
    if task_type == VSTtask.TASK_BIAS_DETECTION:
        # Count colors per quadrant
        quadrant_counts = {}
        for obs in observations:
            if obs['quadrant'] is not None:
                if obs['quadrant'] not in quadrant_counts:
                    quadrant_counts[obs['quadrant']] = {'RED': 0, 'GREEN': 0}
                
                quadrant_counts[obs['quadrant']][obs['color']] += 1
        
        # Print quadrant statistics
        print("Color counts per quadrant:")
        for q in sorted(quadrant_counts.keys()):
            red = quadrant_counts[q]['RED']
            green = quadrant_counts[q]['GREEN']
            total = red + green
            red_ratio = red / total if total > 0 else 0
            
            print(f"Quadrant {q}: RED={red}, GREEN={green}, RED ratio={red_ratio:.2f}")
    
    elif task_type == VSTtask.TASK_PATTERN_DETECTION:
        # Track pattern cue observations
        pattern_observations = []
        for obs in observations:
            if obs['cue'] == task.pattern_cue:
                pattern_observations.append(obs['color'])
        
        print(f"Pattern cue ({task.pattern_cue}) observations: {', '.join(pattern_observations) if pattern_observations else 'None'}")
        
        # Check if pattern is detectable from observations
        if len(pattern_observations) >= 2:
            alternating = True
            for i in range(1, len(pattern_observations)):
                if pattern_observations[i] == pattern_observations[i-1]:
                    alternating = False
                    break
            
            print(f"Pattern was {'detectable' if alternating else 'not clearly detectable'} from your observations.")
        else:
            print("Not enough observations of the pattern cue to detect the pattern.")
    
    elif task_type == VSTtask.TASK_CONDITIONAL_PROBABILITY:
        # Track trigger and response observations
        trigger_blue_count = 0
        response_after_blue_count = 0
        response_red_after_blue_count = 0
        
        # Analyze pairs of observations
        for i in range(len(observations)-1):
            if observations[i]['cue'] == task.conditional_cue_trigger and observations[i]['color'] == 'BLUE':
                trigger_blue_count += 1
                
                if observations[i+1]['cue'] == task.conditional_cue_response:
                    response_after_blue_count += 1
                    if observations[i+1]['color'] == 'RED':
                        response_red_after_blue_count += 1
        
        print(f"Trigger cue ({task.conditional_cue_trigger}) showed BLUE: {trigger_blue_count} times")
        
        if response_after_blue_count > 0:
            red_after_blue_ratio = response_red_after_blue_count / response_after_blue_count
            print(f"Response cue ({task.conditional_cue_response}) after BLUE trigger: {response_after_blue_count} times")
            print(f"RED after BLUE ratio: {red_after_blue_ratio:.2f}")
        else:
            print(f"You never observed the response cue after the trigger cue showed BLUE.")
    
    # Play again?
    print("\nWould you like to play again?")
    play_again = input("Enter 'y' to play again, any other key to quit: ").strip().lower()
    return play_again == 'y'

def main():
    """Main function to run the task tester."""
    clear_screen()
    print_header("VST TASK TESTER")
    print("Welcome to the Visual Sampling Task Tester!")
    print("This program allows you to play different VST tasks.")
    
    play_again = True
    while play_again:
        # Select task type
        task_type = select_task_type()
        
        # Configure task
        n_rounds, n_quadrants, n_cues = configure_task()
        
        # Play task
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
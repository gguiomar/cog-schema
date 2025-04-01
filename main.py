#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
from manager.TaskManager import TaskManager

def parse_args():
    parser = argparse.ArgumentParser(description='Run G1Bbon LLM benchmark')
    
    # Model selection
    parser.add_argument('--models', nargs='+', default=['Qwen_0.5B'],
                        help='Models to benchmark')
    
    # Task configuration
    parser.add_argument('--rounds', nargs='+', type=int, default=[6], 
                        help='Number of rounds for the VST task')
    parser.add_argument('--quadrants', nargs='+', type=int, default=[4], 
                        help='Number of quadrants for the VST task')
    parser.add_argument('--cues', type=int, default=1, 
                        help='Number of cues per quadrant')
    parser.add_argument('--task-type', type=str, default='bias_detection',
                        choices=['bias_detection', 'pattern_detection', 'conditional_probability'],
                        help='Type of task to run')
    
    # Experiment setup
    parser.add_argument('--simulations', type=int, default=10, 
                        help='Number of simulations per configuration')
    parser.add_argument('--trials', type=int, default=1, 
                        help='Number of trials per simulation')
    
    # Hardware settings
    parser.add_argument('--device', type=str, default='mps',
                        help='Device to run inference on')
    parser.add_argument('--no-unsloth', action='store_false', dest='use_unsloth',
                        help='Disable unsloth optimization')
    
    # API keys (optional)
    parser.add_argument('--openai-key', type=str, default=None, 
                        help='OpenAI API key')
    parser.add_argument('--anthropic-key', type=str, default=None, 
                        help='Anthropic API key')
    
    # Output settings
    parser.add_argument('--output-dir', type=str, default='simulation_results', 
                        help='Legacy directory parameter (not used)')
    parser.add_argument('--verbose', action='store_true', 
                        help='Enable verbose output')
                        
    # Visualization settings
    parser.add_argument('--no-plot', action='store_false', dest='plot',
                        help='Skip generating result plots')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create task manager with parameters
    manager = TaskManager(
        agents=args.models,
        rounds=args.rounds,
        quadrants=args.quadrants,
        n_simulations=args.simulations,
        n_trials=args.trials,
        num_cues=args.cues,
        device=args.device,
        verbose=args.verbose,
        output_dir=args.output_dir,
        openai_api_key=args.openai_key,
        anthropic_api_key=args.anthropic_key,
        use_unsloth=args.use_unsloth,
        task_type=args.task_type  # Pass the task type to TaskManager
    )
    
    # Run benchmarks
    print(f"Running benchmarks for models: {args.models} on task: {args.task_type}")
    results = manager.multiple_benchmarks()
    
    # Get DataFrame but don't save additional files
    df = manager.save_results()
    
    # Generate plot if requested and not already generated
    if args.plot and not hasattr(manager, 'plot_generated'):
        manager.plot_results()
    
    print(f"Benchmark complete! Results saved to logs/ and benchmark plot saved to benchmarks_plots/")
    
if __name__ == "__main__":
    main()
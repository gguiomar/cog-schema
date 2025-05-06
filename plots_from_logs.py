#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
from manager.TaskManager import TaskManager

def parse_args():
    parser = argparse.ArgumentParser(description='Generate benchmark plots from existing log files.')
    
    # Log filtering arguments
    parser.add_argument('--timestamp', type=str, required=True,
                        help='Target timestamp string to filter logs by (e.g., "20250420_170958").')
    parser.add_argument('--quadrants', type=int, required=True,
                        help='Target number of quadrants to filter logs by.')
                        
    # Output settings
    parser.add_argument('--output-file', type=str, default=None,
                        help='Optional filename for the saved plot image (e.g., "my_plot.png"). If None, a default name is generated.')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output during log processing.')

    return parser.parse_args()

def main():
    args = parse_args()
    
    # Instantiate TaskManager primarily for its log plotting function
    # We only need to pass 'verbose' as other parameters are not directly used by plot_results_from_logs
    manager = TaskManager(verbose=args.verbose)
    
    # Call the function to generate plot from logs
    print(f"Attempting to generate plot for timestamp: {args.timestamp}, quadrants: {args.quadrants}")
    results_df = manager.plot_results_from_logs(
        target_timestamp=args.timestamp,
        target_quadrants=args.quadrants,
        output_filename=args.output_file
    )
    
    if not results_df.empty:
        print("\n--- Aggregated Data Used for Plot ---")
        print(results_df.to_string())
        print("\n------------------------------------")
        print(f"Plot generation complete. Check the '{manager.benchmarks_plots_dir}' directory.")
    else:
        print("Plot generation failed or no matching logs were found.")

if __name__ == "__main__":
    main() 
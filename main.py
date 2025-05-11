import argparse
from manager.TaskManager import TaskManager
from tasks.TaskSelector import TaskSelector

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
    parser.add_argument('--task-type', type=str, default='BiasDetection',
                        choices=TaskSelector.get_list(),
                        help='Type of task to run', dest='task_type')

    # Experiment setup
    parser.add_argument('--simulations', type=int, default=10,
                        help='Number of simulations per configuration')
    parser.add_argument('--trials', type=int, default=1,
                        help='Number of trials per simulation')

    # Hardware settings
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to run inference on')
    parser.add_argument('--no-unsloth', action='store_false', dest='use_unsloth', default=False,
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
    
    parser.add_argument('--log-stats', action='store_true', help='Enable stats logging during benchmark')

    parser.add_argument('--activation-layers', type=str, default='post_attention_layernorm',
                        help='Layers to save activations for activation analysis, see model.named_modules() for options')

    parser.add_argument('--automate-activations-gathering', action='store_true', default=True,
                        help='Whether to automate the gathering of activations based on the layer ending. If True, activation-layers argument'
                             'will represent layer ending, e.g. post_attention_layernorm')

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
        log_stats=args.log_stats,
        task_type=TaskSelector.from_string(args.task_type),
        activation_layers=args.activation_layers,
        automate_activations_gathering=args.automate_activations_gathering,
    )

    # Run benchmarks
    print(f"Running benchmarks for models: {args.models} on task: {args.task_type}")
    results = manager.multiple_benchmarks()

    # Get DataFrame but don't save additional files
    df = manager.save_results()

    # Need to fix metrics
    
    # Generate plot if requested and not already generated
    if args.plot and not hasattr(manager, 'plot_generated'):
        manager.plot_results()


if __name__ == "__main__":
    main()
# g1Bbon: Language Model Perceptual-Inference Benchmark

G1Bbon is a benchmark for evaluating (small) Language Models on perceptual inference tasks. It tests an LLM's ability to maintain and update beliefs about the world based on limited, noisy observations.

This repository contains the code to run the Perceptual Sampling Task (PST) benchmark on LLMs and compare their performance across different model sizes and architectures.

## Repository Structure

```
├── agents/                  # Agent implementations
│   ├── LLMagent.py          # Base LLM agent interface
│   └── ...
├── tasks/                   # Task implementations
│   ├── VSTtask.py           # Perceptual Sampling Task implementation
│   └── ...
├── manager/                 # Task and benchmark management
│   ├── TaskManager.py       # Manages running tasks and benchmarks
│   └── ...
├── benchmarks_plots/        # Output directory for benchmark plots
├── logs/                    # Logs organized by agent name
│   ├── agent1_name/         # JSON results for agent1
│   ├── agent2_name/         # JSON results for agent2
│   └── ...
├── main.ipynb               # Entry point notebook
├── README.md                # This file
└── requirements.txt         # Dependencies
```

## Benchmark Structure

The g1Bbon benchmark is organized in a hierarchical structure:

1. **Simulations**: The outermost loop, representing independent benchmark runs
2. **Trials**: Multiple iterations of a task within a simulation
3. **Rounds**: Individual observation rounds within a trial
4. **Quadrants**: The number of spatial regions for sampling
5. **Cues**: Observable signals in each quadrant

This hierarchical structure allows for more robust evaluation by:
- Providing multiple trials with the same agent to test consistency
- Maintaining conversation context across trials to test memory/learning
- Aggregating metrics across multiple levels for reliable comparisons

## Components

### 1. Perceptual Sampling Task (PST)

The Perceptual Sampling Task is a multi-armed bandit problem designed to test an agent's ability to infer hidden properties from limited observations:

- The task presents multiple quadrants (2-4), each with cues (typically 1 per quadrant).
- One quadrant is biased (90% one color, 10% the other), while others have a 50/50 distribution.
- The agent must sample different cues across a fixed number of rounds.
- After all rounds, the agent must identify which quadrant had the biased distribution.

This tests the agent's ability to:
- Balance exploration vs. exploitation
- Maintain belief updates based on observations
- Make inferences about hidden properties

### 2. LLM Agent

The `LLMagent` class provides a unified interface for different types of LLMs:

- Supports local models via Hugging Face transformers
- Supports unsloth-optimized models
- Supports API-based models (OpenAI, Anthropic)
- Handles reasoning-specialized models with internal chain-of-thought

For reasoning models, it:
- Enables internal chain-of-thought reasoning
- Tracks thinking time for performance analysis
- Captures thinking tokens for analysis
- Sets minimum and maximum thinking durations

### 3. Task Manager

The `TaskManager` class orchestrates running the tasks and benchmarks:

- Manages the hierarchical benchmark structure (simulations, trials, rounds)
- Handles conversation history and prompt construction across trials
- Performs comprehensive benchmarking across models/configurations
- Records detailed timing metrics at all levels (total, trial, round, thinking)
- Generates visualizations and exports results
- Organizes results in a structured folder hierarchy

## Supported Models

The benchmark supports various model types:

### Local Models
- Deepseek: R1-1B-Qwen, R1-7B-Qwen, R1-8B-Llama
- Qwen: 0.5B, 1.5B, 3B, 7B (base and instruct versions)
- Mistral: 7B, 7B-Instruct
- Phi-mini: 2B-Instruct
- Gemma: 2B, 2B-Instruct
- Centaur: 8B

### API Models
- OpenAI: GPT-4o, GPT-4o-mini, o1-mini
- Anthropic: Claude 3.5 Sonnet, Claude 3.5 Haiku

## Benchmark Metrics

The benchmark collects the following metrics at different levels:

- **Success Rate**: Percentage of correctly identified biased quadrants
- **Time Metrics**:
  - **Total Time**: Overall time for all simulations
  - **Trial Time**: Time per trial (including multiple rounds)
  - **Round Time**: Time per individual round
  - **Thinking Time**: For reasoning models, time spent in internal thinking
- **Quadrant Distribution**: Analysis of which quadrants were chosen
- **Thinking Tokens**: For reasoning models, the internal reasoning process

## Result Format

Benchmark results are saved in JSON format with the following structure:

```json
{
  "metrics": {
    "agent": "MODEL_NAME",
    "timestamp": "20240225_123456",
    "is_reasoning_model": false,
    "n_simulations": 10,
    "n_trials": 3,
    "n_rounds": 5,
    "n_quadrants": 4,
    "n_cues": 1,
    "success_rate": 0.65,
    "total_time": 3600.5,
    "avg_trial_time": 120.3,
    "std_trial_time": 10.2,
    "avg_round_time": 24.5,
    "std_round_time": 3.1,
    "avg_thinking_time": null,
    "std_thinking_time": null,
    "reasoning_mode": null,
    "min_thinking_time": null,
    "max_thinking_time": null,
    "min_thinking_tokens": null,
    "max_thinking_tokens": null,
    "quadrant_distribution": {
      "quadrant_1": {
        "times_chosen": 15,
        "times_correct": 5,
        "accuracy_when_chosen": 0.33
      },
      "quadrant_2": {
        "times_chosen": 20,
        "times_correct": 10,
        "accuracy_when_chosen": 0.5
      },
      "quadrant_3": {
        "times_chosen": 25,
        "times_correct": 15,
        "accuracy_when_chosen": 0.6
      },
      "quadrant_4": {
        "times_chosen": 30,
        "times_correct": 20,
        "accuracy_when_chosen": 0.67
      }
    }
  },
  "raw_results": [
    {
      "trials": [
        {
          "rounds": [
            {
              "available_cues": ["A", "B"],
              "choice": "A",
              "quadrant": 1,
              "result": "RED",
              "round_time": 5.2,
              "thinking_time": 0
            },
            ...
          ],
          "final_choice": "3",
          "correct_quadrant": 3,
          "success": true,
          "trial_time": 25.6,
          "round_times": [5.2, 4.8, 5.1, 5.3, 5.2],
          "thinking_times": []
        },
        ...
      ],
      "avg_trial_time": 26.3,
      "std_trial_time": 2.1,
      "success_rate": 0.67
    },
    ...
  ]
}
```

## Output Structure

Results are organized in the following directory structure:

- **benchmarks_plots/**: Contains all benchmark plots, with one plot per configuration
- **logs/**: Contains agent-specific folders with JSON results
  - **logs/agent_name/**: Contains JSON files for each benchmark run with this agent

## Usage

### Basic Usage

To run a benchmark with a single model:

```python
from manager.TaskManager import TaskManager

# Create a task manager with specific configuration
manager = TaskManager(
    agents=["Deepseek_R1_7B_Qwen"],
    rounds=[4, 6],
    quadrants=[2, 4],
    n_simulations=10,
    n_trials=3,  # Run 3 trials per simulation
    n_runs=3,
    device="cuda:0",
    verbose=False
)

# Run the benchmark
results = manager.multiple_benchmarks()

# Save the results
df = manager.save_results()
```

### Command-Line Usage

You can also run the benchmark from the command line:

```bash
python main.py --models Deepseek_R1_7B_Qwen --rounds 4 6 --quadrants 2 4 --simulations 10 --trials 3
```

### Comparing Multiple Models

To benchmark and compare multiple models:

```python
manager = TaskManager(
    agents=[
        "Deepseek_R1_1B_Qwen", 
        "Deepseek_R1_7B_Qwen",
        "gpt4o-mini"
    ],
    rounds=[4, 6, 8],
    quadrants=[2, 3, 4],
    n_simulations=10,
    n_trials=3,
    n_runs=3,
    device="cuda:0",
    openai_api_key="YOUR_API_KEY",  # Only needed for API models
    verbose=False
)

# Run benchmarks
results = manager.multiple_benchmarks()

# Save and visualize results
df = manager.save_results()
manager.plot_results()
```

## Time Measurement Details

The benchmark measures time at multiple levels:

1. **Total Time**: Total wall-clock time for all simulations
2. **Trial Time**: Time to complete a full trial (all rounds)
3. **Round Time**: Time to complete a single round
4. **Thinking Time**: For reasoning models, the time spent in the internal thinking phase

For reasoning models, the thinking phase can be controlled by:
- **Time-based reasoning**: Set minimum and maximum thinking times
- **Token-based reasoning**: Set minimum and maximum token counts

## Advanced Analysis

The benchmark supports advanced analysis including:

- Thinking time correlation with performance
- Per-round analysis of model behavior
- Examination of thinking tokens for reasoning models
- Cross-trial learning analysis
- Integration with model introspection tools

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- Unsloth (for optimized models)
- CUDA-capable GPU (recommended)

See `requirements.txt` for the full list of dependencies.

## Citation

If you use this benchmark in your research, please cite:

```
@misc{g1bbon2024,
  author = {Your Name},
  title = {G1Bbon: A Benchmark for Large Language Model Perceptual Inference},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub Repository},
  howpublished = {\url{https://github.com/yourusername/g1bbon}}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

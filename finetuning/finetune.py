import os
import argparse
from datasets import Dataset

from temporal_reasoning.data_processing import GameLogProcessor, TemporalDatasetPreparator
from temporal_reasoning.prompt_builder import TemporalPromptBuilder
from temporal_reasoning.model_trainer import ModelTrainer


def prepare_dataset(logs_folder: str) -> Dataset:
    """
    Prepare the dataset from game logs.
    
    Args:
        logs_folder (str): Path to folder containing game logs.
    
    Returns:
        Hugging Face Dataset
    """
    # Load game logs
    game_logs = GameLogProcessor.load_game_logs(logs_folder)
    print(f"Loaded {len(game_logs)} game logs.")

    # Prepare dataset
    temporal_data = TemporalDatasetPreparator.prepare_dataset(
        game_logs, 
        prompt_builder_cls=TemporalPromptBuilder
    )
    print(f"Prepared {len(temporal_data)} training examples.")

    # Create Hugging Face Dataset
    dataset = Dataset.from_dict({
        "prompt": [ex["prompt"] for ex in temporal_data],
        "answer": [ex["answer"] for ex in temporal_data]
    })
    
    return dataset


def main():
    """
    Main entry point for fine-tuning the temporal reasoning model.
    """
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Fine-tune a temporal reasoning model"
    )
    
    # Add arguments
    parser.add_argument(
        "--logs_folder", 
        type=str, 
        default="./logs", 
        help="Path to folder containing game logs"
    )
    parser.add_argument(
        "--model_name", 
        type=str, 
        default="Qwen/Qwen2.5-3B-Instruct", 
        help="Hugging Face model name to fine-tune"
    )
    parser.add_argument(
        "--max_steps", 
        type=int, 
        default=250, 
        help="Number of training steps"
    )
    parser.add_argument(
        "--learning_rate", 
        type=float, 
        default=5e-6, 
        help="Learning rate for training"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="outputs", 
        help="Directory to save model outputs"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Validate logs folder
    if not os.path.exists(args.logs_folder):
        raise ValueError(f"Logs folder does not exist: {args.logs_folder}")
    
    # Print training configuration
    print("Training Configuration:")
    print(f"Logs Folder: {args.logs_folder}")
    print(f"Model Name: {args.model_name}")
    print(f"Max Steps: {args.max_steps}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"Output Directory: {args.output_dir}")
    
    # Prepare dataset
    dataset = prepare_dataset(args.logs_folder)
    
    # Run training
    ModelTrainer.train(
        training_data=dataset,
        model_name=args.model_name,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
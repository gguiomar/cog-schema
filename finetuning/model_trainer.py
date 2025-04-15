import os
import torch
from typing import Optional, Tuple, List

from unsloth import FastLanguageModel, is_bfloat16_supported
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer


class ModelTrainer:
    """
    Handles model initialization, dataset preparation, and training for temporal reasoning.
    """
    @staticmethod
    def initialize_model(
        model_name: str = "Qwen/Qwen2.5-3B-Instruct", 
        max_seq_length: int = 1024, 
        lora_rank: int = 128,
        gpu_memory_utilization: float = 0.5
    ) -> Tuple[torch.nn.Module, object]:
        """
        Initialize the model for fine-tuning.
        
        Args:
            model_name (str): Hugging Face model name.
            max_seq_length (int): Maximum sequence length.
            lora_rank (int): LoRA rank.
            gpu_memory_utilization (float): Fraction of GPU memory to use.
        
        Returns:
            Tuple of (model, tokenizer)
        """
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = model_name,
            max_seq_length = max_seq_length,
            load_in_4bit = True,
            fast_inference = True,
            max_lora_rank = lora_rank,
            gpu_memory_utilization = gpu_memory_utilization,
        )

        model = FastLanguageModel.get_peft_model(
            model,
            r = lora_rank,
            target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            lora_alpha = lora_rank,
            use_gradient_checkpointing = "unsloth",
            random_state = 3407,
        )
        
        return model, tokenizer

    @staticmethod
    def prepare_training_configuration(
        learning_rate: float = 5e-6,
        max_steps: int = 250,
        output_dir: str = "outputs"
    ) -> GRPOConfig:
        """
        Prepare training configuration.
        
        Args:
            learning_rate (float): Learning rate for training.
            max_steps (int): Maximum training steps.
            output_dir (str): Directory to save outputs.
        
        Returns:
            GRPOConfig object
        """
        return GRPOConfig(
            use_vllm = True,
            learning_rate = learning_rate,
            adam_beta1 = 0.9,
            adam_beta2 = 0.99,
            weight_decay = 0.1,
            warmup_ratio = 0.1,
            lr_scheduler_type = "cosine",
            optim = "adamw_8bit",
            logging_steps = 1,
            bf16 = is_bfloat16_supported(),
            fp16 = not is_bfloat16_supported(),
            per_device_train_batch_size = 1,
            gradient_accumulation_steps = 1,
            num_generations = 8,
            max_prompt_length = 256,
            max_completion_length = 200,
            max_steps = max_steps,
            save_steps = max_steps,
            max_grad_norm = 0.1,
            report_to = "none",
            output_dir = output_dir,
        )

    @classmethod
    def train(
        cls, 
        training_data: Dataset,
        model_name: str = "Qwen/Qwen2.5-3B-Instruct",
        max_steps: int = 250,
        learning_rate: float = 5e-6,
        output_dir: str = "outputs",
        reward_funcs: Optional[List[callable]] = None
    ):
        """
        Full training pipeline.
        
        Args:
            training_data (Dataset): Hugging Face Dataset for training.
            model_name (str): Hugging Face model name to fine-tune.
            max_steps (int): Number of training steps.
            learning_rate (float): Learning rate for training.
            output_dir (str): Directory to save outputs.
            reward_funcs (Optional[List[callable]]): List of reward functions.
        
        Returns:
            Trained model
        """
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Initialize model and tokenizer
        model, tokenizer = cls.initialize_model(model_name)

        # Prepare training configuration
        training_args = cls.prepare_training_configuration(
            learning_rate=learning_rate, 
            max_steps=max_steps,
            output_dir=output_dir
        )

        # Default reward functions if not provided
        if reward_funcs is None:
            from .reward_functions import RewardFunctions
            reward_funcs = [
                RewardFunctions.xmlcount_reward_func,
                RewardFunctions.soft_format_reward_func,
                RewardFunctions.strict_format_reward_func,
                RewardFunctions.int_reward_func,
                RewardFunctions.correctness_reward_func,
            ]

        # Initialize trainer
        trainer = GRPOTrainer(
            model = model,
            processing_class = tokenizer,
            reward_funcs = reward_funcs,
            args = training_args,
            train_dataset = training_data,
        )

        # Start training
        trainer.train()

        # Save the model and tokenizer
        save_path = os.path.join(output_dir, "temporal_reasoning_model")
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)

        print(f"Model saved to {save_path}")
        return model
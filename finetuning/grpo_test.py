from unsloth import FastLanguageModel, is_bfloat16_supported
import torch
max_seq_length = 1024
lora_rank = 128
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "Qwen/Qwen2.5-3B-Instruct",
    max_seq_length = max_seq_length,
    load_in_4bit = True, # False for LoRA 16bit
    fast_inference = True, # Enable vLLM fast inference
    max_lora_rank = lora_rank,
    gpu_memory_utilization = 0.5, # Reduce if out of memory
)

model = FastLanguageModel.get_peft_model(
    model,
    r = lora_rank,
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ], # Remove QKVO if out of memory
    lora_alpha = lora_rank,
    use_gradient_checkpointing = "unsloth", # Enable long context finetuning
    random_state = 3407,
)

# #FOR CUSTOM PART-reward can be made better
import json
from datasets import load_dataset, Dataset

def parse_game_log(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    normalized_choices = []
    for choice in data.get("choices", []):
        # Use 'cue_name' if available, else 'choice'
        choice_key = choice.get("cue_name") or choice.get("choice")
        normalized_choices.append({
            "round": choice.get("round"),
            "quadrant": choice.get("quadrant"),
            "choice": choice_key,
            "color": choice.get("color"),
            "timestamp": choice.get("timestamp") or choice.get("client_timestamp")
        })
    final_choice = data.get("final_choice") or {}
    success = data.get("success") if data.get("success") is not None else final_choice.get("correct")
    return {
        "game_id": data.get("game_id"),
        "start_time": data.get("start_time"),
        "choices": normalized_choices,
        "final_choice": final_choice,
        "completion_time": data.get("completion_time"),
        "total_duration": data.get("total_duration"),
        "success": success
    }

def load_game_logs(folder_path):
    logs = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)
            logs.append(parse_game_log(file_path))
    return logs

game_logs = load_game_logs("./logs")
print(f"Loaded {len(game_logs)} game logs.")

# </reasoning>
# <answer>
# ...
# </answer>""" <-- too simple?

# SYSTEM_PROMPT = """Respond in the following format:

# <reasoning>
# ...
# import re
# from datasets import Dataset

#=============================================================#

# SYSTEM_PROMPT = """
# You are a temporal reasoning assistant.
# Follow this exact format when responding:

# <reasoning>
# Your chain-of-thought here...
# </reasoning>
# <answer>
# Your final answer here...
# </answer>

# Example:
# <reasoning>
# After analyzing the game log, quadrant 2 had the highest ratio of RED cues.
# </reasoning>
# <answer>
# 2
# </answer>  <--- might confuse?
# """

SYSTEM_PROMPT = """
You are a temporal reasoning assistant. When answering, follow the exact format below.

Example:
<reasoning>
I thought about the sequence of rounds and noted that quadrant [ANSWER] was the biased one.
</reasoning>
<answer>
[ANSWER]
</answer>

Now, given the game log, provide your chain-of-thought within <reasoning> tags and your final answer (the winning quadrant) within <answer> tags.
"""

def build_temporal_prompt(log):
    history_lines = []
    for idx, choice in enumerate(log["choices"]):
        # Get the choice value (defaulting to '?' if missing)
        choice_val = choice.get("choice", "?")
        color_val = choice.get("color", "?")
        quad = choice.get("quadrant")
        # If quadrant is not provided, set to 'unknown'; otherwise add 1 to convert 0-index.
        quadrant_str = f"{quad+1}" if quad is not None else "unknown"
        line = f"Round {idx+1}: Chose {choice_val} (Color: {color_val}) from quadrant {quadrant_str}"
        history_lines.append(line)
    history = "\n".join(history_lines)
    prompt = f"{SYSTEM_PROMPT}\nGame Log:\n{history}\nBased on the above rounds, which quadrant was the biased one?"
    return prompt


def prepare_temporal_dataset():
    examples = []
    for log in game_logs:
        prompt = build_temporal_prompt(log)
        # Use final_choice's "chosen_quadrant" if present; otherwise, default to "?".
        answer = str(log["final_choice"].get("chosen_quadrant", "?"))
        examples.append({"prompt": prompt, "answer": answer})
    return examples

temporal_data = prepare_temporal_dataset()
print(f"Prepared {len(temporal_data)} training examples.")


# A helper to extract the answer from the model output.
def extract_xml_answer(text: str) -> str:
    # Use regex to capture the content between <answer> and </answer>
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.DOTALL)
    return match.group(1).strip() if match else ""

# Function to create a Hugging Face Dataset from your temporal data.
def get_temporal_dataset() -> Dataset:
    # Assumes that `temporal_data` is a list of dicts with keys "prompt" and "answer"
    return Dataset.from_dict({
        "prompt": [ex["prompt"] for ex in temporal_data],
        "answer": [ex["answer"] for ex in temporal_data]
    })

dataset = get_temporal_dataset()

# Reward Functions-
def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    # Assume each completion is a list with a dictionary that contains the key 'content'
    responses = [completion for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    print("AAASS",answer)
    print("-" * 20)
    print("Question:", prompts[0])
    print("Expected:", answer[0])
    print("Response:", responses[0])
    print("Extracted:", extracted_responses[0])
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]

def int_reward_func(completions, **kwargs) -> list[float]:
    responses = [completion for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]

def strict_format_reward_func(completions, **kwargs) -> list[float]:
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def soft_format_reward_func(completions, **kwargs) -> list[float]:
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]


def count_xml(text: str) -> float:
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1]) * 0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1) * 0.001
    return count

def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    # Assume completions is a list of strings
    contents = [completion for completion in completions]
    return [count_xml(c) for c in contents]


from trl import GRPOConfig, GRPOTrainer
training_args = GRPOConfig(
    use_vllm = True, # use vLLM for fast inference!
    learning_rate = 5e-6,
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
    gradient_accumulation_steps = 1, # Increase to 4 for smoother training
    num_generations = 8, # Decrease if out of memory
    max_prompt_length = 256,
    max_completion_length = 200,
    # num_train_epochs = 1, # Set to 1 for a full training run
    max_steps = 250,
    save_steps = 250,
    max_grad_norm = 0.1,
    report_to = "none", # Can use Weights & Biases
    output_dir = "outputs",
)


trainer = GRPOTrainer(
    model = model,
    processing_class = tokenizer,
    reward_funcs = [
        xmlcount_reward_func,
        soft_format_reward_func,
        strict_format_reward_func,
        int_reward_func,
        correctness_reward_func,
    ],
    args = training_args,
    train_dataset = dataset,
)
trainer.train()
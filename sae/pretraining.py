import os
from hooks import Hook
from datetime import datetime
from datasets import Dataset, load_dataset
import torch

def pretrain(agent_name: str, activation_layer, dataset_path, num_tokens, context_size=1024):
    from agents.LLMagent import LLMagent
    agent = LLMagent(
        model_name=agent_name,
        use_unsloth=True,
        device_map='cuda:0',
    )

    path_parts = activation_layer.split('.')
    layer = agent.model
    # Get the model component from the input string
    for part in path_parts:
        if '[' in part and ']' in part:
            list_name, index = part.split('[')
            index = int(index[:-1])
            layer = getattr(layer, list_name)[index]
        else:
            layer = getattr(layer, part)

    # Create the directory for saving activations
    path = os.path.join("./pretraining/activations", agent_name, f"{'_'.join(path_parts)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"Saving activations to {path}")

    hook = Hook(layer, save_path=path)

    dataset = iter(load_dataset(dataset_path, split='train', streaming=True))

    sample = next(dataset)
    if "tokens" in sample:
        tokens_column = "tokens"
    elif "input_ids" in sample:
        tokens_column = "input_ids"
    elif "text" in sample:
        tokens_column = "text"
    else:
        raise ValueError("Dataset must have a 'tokens', 'input_ids', or 'text' column.")

    num_runs = num_tokens // context_size

    for run in range(num_runs):
        print("Run", run)
        all_tokens = torch.empty(0, device=agent.model.device)
        while len(all_tokens) < context_size:
            batch = next(dataset)
            if tokens_column == "text":
                tokens = agent.tokenizer.encode(batch["text"], return_tensors="pt").to(agent.model.device)
            else:
                tokens = batch[tokens_column]
            tokens = tokens.view(-1)
            all_tokens = torch.cat((all_tokens, tokens))
        token_tensor = torch.tensor(all_tokens, dtype=torch.long, device=agent.model.device)[:context_size]
        # Generate tokens by passing directly the tokenized input
        agent.model.generate(input_ids=token_tensor.view(1,-1), max_new_tokens=50, do_sample=True, temperature=1.0)


pretrain('Qwen_0.5B', 'model.layers[-1].post_attention_layernorm', "NeelNanda/c4-10k", 1024, context_size=1024)
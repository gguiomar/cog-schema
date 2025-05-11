import os
from hooks import Hook
from datasets import Dataset, load_dataset
from train_sae import train_sparse_autoencoder
import argparse
from datetime import datetime
from SparseAutoencoder import *
from config import get_default_cfg
from dataset import SAEDataLoader
from logs import init_wandb, log_batch_wandb, save_checkpoint
from tqdm import tqdm

import sys
import os

module_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'agents'))
if module_dir not in sys.path:
    sys.path.insert(0, module_dir)

# Import the desired module
from LLMagent import LLMagent

#from ..agents.LLMagent import LLMagent

def pretrain(device, agent_name: str, activation_layer, dataset_path, num_tokens, context_size=1024):
    if device == 'cuda':
        device = 'cuda:0'
    agent = LLMagent(
        model_name=agent_name,
        use_unsloth=False, # Set permanently because unsloth does not properly store activations from registered hooks
        device_map=device,
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
    path = os.path.join("../activations", agent_name,
                        f"{'_'.join(path_parts)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    subpath = os.path.join(agent_name,
                        f"{'_'.join(path_parts)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
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

    for run in tqdm(range(num_runs), desc="Generating Activations"):
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
        agent.model.generate(input_ids=token_tensor.view(1,-1), max_new_tokens=1, do_sample=True, temperature=1.0)

    return subpath

if __name__ == "__main__":
    cfg = get_default_cfg()
    parser = argparse.ArgumentParser(description="Pretrain a Sparse Autoencoder")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the training on (e.g., 'cpu', 'cuda', 'mps')")
    parser.add_argument("--model", type=str, default="Qwen_0.5B", help="LLM to use for pretraining SAE")
    parser.add_argument("--activations_layer", type=str, default='model.layers[-1].post_attention_layernorm', help="Model layer from which to store activations)")
    parser.add_argument("--sae_type", type=str, default="topk", choices=["vanilla", "topk", "batchtopk", "jumprelu"], help="Type of Sparse Autoencoder to use")
    parser.add_argument("--dataset", type=str, default="NeelNanda/c4-10k", help="Huggingface dataset on which to pretrain")
    parser.add_argument("-wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_name", type=str, default=None, help="Weights & Biases run name")
    parser.add_argument("--num_tokens", type = int, default=1024, help="Number of tokens to pretrain on")
    parser.add_argument("--context_size", type=str, default=512, help="Context size for pretraining")

    args = parser.parse_args()

    cfg["data"] = pretrain(args.device, args.model, args.activations_layer, args.dataset, args.num_tokens, args.context_size)
    train_loader = SAEDataLoader(cfg["data"], batch_size=cfg["batch_size"], shuffle=True)
    cfg["act_size"] = train_loader.get_activation_dim()
    cfg["device"] = args.device
    cfg["sae_type"] = args.sae_type
    cfg["dict_size"] = cfg["act_size"] * 16

    wandb_cfg = {
        "project": "SAE training",
        "name": args.wandb_name if args.wandb_name is not None else f"{cfg['model']}_{cfg['sae_type']}_{cfg['dataset']}_{cfg['num_tokens']}_{cfg['context_size']},{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}",
        "save_interval": 50,
        "log_batch_interval": 10,
    }
    if args.wandb:
        wandb_run = init_wandb(wandb_cfg, cfg=cfg)

    # Initialize model
    if cfg["sae_type"] == "vanilla":
        model = VanillaSAE(cfg)
    elif cfg["sae_type"] == "topk":
        model = TopKSAE(cfg)
    elif cfg["sae_type"] == "batchtopk":
        model = BatchTopKSAE(cfg)
    elif cfg["sae_type"] == 'jumprelu':
        model = JumpReLUSAE(cfg)

    print(f"Running training with model: {cfg['sae_type']} with {sum(p.numel() for p in model.parameters())} parameters")
    train_sparse_autoencoder(model, train_loader, cfg, wandb_cfg, wandb_run if args.wandb else None)

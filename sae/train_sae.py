import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm, trange
import argparse
from datetime import datetime

from SparseAutoencoder import *
from config import get_default_cfg
from dataset import SAEDataLoader
from logs import init_wandb, log_batch_wandb, save_checkpoint

def train_sparse_autoencoder(model, data_loader, cfg, wandb_cfg, wandb_run = None):
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"], betas=(cfg["beta1"], cfg["beta2"]))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2)

    epoch_range = trange(cfg["num_epochs"], desc="Training", unit="epoch")

    for (i, epoch) in enumerate(epoch_range):
        model.train()
        total_loss = 0.0

        pbar = tqdm(data_loader, desc=f"Epoch {epoch+1}", unit="batch", leave=False)

        for (b, batch) in enumerate(pbar):
            batch = batch.to(cfg["device"])
            optimizer.zero_grad()
            sae_output = model(batch)

            loss = sae_output["loss"]
            pbar.set_postfix({"Loss": f"{loss.item():.4f}", "L0": f"{sae_output['l0_norm']:.4f}", "L2": f"{sae_output['l2_loss']:.4f}", "L1": f"{sae_output['l1_loss']:.4f}", "L1_norm": f"{sae_output['l1_norm']:.4f}"})
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["max_grad_norm"])
            model.make_decoder_weights_and_grad_unit_norm()

            if wandb_run is not None and b % wandb_cfg["log_batch_interval"] == 0:
                log_batch_wandb(sae_output, wandb_run)

            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(data_loader)
        epoch_range.set_postfix({"Average loss": f"{avg_loss:.4f}"})

        if wandb_run is not None and i % wandb_cfg["save_interval"] == 0 and i != 0:
            save_checkpoint(wandb_run, model, optimizer, scheduler, cfg, wandb_cfg, i)

        scheduler.step(avg_loss)
    if wandb_run is not None:
        save_checkpoint(wandb_run, model, optimizer, scheduler, cfg, wandb_cfg, len(epoch_range)-1)
        wandb_run.finish()

if __name__ == "__main__":
    cfg = get_default_cfg()
    parser = argparse.ArgumentParser(description="Train a Sparse Autoencoder")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the training on (e.g., 'cpu', 'cuda', 'mps')")
    parser.add_argument("--data", type=str, help="Directory containing activation data", default='../activations/Qwen_0.5B/model_layers[0]_post_attention_layernorm/20250511_142949')
    parser.add_argument("--sae_type", type=str, default="topk", choices=["vanilla", "topk", "batchtopk", "jumprelu"], help="Type of Sparse Autoencoder to use")
    parser.add_argument("-wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_name", type=str, default=None, help="Weights & Biases run name")

    args = parser.parse_args()

    cfg["device"] = args.device
    cfg["data"] = args.data
    cfg["sae_type"] = args.sae_type

    train_loader = SAEDataLoader(cfg["data"], batch_size=cfg["batch_size"], shuffle=True)

    cfg["act_size"] = train_loader.get_activation_dim()
    cfg["dict_size"] = cfg["act_size"] * 16

    wandb_cfg = {
        "project": "SAE training",
        "name": args.wandb_name if args.wandb_name is not None else f"{cfg['sae_type']}_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}",
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

    # Train model
    train_sparse_autoencoder(model, train_loader, cfg, wandb_cfg, wandb_run if args.wandb else None)
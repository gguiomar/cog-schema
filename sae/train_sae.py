import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm, trange

from SparseAutoencoder import *
from config import get_default_cfg

def train_sparse_autoencoder(model, data_loader, cfg):
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"], betas=(cfg["beta1"], cfg["beta2"]))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2)

    epoch_range = trange(cfg["num_epochs"], desc="Training", unit="epoch")

    for epoch in epoch_range:
        model.train()
        total_loss = 0.0

        pbar = tqdm(data_loader, desc=f"Epoch {epoch+1}", unit="batch", leave=False)

        for (batch,) in pbar:
            optimizer.zero_grad()
            sae_output = model(batch)

            loss = sae_output["loss"]
            pbar.set_postfix({"Loss": f"{loss.item():.4f}", "L0": f"{sae_output['l0_norm']:.4f}", "L2": f"{sae_output['l2_loss']:.4f}", "L1": f"{sae_output['l1_loss']:.4f}", "L1_norm": f"{sae_output['l1_norm']:.4f}"})
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["max_grad_norm"])
            model.make_decoder_weights_and_grad_unit_norm()

            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(data_loader)
        epoch_range.set_postfix({"Average loss": f"{avg_loss:.4f}"})

        scheduler.step(avg_loss)

if __name__ == "__main__":
    cfg = get_default_cfg()
    cfg["device"] = "mps"

    # Generate synthetic data (replace with actual dataset)
    x_train = np.random.rand(10000, cfg["act_size"]).astype(np.float32)
    train_dataset = TensorDataset(torch.tensor(x_train).to(cfg["device"]))
    train_loader = DataLoader(train_dataset, batch_size=cfg["batch_size"], shuffle=True)

    # Initialize model
    if cfg["sae_type"] == "vanilla":
        model = VanillaSAE(cfg)
    elif cfg["sae_type"] == "topk":
        model = TopKSAE(cfg)
    elif cfg["sae_type"] == "batchtopk":
        model = BatchTopKSAE(cfg)
    elif cfg["sae_type"] == 'jumprelu':
        model = JumpReLUSAE(cfg)

    # Train model
    train_sparse_autoencoder(model, train_loader, cfg)
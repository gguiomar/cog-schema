import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

import torch.nn as nn
import torch.optim as optim

def train_sparse_autoencoder(model, data_loader, num_epochs=20, learning_rate=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, betas=(0.9,0.999))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2)
    # (Optionally, set up weight EMA here)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:          # train_loader yields [B, D] activations
            x = batch.to(device)
            optimizer.zero_grad()
            x_recon, z = model(x)           # forward
            loss = sae_loss(x, x_recon, z, l1_coeff)
            loss.backward()
            optimizer.step()
            # Renormalize decoder weights columns to unit norm:
            with torch.no_grad():
                w = model.decoder.weight.data   # shape [D, F]
                norm = w.norm(dim=0, keepdim=True) + 1e-6
                model.decoder.weight.data.div_(norm)
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        # Validate, adjust lr, early stop, etc.
        scheduler.step(avg_loss)
        print(f"Epoch {epoch}: loss={avg_loss:.4f}")

if __name__ == "__main__":
    # Hyperparameters
    input_dim = 784  # Example for MNIST dataset
    hidden_dim = 64
    batch_size = 128
    num_epochs = 20
    learning_rate = 1e-3
    device = "cuda"

    # Generate synthetic data (replace with actual dataset)
    x_train = np.random.rand(10000, input_dim).astype(np.float32)
    train_dataset = TensorDataset(torch.tensor(x_train), torch.tensor(x_train))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Initialize model
    model = SparseAutoencoder(input_dim=input_dim, hidden_dim=hidden_dim)

    # Train model
    train_sparse_autoencoder(model, train_loader, num_epochs=num_epochs, learning_rate=learning_rate)
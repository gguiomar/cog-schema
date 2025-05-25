import os
import torch
from SparseAutoencoder import BatchTopKSAE
import json

# Define the directory path and file name.
save_dir = '../classical_conditioning_qwen_0.5b_instruct_layer23_1199'

model_path = os.path.join(save_dir, "sae.pt")
config_path = os.path.join(save_dir, "config.json")

with open(config_path, "r") as f:
    cfg = json.load(f)

# Load the model; adjust map_location as needed (e.g., 'cuda' or 'cpu')
model = torch.load(model_path, map_location='mps')

print(f'VAE loaded from: {model_path}')

VAE = BatchTopKSAE(cfg)


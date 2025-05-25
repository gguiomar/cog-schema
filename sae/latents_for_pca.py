import os
import torch
from SparseAutoencoder import BatchTopKSAE
import json

# Define the directory path and file name.
save_dir = 'checkpoints/classical_conditioning_qwen_0.5b_instruct_layer23_1199$'

model_path = os.path.join(save_dir, "model.pt")
config_path = os.path.join(save_dir, "config.json")

with open(config_path, "r") as f:
    cfg = json.load(config_path)

# Load the model; adjust map_location as needed (e.g., 'cuda' or 'cpu')
model = torch.load(model_path, map_location='cuda')

print(f'VAE loaded from: {model_path}')

VAE = BatchTopKSAE(cfg)

VAE.load_state_dict(model)
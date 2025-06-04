import numpy as np
import os
from tqdm import tqdm
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def organized_activations(num_trials, num_rounds, activations_base_path="activations/Qwen_0.5B"):
    """
    Load saved activations and organize them in a dataset with shape [num_trials * (num_rounds + 1), block (layer), tokens, hidden_dimension].
    """
    
    # Calculate total number of files
    total_files = num_trials * (num_rounds + 1)
    layer_dirs = []
    for item in sorted(os.listdir(activations_base_path)):
        if item.startswith("model_layers["):
            layer_dirs.append(item)
    num_layers = len(layer_dirs)
    
    # Load the last file to extract dimensions
    last_file_idx = total_files - 1
    first_layer_path = os.path.join(activations_base_path, layer_dirs[0])
    timestamp_dirs = [d for d in os.listdir(first_layer_path) if os.path.isdir(os.path.join(first_layer_path, d))]
    timestamp_dir = timestamp_dirs[0]
    last_file_path = os.path.join(activations_base_path, layer_dirs[0], timestamp_dir, f"activations_{last_file_idx}.npy")
    last_activation = np.load(last_file_path)
    max_tokens, hidden_dim = last_activation.shape[1], last_activation.shape[2]  # tokens and hidden dimension
    print(f"Dimensions - Max tokens: {max_tokens}, Hidden dim: {hidden_dim}, Layers: {num_layers}")
    
    # Create the target array
    target_shape = (total_files, num_layers, max_tokens, hidden_dim)
    organized_activations = np.zeros(target_shape, dtype=np.float32)
    print(f"Created target array with shape: {target_shape}")
    
    # Fill the array
    print("Loading and organizing activations...")
    for file_idx in tqdm(range(total_files), desc="Processing files"):
        for layer_idx, layer_dir in enumerate(layer_dirs):
            file_path = os.path.join(activations_base_path, layer_dir, timestamp_dir, f"activations_{file_idx}.npy")
            activation_data = np.load(file_path)  # Shape: [1, tokens, hidden_dim]
            activation_data = activation_data[0]  # Shape: [tokens, hidden_dim]
            
            # Handle variable token lengths (padding)
            actual_tokens = activation_data.shape[0]
            if actual_tokens <= max_tokens:
                organized_activations[file_idx, layer_idx, :actual_tokens, :] = activation_data
    
    print(f"Successfully organized activations with final shape: {organized_activations.shape}")
    return organized_activations

def to_pca(organized_activations):
    """
    Perform PCA on the organized activations.
    """
    num_trials, num_layers, max_tokens, hidden_dim = organized_activations.shape
    reshaped_activations = organized_activations.reshape(num_trials * num_layers * max_tokens, hidden_dim)
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(reshaped_activations)
    pca_reshaped = pca_result.reshape(num_trials, num_layers, max_tokens, -1)
    print(f"PCA result shape: {pca_reshaped.shape}")
    return pca_reshaped 

def plot_traj(pca_reshaped, num_rounds, num_tokens, layer_idx=0, zero_tol=1e-6):
    """
    Plot trajectories of each token across rounds in 2D PCA space for a specified layer,
    skipping padding tokens where PC1 and PC2 are both essentially zero.
    """
    plt.figure(figsize=(8, 6))
    for token_idx in range(num_tokens):
        token_coords = []
        for round_idx in range(num_rounds + 1):
            file_idx = round_idx  # assuming trial 0 is indexed 0..num_rounds
            coord = pca_reshaped[file_idx, layer_idx, token_idx, :2]
            if np.all(np.abs(coord) <= zero_tol):
                print(f"Skipping token {token_idx} at round {round_idx} due to zero coordinates.")
                continue
            token_coords.append(coord)
        
        token_coords = np.stack(token_coords, axis=0)
        x_vals = token_coords[:, 0]
        y_vals = token_coords[:, 1]
        plt.plot(x_vals, y_vals, marker='o', label=f'Token {token_idx}')
    
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title(f'Token trajectories across rounds in PCA space (Layer {layer_idx})')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

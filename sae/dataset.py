import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
from tqdm import tqdm

class TokenActivationsDataset(Dataset):
    def __init__(self, activations_dir):
        """
        Custom dataset to load token activations from numpy files.
        Args:
            activations_dir (str): Path to the directory containing numpy files.
        """
        self.activations = []
        for root, _, files in os.walk(activations_dir):
            for file in files:
                if file.endswith(".npy"):
                    file_path = os.path.join(root, file)
                    data = np.load(file_path)  # Shape: batch x tokens x hidden_dim
                    # Reshape to (batch * tokens, hidden_dim)
                    reshaped_data = data.reshape(-1, data.shape[-1])
                    self.activations.append(reshaped_data)
        
        # Concatenate all activations into a single array
        self.activations = np.concatenate(self.activations, axis=0)  # Shape: (total_tokens, hidden_dim)

        print(f"Loaded {len(self.activations)} token activations.")

    def __len__(self):
        return len(self.activations)

    def __getitem__(self, idx):
        return torch.tensor(self.activations[idx], dtype=torch.float32)


class SAEDataLoader(DataLoader):
    def __init__(self, activations_dir, batch_size=4096, shuffle=True, num_workers=0):
        """
        Custom DataLoader for loading token activations.
        Args:
            activations_dir (str): Path to the directory containing numpy files.
            batch_size (int): Number of samples per batch.
            shuffle (bool): Whether to shuffle the data.
            num_workers (int): Number of worker threads for data loading.
        """
        dataset = TokenActivationsDataset(f"../activations/{activations_dir}")
        super(SAEDataLoader, self).__init__(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    def get_activation_dim(self):
        """
        Returns the dimensionality of the token activations.
        """
        return self.dataset[0].shape[0]
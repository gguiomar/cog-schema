import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json

class Hook:
    """
    A class to manage hooks for PyTorch modules.
    """
    def __init__(self, module, save_path=None):
        self.module = module
        self.hook_fn = self.hook_to_numpy
        self.handle = module.register_forward_hook(self.hook_fn)
        self.save_path = save_path
        if save_path is not None:
            os.makedirs(save_path, exist_ok=False)
        self.activations = []
        self.current_text = None
        self.current_tokens = None

    def remove(self):
        """
        Remove the hook from the module.
        """
        self.handle.remove()

    def hook_to_numpy(self, module, input, output):
        """
        Hook to convert PyTorch tensors to NumPy arrays.
        """
        if isinstance(output, tuple):
            output = output[0]
        activations = output.detach().cpu().numpy()
        self.activations.append({
            "activations": activations,
            "text": self.current_text,
            "tokens": self.current_tokens.cpu() if self.current_tokens is not None else None,
        })

    def save_all(self):
        """Save all collected activations and metadata to disk."""
        for i, item in enumerate(self.activations):
            act_path = os.path.join(self.save_path, f"activations_{i}.npy")
            meta_path = os.path.join(self.save_path, f"meta_{i}.json")

            # Save activation
            np.save(act_path, item["activations"])

            # Save metadata
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump({
                    "text": item["text"],
                    "tokens": item["tokens"].tolist() if item["tokens"] is not None else None,
                    "file_name": f"activations_{i}.npy"
                }, f, indent=2)

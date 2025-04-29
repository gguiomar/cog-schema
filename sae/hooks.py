import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

class Hook:
    """
    A class to manage hooks for PyTorch modules.
    """
    def __init__(self, module, save_path=None):
        self.module = module
        self.hook_fn = self.hook_to_numpy
        self.handle = module.register_forward_hook(self.hook_fn)
        self.counter = 0
        self.save_path = save_path
        if save_path is not None:
            os.makedirs(save_path, exist_ok=False)

    def remove(self):
        """
        Remove the hook from the module.
        """
        self.handle.remove()

    def hook_to_numpy(self, module, input, output):
        """
        Hook to convert PyTorch tensors to NumPy arrays.
        """
        activations = output.detach().cpu().numpy()
        if self.save_path is not None:
            np.save(f"{self.save_path}/activations_{self.counter}.npy", activations)
            self.counter += 1
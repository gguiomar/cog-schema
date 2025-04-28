import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class SparseAutoencoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        expansion_factor: float = 16,
        loss_type: str = "l1",
        sparsity_weight: float = 1.0,
        k: Optional[int] = None,
        jump_threshold: Optional[float] = 0.0
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = int(input_dim * expansion_factor)
        
        self.encoder = nn.Linear(input_dim, self.latent_dim, bias=True)
        self.decoder = nn.Linear(self.latent_dim, input_dim, bias=True)

        self.loss_type = loss_type.lower()
        self.sparsity_weight = sparsity_weight
        self.k = k
        self.jump_threshold = jump_threshold

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encode_activation(self.encoder(x))
        decoded = self.decoder(encoded)
        return decoded, encoded

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.encode_activation(self.encoder(x))

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.decoder(x)

    def encode_activation(self, x: torch.Tensor) -> torch.Tensor:
        if self.loss_type == "jumprelu":
            raise NotImplementedError("Jump ReLU is not implemented yet.")
        else:
            return F.relu(x)

    def compute_loss(self, x: torch.Tensor, decoded: torch.Tensor, encoded: torch.Tensor) -> torch.Tensor:
        """Compute the full loss with dynamic weighting."""
        recon_loss = F.mse_loss(decoded, x)

        if self.loss_type == "l1":
            sparsity_loss = encoded.abs().mean()
        
        elif self.loss_type == "topk":
            sparsity_loss = self.topk_penalty(encoded, k=self.k)
        
        elif self.loss_type == "batch_topk":
            sparsity_loss = self.batch_topk_penalty(encoded, k=self.k)

        elif self.loss_type == "jumprelu":
            raise NotImplementedError("Jump ReLU is not implemented yet.")

        else:
            raise ValueError(f"Unknown loss_type {self.loss_type}")

        total_loss = recon_loss + self.sparsity_weight * sparsity_loss
        return total_loss

    @staticmethod
    def topk_penalty(encoded: torch.Tensor, k: int) -> torch.Tensor:
        """Penalizes all activations except top-k largest per sample."""
        batch_size, dim = encoded.shape
        if k >= dim:
            return torch.tensor(0.0, device=encoded.device)
        values, _ = torch.topk(encoded, k=k, dim=-1)
        topk_mask = encoded >= values[:, -1].unsqueeze(1)
        sparsity_loss = (encoded * (~topk_mask)).abs().mean()
        return sparsity_loss

    @staticmethod
    def batch_topk_penalty(encoded: torch.Tensor, k: int) -> torch.Tensor:
        """Penalizes activations globally over batch, not per-sample."""
        batch_size, dim = encoded.shape
        flat = encoded.view(-1)
        if k >= flat.numel():
            return torch.tensor(0.0, device=encoded.device)
        values, _ = torch.topk(flat, k=k)
        threshold = values[-1]
        sparsity_loss = (flat[flat < threshold]).abs().mean()
        return sparsity_loss

    @classmethod
    def from_pretrained(cls, path: str, input_dim: int, expansion_factor: float = 16, device: str = "cuda", **kwargs) -> "SparseAutoencoder":
        model = cls(input_dim=input_dim, expansion_factor=expansion_factor, **kwargs)
        state_dict = torch.load(path, map_location=device)
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()
        return model
    
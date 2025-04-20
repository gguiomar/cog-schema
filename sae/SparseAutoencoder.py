import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim: int, expansion_factor: float = 16):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = int(input_dim * expansion_factor)
        self.decoder = nn.Linear(self.latent_dim, input_dim, bias=True)
        self.encoder = nn.Linear(input_dim, self.latent_dim, bias=True)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        encoded = F.relu(self.encoder(x))
        decoded = self.decoder(encoded)
        return decoded, encoded

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return F.relu(self.encoder(x))

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.decoder(x)

    @classmethod
    def from_pretrained(cls, path: str, input_dim: int, expansion_factor: float = 16, device: str = "cuda") -> "SparseAutoencoder":
        model = cls(input_dim=input_dim, expansion_factor=expansion_factor)
        state_dict = torch.load(path, map_location=device)
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()
        return model
    
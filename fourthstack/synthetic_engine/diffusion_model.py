import torch
import torch.nn as nn
import torch
import torch.nn as nn
from typing import Optional


class DiffusionModel(nn.Module):
        """Minimal diffusion-style denoising autoencoder scaffold.

        This is a lightweight model suitable for smoke tests and small experiments.
        It applies Gaussian noise to inputs and attempts to reconstruct the original.
        """

        def __init__(self, input_dim: int = 32, hidden_dim: int = 128, dropout: float = 0.0):
                super().__init__()

                self.encoder = nn.Sequential(
                        nn.Linear(input_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.ReLU(),
                )

                self.decoder = nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                        nn.Linear(hidden_dim, input_dim),
                )

        def forward(self, x: torch.Tensor, noise_level: float = 0.1, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
                """Forward pass.

                Args:
                        x: input tensor of shape (batch, input_dim)
                        noise_level: standard deviation of additive Gaussian noise
                        noise: optional externally-provided noise tensor; if provided it's used instead

                Returns:
                        reconstructed tensor of shape (batch, input_dim)
                """
                if noise is None:
                        noise = torch.randn_like(x) * float(noise_level)
                x_noisy = x + noise

                latent = self.encoder(x_noisy)
                reconstructed = self.decoder(latent)
                return reconstructed

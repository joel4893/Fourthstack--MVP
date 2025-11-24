import torch
import torch.nn as nn
from .gan import Generator
from .transformer_block import TransformerBlock
from .diffusion_model import DiffusionModel


class HybridGenerator(nn.Module):
    """Compose a GAN generator, a transformer refinement, and a diffusion denoiser.

    This is a small orchestration wrapper intended for experiments and smoke
    testing; it is not tuned for production training.
    """

    def __init__(self, noise_dim: int = 32, data_dim: int = 32):
        super().__init__()

        # GAN generator: maps noise -> data vector
        self.gan = Generator(noise_dim=noise_dim, output_dim=data_dim)

        # Transformer expects (batch, seq_len, embed_dim) when batch_first=True
        self.transformer = TransformerBlock(embed_dim=data_dim, num_heads=4)

        # Diffusion/denoiser: maps (batch, data_dim) -> (batch, data_dim)
        self.diffusion = DiffusionModel(input_dim=data_dim)

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        """Forward: noise -> gan -> transformer -> diffusion."""
        # noise shape: (batch, noise_dim)
        gan_out = self.gan(noise)

        # transformer expects a sequence dimension; use seq_len=1 and remove it after
        trans_in = gan_out.unsqueeze(1)  # (batch, 1, data_dim)
        trans_out = self.transformer(trans_in)  # (batch, 1, data_dim)
        trans_out = trans_out.squeeze(1)  # (batch, data_dim)

        diff_out = self.diffusion(trans_out)
        return diff_out

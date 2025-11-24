import torch
import torch.nn as nn
import torch
import torch.nn as nn
from typing import Optional


class RewardModel(nn.Module):
    """Simple scalar reward scorer for RLHF-style training.

    The model maps an input feature vector to a single scalar score.
    It returns a tensor of shape `(batch,)` for a batched input.
    """

    def __init__(self, input_dim: int = 32, hidden_dim: int = 64):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, data_sample: torch.Tensor) -> torch.Tensor:
        """Compute scalar scores for `data_sample`.

        Args:
                data_sample: Tensor of shape `(batch, input_dim)` or `(input_dim,)`.

        Returns:
                Tensor of shape `(batch,)` or scalar for single input.
        """
        out = self.scorer(data_sample)
        # return shape: (batch, 1) -> squeeze to (batch,)
        return out.squeeze(-1)


__all__ = ["RewardModel"]

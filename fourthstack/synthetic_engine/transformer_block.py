import torch
import torch.nn as nn
from typing import Optional


class TransformerBlock(nn.Module):
        """Simple Transformer block: multi-head self-attention + feed-forward with residuals.

        Args:
                embed_dim: dimensionality of input embeddings
                num_heads: number of attention heads
                ff_dim: hidden size of the feed-forward network
                dropout: dropout probability applied after attention and in FFN
        """

        def __init__(self, embed_dim: int = 64, num_heads: int = 4, ff_dim: int = 256, dropout: float = 0.0):
                super().__init__()

                self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
                self.norm1 = nn.LayerNorm(embed_dim)
                self.norm2 = nn.LayerNorm(embed_dim)

                self.ff = nn.Sequential(
                        nn.Linear(embed_dim, ff_dim),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                        nn.Linear(ff_dim, embed_dim),
                        nn.Dropout(dropout),
                )

        def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
                """Forward pass.

                Expects input `x` of shape (batch, seq_len, embed_dim) when `batch_first=True`.
                Returns tensor of same shape.
                """
                # Self-attention (query, key, value all = x)
                attn_output, _ = self.attention(x, x, x, attn_mask=attn_mask)
                x = self.norm1(x + attn_output)

                # Feed-forward
                ff_output = self.ff(x)
                out = self.norm2(x + ff_output)
                return out

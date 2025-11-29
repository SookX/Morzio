import torch
import torch.nn as nn

from model.encoder.encoder import TokenSelfAttention


class Decoder(nn.Module):
    def __init__(
        self,
        output_dim: int,
        hidden_dim: int,
        latent_dim: int,
        dropout: float = 0.05,
        attn_tokens: int = 4,
        attn_heads: int = 4,
    ) -> None:
        super().__init__()
        self.latent_proj = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.attn = TokenSelfAttention(hidden_dim, attn_tokens, attn_heads, dropout)
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = self.latent_proj(z)
        z = self.attn(z)
        return self.output_proj(z)

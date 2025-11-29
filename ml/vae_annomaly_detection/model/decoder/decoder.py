import torch
import torch.nn as nn


class Decoder(nn.Module):
    """Feed-forward decoder that mirrors the encoder capacity."""

    def __init__(self, latent_dim: int, hidden_dim: int, output_dim: int, layers: int = 3, dropout: float = 0.1) -> None:
        super().__init__()
        if layers < 2:
            raise ValueError("Decoder needs at least two layers")

        dims = [latent_dim]
        for _ in range(layers - 2):
            dims.append(hidden_dim)
        dims.append(output_dim)

        blocks = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            blocks.append(nn.Linear(in_dim, out_dim))
            if out_dim != output_dim:
                blocks.append(nn.LayerNorm(out_dim))
                blocks.append(nn.GELU())
                if dropout > 0:
                    blocks.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*blocks)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)

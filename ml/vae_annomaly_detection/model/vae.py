from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn

from model.decoder.decoder import Decoder
from model.encoder.encoder import Encoder
from model.ls_head.ls import LatentSpaceHead


class VAE(nn.Module):
    """Symmetric VAE for dense tabular inputs."""

    def __init__(
        self,
        input_dim: int,
        base_dim: int = 64,
        latent_dim: int = 32,
        hidden_dims: Iterable[int] | None = None,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [base_dim, base_dim * 2, base_dim * 4]

        self.encoder = Encoder(input_dim=input_dim, hidden_dims=hidden_dims, dropout=dropout)
        self.latent = LatentSpaceHead(encoder_dim=hidden_dims[-1], latent_dim=latent_dim)
        decoder_dims = list(reversed(hidden_dims))
        self.decoder = Decoder(
            output_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dims=decoder_dims,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:  # type: ignore[override]
        encoded = self.encoder(x)
        z, mu, logvar = self.latent(encoded)
        reconstructed = self.decoder(z)
        return reconstructed, mu, logvar


if __name__ == "__main__":
    model = VAE(input_dim=20, base_dim=32, latent_dim=16)
    dummy = torch.randn(8, 20)
    recon, mu, logvar = model(dummy)
    print(recon.shape, mu.shape, logvar.shape)

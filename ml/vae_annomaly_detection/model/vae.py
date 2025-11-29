from __future__ import annotations

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
        hidden_dim: int = 64,
        latent_dim: int = 32,
        dropout: float = 0.05,
        attn_tokens: int = 4,
        attn_heads: int = 4,
    ) -> None:
        super().__init__()

        self.encoder = Encoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            attn_tokens=attn_tokens,
            attn_heads=attn_heads,
        )
        self.latent = LatentSpaceHead(encoder_dim=self.encoder.output_dim, latent_dim=latent_dim)
        self.decoder = Decoder(
            output_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            dropout=dropout,
            attn_tokens=attn_tokens,
            attn_heads=attn_heads,
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

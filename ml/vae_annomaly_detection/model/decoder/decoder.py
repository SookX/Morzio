import torch
import torch.nn as nn

from model.encoder.encoder import ResidualMLPBlock


class Decoder(nn.Module):
    def __init__(
        self,
        output_dim: int = 20,
        latent_dim: int | None = None,
        hidden_dims=None,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        if latent_dim is None:
            raise ValueError("latent_dim must be provided")
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]

        dims = [latent_dim, *hidden_dims]
        blocks = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            blocks.append(ResidualMLPBlock(in_dim, out_dim, dropout))
        self.blocks = nn.Sequential(*blocks)

        self.output_layer = nn.Linear(dims[-1], output_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.blocks(z)
        return self.output_layer(h)

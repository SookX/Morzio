import torch
import torch.nn as nn


class ResidualMLPBlock(nn.Module):
    """Two-layer MLP with residual connection and optional projection."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.05) -> None:
        super().__init__()
        hidden = max(out_dim, in_dim)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
            nn.GELU(),
        )
        self.shortcut = nn.Identity() if in_dim == out_dim else nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        return self.norm(out + self.shortcut(x))


class Encoder(nn.Module):
    def __init__(self, input_dim: int = 20, hidden_dims=None, dropout: float = 0.05) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 128, 256]
        dims = [input_dim, *hidden_dims]
        blocks = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            blocks.append(ResidualMLPBlock(in_dim, out_dim, dropout))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


if __name__ == "__main__":
    x = torch.randn((10, 20))
    encoder = Encoder(input_dim=20, hidden_dims=[64, 128, 256])
    z = encoder(x)
    print("Input shape:", x.shape)
    print("Encoded shape:", z.shape)

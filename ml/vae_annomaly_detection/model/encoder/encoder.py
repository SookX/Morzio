import torch
import torch.nn as nn


class Encoder(nn.Module):
    """Simple feed-forward encoder used by the VAE-GRF model."""

    def __init__(self, input_dim: int, hidden_dim: int, layers: int = 3, dropout: float = 0.1) -> None:
        super().__init__()
        if layers < 2:
            raise ValueError("Encoder needs at least two layers")

        dims = [input_dim]
        for _ in range(layers - 1):
            dims.append(hidden_dim)

        blocks = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            blocks.extend(
                [
                    nn.Linear(in_dim, out_dim),
                    nn.LayerNorm(out_dim),
                    nn.GELU(),
                ]
            )
            if dropout > 0:
                blocks.append(nn.Dropout(dropout))

        self.net = nn.Sequential(*blocks)
        self._output_dim = hidden_dim

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


if __name__ == "__main__":
    x = torch.randn((10, 20))
    encoder = Encoder(input_dim=20, hidden_dim=64)
    z = encoder(x)
    print("Input shape:", x.shape)
    print("Encoded shape:", z.shape)

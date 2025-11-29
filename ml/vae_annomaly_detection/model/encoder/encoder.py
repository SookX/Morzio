import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()

        expand_dim = hidden_dim * 2

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),

            nn.Linear(hidden_dim, expand_dim),
            nn.LayerNorm(expand_dim),
            nn.SiLU(),
            nn.Dropout(0.1),

            nn.Linear(expand_dim, expand_dim),
            nn.LayerNorm(expand_dim),
            nn.SiLU()
        )

        self.res_proj = nn.Linear(hidden_dim, expand_dim)

    def forward(self, x):
        out = self.net(x)

        residual = self.res_proj(self.net[0](x))

        return out + residual


if __name__ == "__main__":
    x = torch.randn((10, 20))
    encoder = Encoder(input_dim=20, hidden_dim=50, latent_dim=10)
    z = encoder(x)
    print("Input shape:", x.shape)
    print("Encoded shape:", z.shape)

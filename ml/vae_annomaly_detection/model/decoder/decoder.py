import torch
import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super().__init__()

        expand_dim = hidden_dim * 2

        self.net = nn.Sequential(
            nn.Linear(latent_dim, expand_dim),
            nn.LayerNorm(expand_dim),
            nn.SiLU(),
            nn.Dropout(0.1),

            nn.Linear(expand_dim, expand_dim),
            nn.LayerNorm(expand_dim),
            nn.SiLU(),
            nn.Dropout(0.1),

            nn.Linear(expand_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU()
        )

        self.res_proj = nn.Linear(latent_dim, hidden_dim)

        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        out = self.net(z)

        residual = self.res_proj(z)

        out = out + residual      
        out = self.fc_out(out)    

        return out


if __name__ == "__main__":
    x = torch.randn((10, 10))  
    decoder = Decoder(latent_dim=10, hidden_dim=50, output_dim=20)
    out = decoder(x)
    print("Input latent shape:", x.shape)
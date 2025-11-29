import torch
import torch.nn as nn

class LatentSpaceHead(nn.Module):
    
    def __init__(self, encoder_dim, latent_dim):
        super().__init__()
        self.fc_mu = nn.Linear(encoder_dim, latent_dim)
        self.fc_logvar = nn.Linear(encoder_dim, latent_dim)
        self.activation = nn.SiLU()

    def forward(self, x):
        x = self.activation(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        z = mu + eps * std
        return z, mu, logvar

    
if __name__ == "__main__":
    x = torch.randn((10, 20))  
    ls_head = LatentSpaceHead(latent_dim=20)
    mean, logvar = ls_head(x)
    print("Input shape:", x.shape)
    print("Mean shape:", mean.shape)
    print("Logvar shape:", logvar.shape)
    
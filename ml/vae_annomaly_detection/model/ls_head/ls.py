import torch
import torch.nn as nn

class LatentSpaceHead(nn.Module):
    def __init__(self, latent_dim):
        super(LatentSpaceHead, self).__init__()
        self.fc1_mean = nn.Linear(latent_dim, latent_dim)
        self.fc2_logvar = nn.Linear(latent_dim, latent_dim)
        self.activation = nn.SiLU()
        

    def forward(self, x):
        mu = self.fc1_mean(self.activation(x))
        logvar = self.fc2_logvar(self.activation(x))

        sigma = torch.exp(0.5*logvar)
        noise = torch.randn_like(sigma, device=sigma.device)
        
        z = mu + sigma*noise
        return z, mu, logvar
    
if __name__ == "__main__":
    x = torch.randn((10, 20))  
    ls_head = LatentSpaceHead(latent_dim=20)
    mean, logvar = ls_head(x)
    print("Input shape:", x.shape)
    print("Mean shape:", mean.shape)
    print("Logvar shape:", logvar.shape)
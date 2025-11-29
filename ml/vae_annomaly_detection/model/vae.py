import torch
import torch.nn as nn

from encoder.encoder import Encoder
from ls_head.ls import LatentSpaceHead
from decoder.decoder import Decoder

class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.ls_head = LatentSpaceHead(latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim)

    def forward(self, x):
        encoded = self.encoder(x)
        z, mu, logvar = self.ls_head(encoded)
        reconstructed = self.decoder(z)
        return reconstructed, mu, logvar
    
if __name__ == "__main__":
    x = torch.randn((10, 20))  
    vae = VAE(input_dim=20, hidden_dim=50, latent_dim=10)
    reconstructed, mu, logvar = vae(x)
    print("Input shape:", x.shape)
    print("Reconstructed shape:", reconstructed.shape)
    print("Mean shape:", mu.shape)
    print("Logvar shape:", logvar.shape)
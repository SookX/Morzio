import torch
import torch.nn as nn
from model.encoder.encoder import Encoder
from model.ls_head.ls import LatentSpaceHead
from model.decoder.decoder import Decoder

class VAE(nn.Module):
    def __init__(self, input_dim=20, base_dim=32, latent_dim=None):
        super(VAE, self).__init__()
        
        if latent_dim is None:
            latent_dim = base_dim * 8

        encoder_output_dim = base_dim * 4
        
        self.encoder = Encoder(input_dim=input_dim, base_dim=base_dim)
        self.ls_head = LatentSpaceHead(encoder_dim=encoder_output_dim, latent_dim=latent_dim)
        self.decoder = Decoder(output_dim=input_dim, base_dim=base_dim, latent_dim=latent_dim)
    
    def forward(self, x):
        encoded = self.encoder(x)
        z, mu, logvar = self.ls_head(encoded)
        reconstructed = self.decoder(z)
        return reconstructed, mu, logvar

if __name__ == "__main__":
    x = torch.randn((10, 20))  
    vae = VAE(input_dim=20, base_dim=32, latent_dim=64)
    z, mu, logvar = vae.forward(x)
    print("Input shape:", x.shape)
    print("Latent vector shape:", z.shape)
    print("Mean shape:", mu.shape)
    print("Logvar shape:", logvar.shape)
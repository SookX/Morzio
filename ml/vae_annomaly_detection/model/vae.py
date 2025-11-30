import torch
import torch.nn as nn

from model.decoder.decoder import Decoder
from model.encoder.encoder import Encoder
from model.ls_head.ls import LatentSpaceHead


class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.ls_head = LatentSpaceHead(hidden_dim * 2, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim)

    def forward(self, x):
        encoded = self.encoder(x)
        z, mu, logvar = self.ls_head(encoded)
        reconstructed = self.decoder(z)
        return reconstructed, mu, logvar

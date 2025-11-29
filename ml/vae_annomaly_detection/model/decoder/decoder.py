import torch
import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

        self.activation = nn.SiLU()

    def forward(self, z):
        z = self.activation(self.fc1(z))
        z = self.fc2(z)
        return z

import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, latent_dim)

        self.activation = nn.SiLU()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x
    
if __name__ == "__main__":
    x = torch.randn((10, 20))  
    encoder = Encoder(input_dim=20, hidden_dim=50, latent_dim=10)
    z = encoder(x)
    print("Input shape:", x.shape)
    print("Encoded shape:", z.shape)
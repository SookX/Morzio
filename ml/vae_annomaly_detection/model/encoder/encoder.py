import torch
import torch.nn as nn
from model.attention.simple_attention import SimpleAttention

class Encoder(nn.Module):
    def __init__(self, input_dim=20, base_dim=32):
        super().__init__()
        
        # 20 -> 1*32
        self.block1 = nn.Sequential(
            nn.Linear(input_dim, base_dim),
            nn.LayerNorm(base_dim),
            nn.LeakyReLU(),
            # nn.Dropout(0.1)
        )
        self.attention1 = SimpleAttention(base_dim)
        
        # 1*32 -> 2*32
        self.block2 = nn.Sequential(
            nn.Linear(base_dim, base_dim * 2),
            nn.LayerNorm(base_dim * 2),
            nn.LeakyReLU(),
            #nn.Dropout(0.1)
        )
        self.attention2 = SimpleAttention(base_dim * 2)
        
        # 2*32 -> 4*32
        self.block3 = nn.Sequential(
            nn.Linear(base_dim * 2, base_dim * 4),
            nn.LayerNorm(base_dim * 4),
            nn.LeakyReLU(),
            #nn.Dropout(0.1)
        )
        self.attention3 = SimpleAttention(base_dim * 4)
        
        
        self.res_proj = nn.Linear(base_dim, base_dim * 4)
    
    def forward(self, x):
        h1 = self.block1(x)
        h1_att = self.attention1(h1)
        
        h2 = self.block2(h1_att)
        h2_att = self.attention2(h2)
        
        h3 = self.block3(h2_att)
        out = self.attention3(h3)
        
        residual = self.res_proj(h1)
        
        return out + residual


if __name__ == "__main__":
    x = torch.randn((10, 20))
    encoder = Encoder(input_dim=20, hidden_dim=50, latent_dim=10)
    z = encoder(x)
    print("Input shape:", x.shape)
    print("Encoded shape:", z.shape)

import torch
import torch.nn as nn
from model.attention.simple_attention import SimpleAttention

class Decoder(nn.Module):
    def __init__(self, output_dim=20, base_dim=32, latent_dim=None):
        super().__init__()
        
        self.block1 = nn.Sequential(
            nn.Linear(latent_dim, base_dim * 4),
            nn.LayerNorm(base_dim * 4),
            nn.SiLU(),
            nn.Dropout(0.1)
        )
        self.attention1 = SimpleAttention(base_dim * 4)
        
        self.block2 = nn.Sequential(
            nn.Linear(base_dim * 4, base_dim * 2),
            nn.LayerNorm(base_dim * 2),
            nn.SiLU(),
            nn.Dropout(0.1)
        )
        self.attention2 = SimpleAttention(base_dim * 2)
        
        self.block3 = nn.Sequential(
            nn.Linear(base_dim * 2, base_dim),
            nn.LayerNorm(base_dim),
            nn.SiLU(),
            nn.Dropout(0.1)
        )
        self.attention3 = SimpleAttention(base_dim)
        
        self.block4 = nn.Sequential(
            nn.Linear(base_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.SiLU()
        )
        
        self.res_proj = nn.Linear(base_dim * 4, output_dim)
    
    def forward(self, z):
        h1 = self.block1(z)
        h1_att = self.attention1(h1)
        
        h2 = self.block2(h1_att)
        h2_att = self.attention2(h2)
        
        h3 = self.block3(h2_att)
        h3_att = self.attention3(h3)
        
        out = self.block4(h3_att)
        residual = self.res_proj(h1)
        
        return out + residual
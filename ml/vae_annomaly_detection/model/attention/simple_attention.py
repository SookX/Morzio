import torch
import torch.nn as nn

import torch
import torch.nn as nn

class SimpleAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        scores = self.proj(x)          
        weights = self.softmax(scores) 
        attended = x * weights         
        return attended

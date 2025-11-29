import torch
import torch.nn as nn


class TokenSelfAttention(nn.Module):
    """Applies multi-head self-attention over a fixed number of tokens."""

    def __init__(self, hidden_dim: int, num_tokens: int = 4, num_heads: int = 4, dropout: float = 0.05) -> None:
        super().__init__()
        if hidden_dim % num_tokens != 0:
            raise ValueError("hidden_dim must be divisible by num_tokens")
        self.num_tokens = num_tokens
        self.token_dim = hidden_dim // num_tokens
        if self.token_dim % num_heads != 0:
            raise ValueError("token dimension must be divisible by num_heads")
        self.attn = nn.MultiheadAttention(
            embed_dim=self.token_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(self.token_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.size(0)
        tokens = x.view(batch, self.num_tokens, self.token_dim)
        attn_out, _ = self.attn(tokens, tokens, tokens)
        tokens = self.norm(attn_out + tokens)
        return tokens.reshape(batch, -1)


class Encoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout: float = 0.05,
        attn_tokens: int = 4,
        attn_heads: int = 4,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.attn = TokenSelfAttention(hidden_dim, attn_tokens, attn_heads, dropout)
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.output_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = self.attn(x)
        return self.output_proj(x)


if __name__ == "__main__":
    x = torch.randn((10, 20))
    encoder = Encoder(input_dim=20, hidden_dim=64)
    z = encoder(x)
    print("Input shape:", x.shape)
    print("Encoded shape:", z.shape)

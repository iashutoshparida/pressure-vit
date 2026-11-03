"""Multi-head self-attention written from first principles.

No nn.MultiheadAttention, no F.scaled_dot_product_attention: every step
(projection, head split, scaled dot product, masking, softmax, head merge)
is spelled out so it can be read, tested and explained line by line.
"""
from __future__ import annotations

import math

import torch
from torch import nn


class MultiHeadSelfAttention(nn.Module):
    """Scaled dot-product self-attention over a token sequence.

    Input  x: (batch, tokens, dim)
    Output y: (batch, tokens, dim), attention weights: (batch, heads, tokens, tokens)
    """

    def __init__(self, dim: int, heads: int, attn_dropout: float = 0.0, proj_dropout: float = 0.0):
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by heads ({heads})")
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        # Scaling by 1/sqrt(d_k) keeps the dot products at unit variance when
        # q and k have unit-variance entries, so softmax does not saturate early.
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # One fused projection for queries, keys and values (3 * dim outputs).
        self.qkv = nn.Linear(dim, 3 * dim, bias=True)
        self.out = nn.Linear(dim, dim, bias=True)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj_drop = nn.Dropout(proj_dropout)

    def split_heads(self, t: torch.Tensor) -> torch.Tensor:
        # (B, N, dim) -> (B, heads, N, head_dim)
        b, n, _ = t.shape
        return t.view(b, n, self.heads, self.head_dim).transpose(1, 2)

    def merge_heads(self, t: torch.Tensor) -> torch.Tensor:
        # (B, heads, N, head_dim) -> (B, N, dim)
        b, h, n, d = t.shape
        return t.transpose(1, 2).contiguous().view(b, n, h * d)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = self.split_heads(q), self.split_heads(k), self.split_heads(v)

        # Attention scores: similarity of every query with every key.
        scores = (q @ k.transpose(-2, -1)) * self.scale          # (B, h, N, N)

        if mask is not None:
            # mask: True where attention is allowed; broadcastable to (B, h, N, N).
            scores = scores.masked_fill(~mask, float("-inf"))

        # Subtracting the row max before exp is what softmax does internally;
        # torch.softmax is numerically stable, so it is used for the normalisation step.
        weights = torch.softmax(scores, dim=-1)                   # rows sum to 1
        weights = self.attn_drop(weights)

        context = weights @ v                                     # (B, h, N, head_dim)
        y = self.proj_drop(self.out(self.merge_heads(context)))
        return y, weights

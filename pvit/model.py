"""A small Vision Transformer for bed pressure maps, built from parts in this repo.

Pressure frame (1 x 64 x 32)  ->  non-overlapping patches  ->  linear patch embedding
-> [CLS] + positional embedding  ->  pre-LayerNorm transformer blocks  ->  [CLS] -> classes
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .attention import MultiHeadSelfAttention


@dataclass
class ViTConfig:
    img_h: int = 64
    img_w: int = 32
    patch: int = 8
    dim: int = 64
    depth: int = 4
    heads: int = 4
    mlp_ratio: float = 2.0
    num_classes: int = 3
    dropout: float = 0.1
    pos_embedding: bool = True


class PatchEmbed(nn.Module):
    """Cut the frame into patch x patch tiles and project each tile to `dim`.

    Implemented with reshape/permute (no Conv2d) so the tokenisation is explicit.
    Patches are ordered row-major: head of the bed first, left to right.
    """

    def __init__(self, img_h: int, img_w: int, patch: int, dim: int):
        super().__init__()
        if img_h % patch or img_w % patch:
            raise ValueError("image size must be divisible by patch size")
        self.patch = patch
        self.grid = (img_h // patch, img_w // patch)
        self.num_patches = self.grid[0] * self.grid[1]
        self.proj = nn.Linear(patch * patch, dim)

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        # (B, 1, H, W) -> (B, num_patches, patch*patch)
        b, c, h, w = x.shape
        p = self.patch
        x = x.view(b, c, h // p, p, w // p, p)          # B, C, gh, p, gw, p
        x = x.permute(0, 2, 4, 1, 3, 5)                 # B, gh, gw, C, p, p
        return x.reshape(b, (h // p) * (w // p), c * p * p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.patchify(x))


class MLP(nn.Module):
    def __init__(self, dim: int, hidden: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, dim), nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Pre-LN transformer block: x + Attn(LN(x)), then x + MLP(LN(x)).

    Pre-LN keeps an identity path through the residual stream, which trains
    stably without learning-rate warm-up tricks at this small scale.
    """

    def __init__(self, dim: int, heads: int, mlp_ratio: float, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, heads, attn_dropout=dropout, proj_dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), dropout)

    def forward(self, x):
        a, w = self.attn(self.norm1(x))
        x = x + a
        x = x + self.mlp(self.norm2(x))
        return x, w


class PressureViT(nn.Module):
    def __init__(self, cfg: ViTConfig):
        super().__init__()
        self.cfg = cfg
        self.patch_embed = PatchEmbed(cfg.img_h, cfg.img_w, cfg.patch, cfg.dim)
        n = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.dim))
        # Learned absolute positions: lets attention know *where* on the mattress a patch is.
        self.pos = nn.Parameter(torch.zeros(1, n + 1, cfg.dim)) if cfg.pos_embedding else None
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(
            [Block(cfg.dim, cfg.heads, cfg.mlp_ratio, cfg.dropout) for _ in range(cfg.depth)]
        )
        self.norm = nn.LayerNorm(cfg.dim)
        self.head = nn.Linear(cfg.dim, cfg.num_classes)
        self._init()

    def _init(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        if self.pos is not None:
            nn.init.trunc_normal_(self.pos, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        tokens = self.patch_embed(x)                                   # B, N, D
        cls = self.cls_token.expand(tokens.shape[0], -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)                       # B, N+1, D
        if self.pos is not None:
            tokens = tokens + self.pos
        tokens = self.drop(tokens)
        attns = []
        for blk in self.blocks:
            tokens, w = blk(tokens)
            attns.append(w)
        logits = self.head(self.norm(tokens[:, 0]))
        return (logits, attns) if return_attn else logits


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def attention_rollout(attns: list[torch.Tensor]) -> torch.Tensor:
    """Abnar & Zuidema (2020) rollout: how much each patch flows into [CLS].

    Averages heads, adds the residual identity, renormalises rows, and multiplies
    through the layers. Returns (B, num_patches) relevance of each patch for [CLS].
    """
    result = None
    for w in attns:
        a = w.mean(dim=1)                                              # B, T, T
        a = a + torch.eye(a.shape[-1], device=a.device)
        a = a / a.sum(dim=-1, keepdim=True)
        result = a if result is None else a @ result
    return result[:, 0, 1:]

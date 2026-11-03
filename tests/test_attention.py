"""Correctness tests for the hand-written attention and the ViT plumbing."""
import torch
import torch.nn.functional as F
from torch import nn

from pvit import MultiHeadSelfAttention, PressureViT, ViTConfig, attention_rollout, count_params
from pvit.model import PatchEmbed

torch.manual_seed(0)


def _attn(dim=32, heads=4):
    m = MultiHeadSelfAttention(dim, heads)
    m.eval()
    return m


def test_matches_pytorch_sdpa():
    """Our attention equals PyTorch's fused scaled_dot_product_attention on the same q, k, v."""
    m = _attn()
    x = torch.randn(3, 10, 32)
    y, _ = m(x)
    q, k, v = m.qkv(x).chunk(3, dim=-1)
    q, k, v = m.split_heads(q), m.split_heads(k), m.split_heads(v)
    ref = m.out(m.merge_heads(F.scaled_dot_product_attention(q, k, v)))
    assert torch.allclose(y, ref, atol=1e-6)


def test_matches_nn_multiheadattention():
    """Copying our weights into nn.MultiheadAttention gives the same outputs and head-averaged weights."""
    m = _attn()
    ref = nn.MultiheadAttention(32, 4, batch_first=True)
    ref.eval()
    with torch.no_grad():
        ref.in_proj_weight.copy_(m.qkv.weight)
        ref.in_proj_bias.copy_(m.qkv.bias)
        ref.out_proj.weight.copy_(m.out.weight)
        ref.out_proj.bias.copy_(m.out.bias)
    x = torch.randn(2, 7, 32)
    y, w = m(x)
    y_ref, w_ref = ref(x, x, x, need_weights=True, average_attn_weights=True)
    assert torch.allclose(y, y_ref, atol=1e-6)
    assert torch.allclose(w.mean(dim=1), w_ref, atol=1e-6)


def test_rows_sum_to_one():
    _, w = _attn()(torch.randn(2, 9, 32))
    assert torch.allclose(w.sum(-1), torch.ones_like(w.sum(-1)), atol=1e-6)


def test_permutation_equivariance():
    """Self-attention without positions is permutation-equivariant: shuffling tokens shuffles outputs."""
    m = _attn()
    x = torch.randn(1, 8, 32)
    perm = torch.randperm(8)
    y, _ = m(x)
    y_perm, _ = m(x[:, perm])
    assert torch.allclose(y[:, perm], y_perm, atol=1e-6)


def test_mask_blocks_attention():
    m = _attn()
    x = torch.randn(1, 5, 32)
    mask = torch.tril(torch.ones(5, 5, dtype=torch.bool))       # causal
    _, w = m(x, mask=mask)
    assert torch.all(w[..., ~mask] == 0)


def test_patchify_roundtrip():
    """Patches are exact tiles of the frame, in row-major order."""
    pe = PatchEmbed(64, 32, 8, 16)
    x = torch.arange(64 * 32, dtype=torch.float32).view(1, 1, 64, 32)
    patches = pe.patchify(x)
    assert patches.shape == (1, 32, 64)
    assert torch.equal(patches[0, 0], x[0, 0, :8, :8].reshape(-1))
    assert torch.equal(patches[0, 1], x[0, 0, :8, 8:16].reshape(-1))
    assert torch.equal(patches[0, 4], x[0, 0, 8:16, :8].reshape(-1))


def test_vit_shapes_and_gradients():
    cfg = ViTConfig()
    model = PressureViT(cfg)
    x = torch.randn(4, 1, 64, 32)
    logits, attns = model(x, return_attn=True)
    assert logits.shape == (4, cfg.num_classes)
    assert len(attns) == cfg.depth and attns[0].shape == (4, cfg.heads, 33, 33)
    logits.sum().backward()
    missing = [n for n, p in model.named_parameters() if p.grad is None]
    assert not missing, f"no gradient reached: {missing}"


def test_rollout_is_a_distribution():
    model = PressureViT(ViTConfig()).eval()
    _, attns = model(torch.randn(2, 1, 64, 32), return_attn=True)
    r = attention_rollout(attns)
    assert r.shape == (2, 32) and torch.all(r >= 0)


def test_small_enough_for_edge():
    assert count_params(PressureViT(ViTConfig())) < 250_000

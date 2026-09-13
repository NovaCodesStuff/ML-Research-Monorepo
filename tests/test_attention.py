"""These tests are the spec for the CORE attention file. They fail until you write it.

Run: pytest tests/test_attention.py -x
"""
import math

import torch

from models.transformer.attention import MultiHeadAttention, scaled_dot_product_attention


def test_attention_matches_reference():
    torch.manual_seed(0)
    q = torch.randn(2, 3, 5, 8); k = torch.randn(2, 3, 5, 8); v = torch.randn(2, 3, 5, 8)
    out, w = scaled_dot_product_attention(q, k, v)
    ref = torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(8), dim=-1) @ v
    assert torch.allclose(out, ref, atol=1e-6)
    assert torch.allclose(w.sum(-1), torch.ones(2, 3, 5), atol=1e-6)


def test_causal_mask_blocks_future():
    q = k = v = torch.randn(1, 1, 4, 8)
    mask = torch.triu(torch.ones(4, 4, dtype=torch.bool), diagonal=1)
    _, w = scaled_dot_product_attention(q, k, v, mask=mask)
    assert torch.all(w.masked_select(mask) == 0)


def test_mha_shapes_and_params():
    mha = MultiHeadAttention(d_model=32, n_heads=4)
    x = torch.randn(2, 10, 32)
    assert mha(x).shape == (2, 10, 32)
    n = sum(p.numel() for p in mha.parameters())
    assert n == 4 * (32 * 32 + 32), "expected exactly W_q, W_k, W_v, W_o with biases"

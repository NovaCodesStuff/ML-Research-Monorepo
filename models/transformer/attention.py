# CORE — hand-written from the paper, no library shortcuts.
"""Scaled dot-product attention and multi-head attention, from the paper.

Attention(Q, K, V) = softmax(QKᵀ / √d_k) V

Write this yourself. Shapes to keep straight:
  Q, K, V : (batch, heads, seq, d_k)
  scores  : (batch, heads, seq, seq)
  mask    : broadcastable to scores; True where attention is NOT allowed
"""
from torch import nn


def scaled_dot_product_attention(q, k, v, mask=None, dropout=None):
    raise NotImplementedError("write me")


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        raise NotImplementedError("write me")

    def forward(self, x, mask=None):
        raise NotImplementedError("write me")

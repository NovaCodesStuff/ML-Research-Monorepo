# CORE — hand-written from the paper, no library shortcuts.
"""Positional encoding, feed-forward, and the pre-norm encoder block."""
from torch import nn


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        raise NotImplementedError("write me")

    def forward(self, x):
        raise NotImplementedError("write me")


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        raise NotImplementedError("write me")

    def forward(self, x):
        raise NotImplementedError("write me")


class EncoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        raise NotImplementedError("write me")

    def forward(self, x, mask=None):
        raise NotImplementedError("write me")

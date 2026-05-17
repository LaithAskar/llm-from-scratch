"""
Stand-in implementations used while layers.py is still stubs.

Once layers.py is implemented end-to-end, delete this file and the
`patch_layers` fixture in conftest.py — the same tests will then exercise
the real code with no other changes needed.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class _DummyAttn(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.out_proj = nn.Linear(d, d)

    def forward(self, x):
        return self.out_proj(x)


class _DummyFFN(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.down_proj = nn.Linear(d, d)

    def forward(self, x):
        return self.down_proj(x)


class DummyBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = _DummyAttn(config.d_model)
        self.ffn = _DummyFFN(config.d_model)

    def forward(self, x, mask=None):
        return x + self.ffn(self.attn(x))


class DummyRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        return x * self.weight


def dummy_causal_mask(seq_len, device=None):
    return torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))

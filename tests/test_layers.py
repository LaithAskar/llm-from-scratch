"""
Tests for layers.py — exercise the real implementations as they land.

Patching of unimplemented stubs (still RMSNorm + MultiHeadAttention +
TransformerBlock at time of writing) happens in conftest.py; once a
component is real, its patch comes out and these tests cover it.
"""

from __future__ import annotations

import pytest
import torch

from layers import causal_mask


# --- causal_mask -----------------------------------------------------------


def test_causal_mask_shape_and_dtype():
    m = causal_mask(8)
    assert m.shape == (8, 8)
    assert m.dtype == torch.bool


def test_causal_mask_lower_triangular():
    m = causal_mask(5)
    # row i should have True in columns 0..i and False in columns i+1..n-1
    for i in range(5):
        for j in range(5):
            assert bool(m[i, j]) == (j <= i), f"mask[{i},{j}] wrong"


def test_causal_mask_diagonal_is_attendable():
    # A token must always be able to attend to itself (j == i).
    m = causal_mask(10)
    for i in range(10):
        assert bool(m[i, i]) is True


def test_causal_mask_blocks_future():
    # Strict upper triangle (j > i) must be False — no peeking ahead.
    m = causal_mask(10)
    upper = m.triu(diagonal=1)
    assert not upper.any(), "causal mask leaks future positions"


def test_causal_mask_works_with_masked_fill():
    # Integration check: the mask must be usable as a masked_fill predicate
    # the way MultiHeadAttention will use it.
    scores = torch.zeros(1, 1, 4, 4)
    m = causal_mask(4)
    masked = scores.masked_fill(~m, float("-inf"))
    # First row attends only to position 0; positions 1,2,3 are -inf
    assert masked[0, 0, 0, 0].item() == 0.0
    assert masked[0, 0, 0, 1].item() == float("-inf")
    # Last row attends to all 4 positions; all entries are 0
    assert torch.isfinite(masked[0, 0, 3, :]).all()


def test_causal_mask_respects_device():
    # CPU device should round-trip cleanly. (CUDA path covered implicitly by
    # model.py register_buffer + .to(device) movement.)
    m = causal_mask(3, device=torch.device("cpu"))
    assert m.device.type == "cpu"

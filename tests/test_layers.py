"""
Tests for layers.py — exercise the real implementations as they land.

Patching of unimplemented stubs (still RMSNorm + MultiHeadAttention +
TransformerBlock at time of writing) happens in conftest.py; once a
component is real, its patch comes out and these tests cover it.
"""

from __future__ import annotations

import pytest
import torch

from layers import RMSNorm, causal_mask


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


# --- RMSNorm ---------------------------------------------------------------


def test_rmsnorm_preserves_shape():
    rn = RMSNorm(dim=16)
    x = torch.randn(2, 5, 16)
    y = rn(x)
    assert y.shape == x.shape


def test_rmsnorm_weight_is_parameter_of_correct_shape():
    rn = RMSNorm(dim=32)
    assert isinstance(rn.weight, torch.nn.Parameter)
    assert rn.weight.shape == (32,)
    # init: ones (identity at first forward, weight only diverges via training).
    assert torch.allclose(rn.weight, torch.ones(32))


def test_rmsnorm_output_has_unit_rms_at_init():
    # With weight=1, the per-token RMS of the output should be ~1 (up to eps).
    torch.manual_seed(0)
    rn = RMSNorm(dim=64)
    x = torch.randn(4, 8, 64) * 3.0  # arbitrary scale — RMSNorm should rescale to 1
    y = rn(x)
    per_token_rms = y.pow(2).mean(dim=-1).sqrt()
    assert torch.allclose(per_token_rms, torch.ones_like(per_token_rms), atol=1e-3)


def test_rmsnorm_gradient_flows_to_input_and_weight():
    rn = RMSNorm(dim=8)
    x = torch.randn(3, 8, requires_grad=True)
    y = rn(x).sum()
    y.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert rn.weight.grad is not None and torch.isfinite(rn.weight.grad).all()


def test_rmsnorm_promotes_bf16_input_to_fp32_for_stability():
    # Computing mean(x²) in bf16 over a 1024-dim vector with values ~N(0,1)
    # loses ~half the digits and routinely produces RMS ≠ 1 by a few %.
    # The fp32-promotion fix should bring it within ~1e-2 of unit RMS even
    # when input is bf16. This test would FAIL on a naive bf16 implementation.
    torch.manual_seed(0)
    rn = RMSNorm(dim=1024).to(torch.bfloat16)
    x = torch.randn(2, 1024, dtype=torch.bfloat16)
    y = rn(x)
    assert y.dtype == torch.bfloat16
    # Cast back to fp32 just for the assertion math (otherwise the check itself
    # would lose precision).
    rms = y.to(torch.float32).pow(2).mean(dim=-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=2e-2)


def test_rmsnorm_eps_prevents_division_by_zero():
    rn = RMSNorm(dim=4, eps=1e-5)
    x = torch.zeros(2, 4)  # all-zero input would otherwise divide by zero
    y = rn(x)
    assert torch.isfinite(y).all()


def test_rmsnorm_weight_scales_output():
    # If weight is set to 2.0, output magnitude should scale by 2x relative
    # to weight=1.0 (since the normalization is the same).
    torch.manual_seed(0)
    x = torch.randn(2, 8, 16)

    rn1 = RMSNorm(dim=16)
    rn2 = RMSNorm(dim=16)
    with torch.no_grad():
        rn2.weight.fill_(2.0)

    y1 = rn1(x)
    y2 = rn2(x)
    assert torch.allclose(y2, 2.0 * y1, atol=1e-5)

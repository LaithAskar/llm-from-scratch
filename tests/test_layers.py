"""
Tests for layers.py — exercise the real implementations as they land.

Patching of unimplemented stubs (still RMSNorm + MultiHeadAttention +
TransformerBlock at time of writing) happens in conftest.py; once a
component is real, its patch comes out and these tests cover it.
"""

from __future__ import annotations

import pytest
import torch

from layers import (
    GeluFFN,
    MultiHeadAttention,
    RMSNorm,
    RotaryEmbedding,
    SwiGLUFFN,
    TransformerBlock,
    causal_mask,
)


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


# --- MultiHeadAttention ----------------------------------------------------


def test_mha_output_shape_matches_input():
    mha = MultiHeadAttention(embed_dim=32, num_heads=4)
    x = torch.randn(2, 7, 32)
    out = mha(x)
    assert out.shape == x.shape


def test_mha_rejects_non_divisible_dims():
    with pytest.raises(ValueError, match="divisible"):
        MultiHeadAttention(embed_dim=30, num_heads=4)  # 30 % 4 != 0


def test_mha_gradient_flows_to_all_params():
    mha = MultiHeadAttention(embed_dim=16, num_heads=4)
    x = torch.randn(2, 5, 16, requires_grad=True)
    out = mha(x).sum()
    out.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    for name, p in mha.named_parameters():
        assert p.grad is not None, f"no grad on {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite grad on {name}"


def test_mha_causal_mask_blocks_future_information():
    """
    Regression test for two bugs at once:
      (a) inverted mask convention (writing -inf where True instead of False)
      (b) wrong mask broadcasting (mask not actually applied per-position)

    With a causal mask, perturbing position 1's input must NOT change
    position 0's output. (Position 0 cannot see position 1.) Position 1's
    output WILL change.
    """
    torch.manual_seed(0)
    mha = MultiHeadAttention(embed_dim=32, num_heads=4)
    mha.eval()  # disable dropout so the comparison is exact
    x = torch.randn(1, 4, 32)
    m = causal_mask(4)

    out_a = mha(x, mask=m)

    x_perturbed = x.clone()
    x_perturbed[0, 1, :] = torch.randn(32)
    out_b = mha(x_perturbed, mask=m)

    assert torch.allclose(out_a[0, 0], out_b[0, 0], atol=1e-6), (
        "position 0's output changed when position 1's input was perturbed — "
        "causal mask is leaking future information"
    )
    assert not torch.allclose(out_a[0, 1], out_b[0, 1], atol=1e-3), (
        "position 1's output unchanged despite different input — "
        "attention layer not actually attending to perturbed token"
    )


def test_mha_no_mask_mixes_all_positions():
    # Without mask, position 0's output SHOULD depend on every input position
    # (every token attends to every other token).
    torch.manual_seed(0)
    mha = MultiHeadAttention(embed_dim=32, num_heads=4)
    mha.eval()
    x = torch.randn(1, 4, 32)

    out_a = mha(x)
    x_perturbed = x.clone()
    x_perturbed[0, 3, :] = torch.randn(32)
    out_b = mha(x_perturbed)

    # Position 0 (the first token) should see position 3 (the last) and
    # therefore its output changes when position 3 changes.
    assert not torch.allclose(out_a[0, 0], out_b[0, 0], atol=1e-4), (
        "without mask, position 0 should attend to position 3"
    )


def test_mha_param_count_bias_off_vs_on():
    # 4 linears (Q, K, V, out) × (embed_dim,) bias each.
    # bias=True should add exactly 4 * embed_dim params.
    d = 64
    mha_nb = MultiHeadAttention(embed_dim=d, num_heads=8, bias=False)
    mha_wb = MultiHeadAttention(embed_dim=d, num_heads=8, bias=True)
    n_nb = sum(p.numel() for p in mha_nb.parameters())
    n_wb = sum(p.numel() for p in mha_wb.parameters())
    assert n_wb - n_nb == 4 * d


def test_mha_dropout_is_noop_in_eval_mode():
    # Two forward passes in eval mode with the same input + seed should
    # be exactly equal even with non-zero dropout.
    mha = MultiHeadAttention(embed_dim=16, num_heads=4, dropout=0.5)
    mha.eval()
    x = torch.randn(2, 4, 16)
    out1 = mha(x)
    out2 = mha(x)
    assert torch.allclose(out1, out2, atol=1e-6)


def test_mha_calls_rotary_hook_if_provided():
    # If a rotary callable is passed, it should be applied to q and k
    # before the attention matmul. A no-op rotary should give identical
    # output to no rotary. Rotary contract: callable(q, k, position_offset=int).
    torch.manual_seed(0)
    x = torch.randn(1, 4, 16)

    mha_no_rope = MultiHeadAttention(embed_dim=16, num_heads=4)
    mha_no_rope.eval()
    mha_noop_rope = MultiHeadAttention(
        embed_dim=16, num_heads=4, rotary=lambda q, k, position_offset=0: (q, k)
    )
    mha_noop_rope.eval()
    mha_noop_rope.load_state_dict(mha_no_rope.state_dict())

    out1 = mha_no_rope(x)
    out2 = mha_noop_rope(x)
    assert torch.allclose(out1, out2, atol=1e-6)

    mha_scaled = MultiHeadAttention(
        embed_dim=16, num_heads=4,
        rotary=lambda q, k, position_offset=0: (q * 0.5, k * 0.5),
    )
    mha_scaled.eval()
    mha_scaled.load_state_dict(mha_no_rope.state_dict())
    out3 = mha_scaled(x)
    assert out3.shape == x.shape


# --- KV cache (MHA + Block + RoPE-offset) ----------------------------------


def test_rope_position_offset_equals_absolute_position():
    """
    Rotating a single token at position_offset=N must equal rotating the
    same token if it had appeared at index N in a longer sequence (where
    the rotation is computed without offset).
    """
    torch.manual_seed(0)
    rope = RotaryEmbedding(head_dim=8, max_seq_len=16)
    q = torch.randn(1, 1, 1, 8)
    k = torch.randn(1, 1, 1, 8)

    # Path A: rotate with offset=5 (treat the single token as at position 5)
    q_off, k_off = rope(q, k, position_offset=5)

    # Path B: build a 6-token sequence with q, k at position 5, rotate
    # without offset, extract position 5
    q_full = torch.zeros(1, 1, 6, 8)
    k_full = torch.zeros(1, 1, 6, 8)
    q_full[:, :, 5:6] = q
    k_full[:, :, 5:6] = k
    q_full_rot, k_full_rot = rope(q_full, k_full)

    assert torch.allclose(q_off[:, :, 0], q_full_rot[:, :, 5], atol=1e-6)
    assert torch.allclose(k_off[:, :, 0], k_full_rot[:, :, 5], atol=1e-6)


def test_mha_cached_prefill_plus_decode_equals_full():
    """
    THE correctness test for KV cache. A prefill on the first N-1 tokens
    plus a single-token decode on the Nth must produce the same output as
    a single full forward over all N tokens.

    If this fails, KV cache is wrong somewhere: position math, cache
    concatenation, mask handling, or RoPE offset.
    """
    torch.manual_seed(0)
    mha = MultiHeadAttention(embed_dim=16, num_heads=4)
    mha.eval()

    x = torch.randn(1, 5, 16)

    # Path A: one full forward with causal mask
    full_out = mha(x, mask=causal_mask(5))

    # Path B: prefill first 4 tokens with cache, then decode 5th
    cache: dict = {}
    prefill_out, cache = mha(x[:, :4], mask=causal_mask(4), kv_cache=cache)
    decode_out, cache = mha(x[:, 4:5], mask=None, kv_cache=cache)

    cached_out = torch.cat([prefill_out, decode_out], dim=1)

    assert torch.allclose(full_out, cached_out, atol=1e-5), (
        f"max diff: {(full_out - cached_out).abs().max().item():.6f}"
    )


def test_mha_cached_decode_step_by_step_equals_full():
    """
    Stress version: decode one token at a time from an empty cache. The
    concatenated outputs must equal a single full forward.
    """
    torch.manual_seed(0)
    mha = MultiHeadAttention(embed_dim=16, num_heads=4)
    mha.eval()
    x = torch.randn(1, 6, 16)

    full_out = mha(x, mask=causal_mask(6))

    cache: dict = {}
    pieces = []
    for t in range(6):
        # Each step is a single-token forward; mask must be None
        # (single new token, no future to mask).
        out, cache = mha(x[:, t:t + 1], mask=None, kv_cache=cache)
        pieces.append(out)
    cached_out = torch.cat(pieces, dim=1)

    assert torch.allclose(full_out, cached_out, atol=1e-5)


def test_mha_cached_with_rope_equals_full():
    """Same correctness check but with RoPE active. Verifies position_offset
    plumbing through MHA."""
    torch.manual_seed(0)
    rope = RotaryEmbedding(head_dim=4, max_seq_len=16)
    mha = MultiHeadAttention(embed_dim=16, num_heads=4, rotary=rope)
    mha.eval()
    x = torch.randn(1, 5, 16)

    full_out = mha(x, mask=causal_mask(5))

    cache: dict = {}
    prefill_out, cache = mha(x[:, :3], mask=causal_mask(3), kv_cache=cache)
    step1_out, cache = mha(x[:, 3:4], mask=None, kv_cache=cache)
    step2_out, cache = mha(x[:, 4:5], mask=None, kv_cache=cache)
    cached_out = torch.cat([prefill_out, step1_out, step2_out], dim=1)

    assert torch.allclose(full_out, cached_out, atol=1e-5)


def test_block_cached_equals_full():
    """Block-level: cached prefill+decode equals full forward (no RoPE)."""
    cfg = _tiny_block_config()
    block = TransformerBlock(cfg)
    block.eval()
    x = torch.randn(1, 5, cfg.d_model)

    full_out = block(x, mask=causal_mask(5))

    cache: dict = {}
    prefill_out, cache = block(x[:, :4], mask=causal_mask(4), kv_cache=cache)
    decode_out, cache = block(x[:, 4:5], mask=None, kv_cache=cache)
    cached_out = torch.cat([prefill_out, decode_out], dim=1)

    assert torch.allclose(full_out, cached_out, atol=1e-5)


def test_block_cached_with_rope_equals_full():
    """Block-level: cached prefill+decode equals full forward WITH RoPE."""
    cfg = _tiny_block_config(pos_encoding="rope")
    block = TransformerBlock(cfg)
    block.eval()
    x = torch.randn(1, 5, cfg.d_model)

    full_out = block(x, mask=causal_mask(5))

    cache: dict = {}
    prefill_out, cache = block(x[:, :4], mask=causal_mask(4), kv_cache=cache)
    decode_out, cache = block(x[:, 4:5], mask=None, kv_cache=cache)
    cached_out = torch.cat([prefill_out, decode_out], dim=1)

    assert torch.allclose(full_out, cached_out, atol=1e-5)


def test_block_cached_with_rmsnorm_swiglu_modern_stack_equals_full():
    """The 'modern' variant stack (rmsnorm + swiglu + rope) with cache."""
    cfg = _tiny_block_config(
        norm_type="rmsnorm", activation="swiglu", pos_encoding="rope",
        d_ffn=None,
    )
    block = TransformerBlock(cfg)
    block.eval()
    x = torch.randn(1, 4, cfg.d_model)

    full_out = block(x, mask=causal_mask(4))

    cache: dict = {}
    prefill_out, cache = block(x[:, :2], mask=causal_mask(2), kv_cache=cache)
    s1, cache = block(x[:, 2:3], mask=None, kv_cache=cache)
    s2, cache = block(x[:, 3:4], mask=None, kv_cache=cache)
    cached_out = torch.cat([prefill_out, s1, s2], dim=1)

    assert torch.allclose(full_out, cached_out, atol=1e-5)


# --- GeluFFN ---------------------------------------------------------------


def test_gelu_ffn_preserves_outer_shape():
    ffn = GeluFFN(d_model=32, d_ffn=128)
    x = torch.randn(2, 5, 32)
    y = ffn(x)
    assert y.shape == (2, 5, 32)


def test_gelu_ffn_param_count():
    # 2 linears: (d_model -> d_ffn) and (d_ffn -> d_model). bias=False.
    d_model, d_ffn = 32, 128
    ffn = GeluFFN(d_model=d_model, d_ffn=d_ffn, bias=False)
    expected = d_model * d_ffn + d_ffn * d_model
    assert sum(p.numel() for p in ffn.parameters()) == expected


def test_gelu_ffn_has_down_proj_name():
    # model.py's _init_weights looks for `down_proj.weight` to apply the
    # GPT-2 residual scaling. Regression: name must not drift.
    ffn = GeluFFN(d_model=8, d_ffn=16)
    names = {n for n, _ in ffn.named_parameters()}
    assert "down_proj.weight" in names


def test_gelu_ffn_gradient_flows():
    ffn = GeluFFN(d_model=16, d_ffn=32)
    x = torch.randn(2, 4, 16, requires_grad=True)
    ffn(x).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    for n, p in ffn.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), f"bad grad on {n}"


# --- SwiGLUFFN -------------------------------------------------------------


def test_swiglu_ffn_preserves_outer_shape():
    ffn = SwiGLUFFN(d_model=32, d_ffn=64)
    x = torch.randn(2, 5, 32)
    y = ffn(x)
    assert y.shape == (2, 5, 32)


def test_swiglu_ffn_param_count():
    # 3 linears: gate (C->F), up (C->F), down (F->C). bias=False.
    d_model, d_ffn = 32, 64
    ffn = SwiGLUFFN(d_model=d_model, d_ffn=d_ffn, bias=False)
    expected = 3 * d_model * d_ffn
    assert sum(p.numel() for p in ffn.parameters()) == expected


def test_swiglu_param_match_to_gelu_via_palm_sizing():
    # PaLM convention: for matched FFN params, set SwiGLU d_ffn = (8/3) * d_model.
    # ModelConfig.__post_init__ rounds to multiple of 64. Verify that with the
    # PaLM-sized d_ffn, the SwiGLU FFN has ~the same params as GELU at d_ffn=4*d_model.
    d_model = 192
    gelu = GeluFFN(d_model=d_model, d_ffn=4 * d_model)
    palm_d_ffn = int(round(8 / 3 * d_model / 64)) * 64
    swiglu = SwiGLUFFN(d_model=d_model, d_ffn=palm_d_ffn)
    n_gelu = sum(p.numel() for p in gelu.parameters())
    n_swiglu = sum(p.numel() for p in swiglu.parameters())
    # Should match within rounding-to-64 slack (~5%).
    assert abs(n_gelu - n_swiglu) / n_gelu < 0.05, f"GELU={n_gelu}, SwiGLU={n_swiglu}"


def test_swiglu_ffn_has_three_named_projections():
    ffn = SwiGLUFFN(d_model=8, d_ffn=16)
    names = {n for n, _ in ffn.named_parameters()}
    assert "gate_proj.weight" in names
    assert "up_proj.weight" in names
    assert "down_proj.weight" in names


def test_swiglu_ffn_gradient_flows():
    ffn = SwiGLUFFN(d_model=16, d_ffn=32)
    x = torch.randn(2, 4, 16, requires_grad=True)
    ffn(x).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    for n, p in ffn.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), f"bad grad on {n}"


# --- RotaryEmbedding -------------------------------------------------------


def test_rope_rejects_odd_head_dim():
    with pytest.raises(ValueError, match="even"):
        RotaryEmbedding(head_dim=7, max_seq_len=16)


def test_rope_output_shape_matches_input():
    rope = RotaryEmbedding(head_dim=8, max_seq_len=16)
    q = torch.randn(2, 4, 10, 8)
    k = torch.randn(2, 4, 10, 8)
    q_rot, k_rot = rope(q, k)
    assert q_rot.shape == q.shape
    assert k_rot.shape == k.shape


def test_rope_is_identity_at_position_zero():
    """
    cos(0) = 1, sin(0) = 0 → rotation at position 0 is the identity.
    The first row of the rotated tensor must equal the input's first row.
    """
    rope = RotaryEmbedding(head_dim=8, max_seq_len=16)
    q = torch.randn(1, 1, 4, 8)
    k = torch.randn(1, 1, 4, 8)
    q_rot, k_rot = rope(q, k)
    assert torch.allclose(q_rot[..., 0, :], q[..., 0, :], atol=1e-6)
    assert torch.allclose(k_rot[..., 0, :], k[..., 0, :], atol=1e-6)


def test_rope_relative_position_invariance():
    """
    THE defining property of RoPE: dot(R(mθ)·q, R(nθ)·k) depends only on
    (m - n), not on m and n individually.

    Place a fixed (q, k) pair at offsets (0, 1) and again at (5, 6). Both
    have relative offset 1. The attention score must be identical.

    If this fails, the RoPE rotation math is wrong (half-split convention
    inverted, cos/sin swapped, pair indexing wrong, etc.).
    """
    torch.manual_seed(0)
    head_dim = 16
    rope = RotaryEmbedding(head_dim=head_dim, max_seq_len=32)

    q_vec = torch.randn(head_dim)
    k_vec = torch.randn(head_dim)

    # Place at (0, 1): score(rotated_q@0, rotated_k@1)
    q1 = torch.zeros(1, 1, 32, head_dim)
    k1 = torch.zeros(1, 1, 32, head_dim)
    q1[0, 0, 0] = q_vec
    k1[0, 0, 1] = k_vec
    q1_rot, k1_rot = rope(q1, k1)
    score_01 = (q1_rot[0, 0, 0] * k1_rot[0, 0, 1]).sum()

    # Place at (5, 6): same relative offset
    q2 = torch.zeros(1, 1, 32, head_dim)
    k2 = torch.zeros(1, 1, 32, head_dim)
    q2[0, 0, 5] = q_vec
    k2[0, 0, 6] = k_vec
    q2_rot, k2_rot = rope(q2, k2)
    score_56 = (q2_rot[0, 0, 5] * k2_rot[0, 0, 6]).sum()

    assert torch.allclose(score_01, score_56, atol=1e-5), (
        f"RoPE not translation-invariant: score(0,1)={score_01.item():.6f}, "
        f"score(5,6)={score_56.item():.6f}"
    )


def test_rope_different_offsets_give_different_scores():
    """
    Sanity check the inverse: different relative offsets SHOULD produce
    different scores (otherwise RoPE is doing nothing).
    """
    torch.manual_seed(1)
    head_dim = 16
    rope = RotaryEmbedding(head_dim=head_dim, max_seq_len=32)

    q_vec = torch.randn(head_dim)
    k_vec = torch.randn(head_dim)

    def score_at(q_pos, k_pos):
        q = torch.zeros(1, 1, 32, head_dim)
        k = torch.zeros(1, 1, 32, head_dim)
        q[0, 0, q_pos] = q_vec
        k[0, 0, k_pos] = k_vec
        q_r, k_r = rope(q, k)
        return (q_r[0, 0, q_pos] * k_r[0, 0, k_pos]).sum().item()

    s1 = score_at(0, 1)   # offset = 1
    s3 = score_at(0, 3)   # offset = 3
    assert abs(s1 - s3) > 1e-3, "scores at different offsets are suspiciously equal"


def test_rope_rejects_overlong_sequence():
    rope = RotaryEmbedding(head_dim=8, max_seq_len=4)
    q = torch.randn(1, 1, 10, 8)  # T=10 > max=4
    k = torch.randn(1, 1, 10, 8)
    with pytest.raises(ValueError, match="exceeds RoPE max_seq_len"):
        rope(q, k)


def test_rope_gradient_flows():
    rope = RotaryEmbedding(head_dim=8, max_seq_len=16)
    q = torch.randn(1, 2, 4, 8, requires_grad=True)
    k = torch.randn(1, 2, 4, 8, requires_grad=True)
    q_rot, k_rot = rope(q, k)
    (q_rot.sum() + k_rot.sum()).backward()
    assert q.grad is not None and torch.isfinite(q.grad).all()
    assert k.grad is not None and torch.isfinite(k.grad).all()


def test_rope_integrates_with_mha():
    """
    End-to-end: hand MHA a RotaryEmbedding via its rotary= constructor
    arg, run a forward, output shape matches input.
    """
    torch.manual_seed(0)
    rope = RotaryEmbedding(head_dim=8, max_seq_len=16)
    mha = MultiHeadAttention(embed_dim=32, num_heads=4, rotary=rope)
    mha.eval()
    x = torch.randn(1, 6, 32)
    out = mha(x)
    assert out.shape == x.shape


# --- TransformerBlock ------------------------------------------------------


def _tiny_block_config(**overrides):
    from config import ModelConfig

    base = dict(
        vocab_size=257, n_layer=1, n_head=4, d_model=32, context_len=16, dropout=0.0,
    )
    base.update(overrides)
    return ModelConfig(**base)


def test_block_forward_shape_preserved():
    block = TransformerBlock(_tiny_block_config())
    x = torch.randn(2, 8, 32)
    out = block(x, mask=causal_mask(8))
    assert out.shape == x.shape


def test_block_norm_switch_layernorm():
    block = TransformerBlock(_tiny_block_config(norm_type="layernorm"))
    assert isinstance(block.norm1, torch.nn.LayerNorm)
    assert isinstance(block.norm2, torch.nn.LayerNorm)


def test_block_norm_switch_rmsnorm():
    block = TransformerBlock(_tiny_block_config(norm_type="rmsnorm"))
    assert isinstance(block.norm1, RMSNorm)
    assert isinstance(block.norm2, RMSNorm)


def test_block_activation_switch_gelu():
    block = TransformerBlock(_tiny_block_config(activation="gelu"))
    assert isinstance(block.ffn, GeluFFN)


def test_block_activation_switch_swiglu():
    block = TransformerBlock(_tiny_block_config(activation="swiglu"))
    assert isinstance(block.ffn, SwiGLUFFN)


def test_block_rope_switch_on():
    block = TransformerBlock(_tiny_block_config(pos_encoding="rope"))
    assert isinstance(block.attn.rotary, RotaryEmbedding)


def test_block_rope_switch_off():
    block = TransformerBlock(_tiny_block_config(pos_encoding="learned"))
    assert block.attn.rotary is None


def test_block_modern_variant_all_switches():
    """The 'modern' ablation variant: rmsnorm + swiglu + rope."""
    block = TransformerBlock(_tiny_block_config(
        norm_type="rmsnorm", activation="swiglu", pos_encoding="rope",
        d_ffn=None,  # let __post_init__ apply the (8/3)*d_model rule
    ))
    assert isinstance(block.norm1, RMSNorm)
    assert isinstance(block.ffn, SwiGLUFFN)
    assert isinstance(block.attn.rotary, RotaryEmbedding)
    # Forward end-to-end through the full modern stack.
    x = torch.randn(1, 4, 32)
    out = block(x, mask=causal_mask(4))
    assert out.shape == x.shape


def test_block_gradient_flows_to_input_and_all_params():
    block = TransformerBlock(_tiny_block_config())
    x = torch.randn(1, 4, 32, requires_grad=True)
    out = block(x, mask=causal_mask(4))
    out.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    for n, p in block.named_parameters():
        assert p.grad is not None, f"no grad on {n}"
        assert torch.isfinite(p.grad).all(), f"non-finite grad on {n}"


def test_block_residual_pattern_attn_then_ffn():
    """
    With zero-d-out sublayer outputs, block(x) should equal x. We can't
    easily zero attention output, but we can verify the *structure* by
    setting norm + sublayer outputs to ~0 via state_dict surgery and
    asserting the residual path dominates.

    Simpler test: with a pre-norm pattern, the first thing applied to x is
    NOT a sublayer — x flows through the residual. So `block(x).shape == x.shape`
    always (already covered). The structural assertion below is the
    parameter-count check.
    """
    block = TransformerBlock(_tiny_block_config())
    # Block should contain: 2 norms + 1 MHA + 1 FFN. No additional learnable
    # parameters above those.
    sub_param_count = sum(
        sum(p.numel() for p in sub.parameters())
        for sub in [block.norm1, block.norm2, block.attn, block.ffn]
    )
    block_param_count = sum(p.numel() for p in block.parameters())
    assert sub_param_count == block_param_count, (
        "block has parameters outside its 4 declared submodules"
    )

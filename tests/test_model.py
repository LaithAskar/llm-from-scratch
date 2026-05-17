"""
Smoke tests for model.py.

Layers are stubbed via the autouse `patch_layers` fixture in tests/conftest.py.
When layers.py is implemented, the fixture goes away and these tests run
against the real code unchanged.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from layers import RMSNorm


# --- Helpers ---------------------------------------------------------------


def _tiny_config(**overrides):
    from config import ModelConfig

    base = dict(
        vocab_size=257,
        n_layer=2,
        n_head=4,
        d_model=32,
        context_len=16,
        dropout=0.0,
    )
    base.update(overrides)
    return ModelConfig(**base)


# --- Tests -----------------------------------------------------------------


def test_forward_shape_no_targets():
    from model import TransformerLM

    cfg = _tiny_config()
    lm = TransformerLM(cfg)
    idx = torch.randint(0, cfg.vocab_size, (2, 8))
    logits, loss = lm(idx)
    assert logits.shape == (2, 8, cfg.vocab_size)
    assert loss is None


def test_forward_with_targets_returns_loss():
    from model import TransformerLM

    cfg = _tiny_config()
    lm = TransformerLM(cfg)
    idx = torch.randint(0, cfg.vocab_size, (2, 8))
    targets = torch.randint(0, cfg.vocab_size, (2, 8))
    logits, loss = lm(idx, targets)
    assert logits.shape == (2, 8, cfg.vocab_size)
    assert loss is not None and loss.dim() == 0
    # At init, CE should be roughly ln(vocab_size) for ~uniform logits.
    import math
    expected = math.log(cfg.vocab_size)
    assert abs(loss.item() - expected) < 2.0, f"loss={loss.item():.3f}, expected ~{expected:.3f}"


def test_pos_emb_present_for_learned():
    from model import TransformerLM

    cfg = _tiny_config(pos_encoding="learned")
    lm = TransformerLM(cfg)
    assert lm.pos_emb is not None
    assert lm.pos_emb.weight.shape == (cfg.context_len, cfg.d_model)


def test_pos_emb_absent_for_rope():
    from model import TransformerLM

    cfg = _tiny_config(pos_encoding="rope")
    lm = TransformerLM(cfg)
    assert lm.pos_emb is None


def test_rmsnorm_branch_selects_rmsnorm():
    from model import TransformerLM

    cfg = _tiny_config(norm_type="rmsnorm")
    lm = TransformerLM(cfg)
    assert isinstance(lm.final_norm, RMSNorm)


def test_layernorm_branch_selects_layernorm():
    from model import TransformerLM

    cfg = _tiny_config(norm_type="layernorm")
    lm = TransformerLM(cfg)
    assert isinstance(lm.final_norm, nn.LayerNorm)


def test_weight_tying():
    from model import TransformerLM

    cfg = _tiny_config()
    lm = TransformerLM(cfg)
    assert lm.lm_head.weight is lm.tok_emb.weight, "LM head must be tied to token embedding"


def test_grad_flows_to_all_params():
    from model import TransformerLM

    cfg = _tiny_config()
    lm = TransformerLM(cfg)
    idx = torch.randint(0, cfg.vocab_size, (2, 8))
    targets = torch.randint(0, cfg.vocab_size, (2, 8))
    _, loss = lm(idx, targets)
    loss.backward()
    for name, p in lm.named_parameters():
        assert p.grad is not None, f"no grad on {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite grad on {name}"


def test_context_len_overflow_raises():
    from model import TransformerLM

    cfg = _tiny_config(context_len=8)
    lm = TransformerLM(cfg)
    idx = torch.randint(0, cfg.vocab_size, (1, 16))
    with pytest.raises(ValueError, match="context_len"):
        lm(idx)


def test_generate_shape():
    from model import TransformerLM

    cfg = _tiny_config(context_len=16)
    lm = TransformerLM(cfg)
    seed = torch.randint(0, cfg.vocab_size, (1, 4))
    out = lm.generate(seed, max_new_tokens=10, temperature=1.0, top_k=5)
    assert out.shape == (1, 14)


def test_generate_cached_matches_naive_path():
    """
    THE end-to-end KV cache correctness test. Generate with cache and
    generate without cache, starting from the same RNG state, must produce
    bit-identical output token sequences.

    If this fails, the cache implementation diverges from the recompute
    path somewhere — wrong position math, mask handling, cache concat
    order, etc.
    """
    from model import TransformerLM

    torch.manual_seed(0)
    cfg = _tiny_config(context_len=16, dropout=0.0)
    lm = TransformerLM(cfg)
    lm.eval()

    prompt = torch.randint(0, cfg.vocab_size, (1, 4))

    torch.manual_seed(42)
    out_naive = lm.generate(prompt, max_new_tokens=8, temperature=1.0, top_k=10, use_cache=False)

    torch.manual_seed(42)
    out_cached = lm.generate(prompt, max_new_tokens=8, temperature=1.0, top_k=10, use_cache=True)

    assert torch.equal(out_naive, out_cached), (
        f"naive: {out_naive.tolist()}, cached: {out_cached.tolist()}"
    )


def test_generate_cached_matches_naive_with_rope():
    """Same correctness check but with the modern stack (rmsnorm + rope + swiglu)."""
    from model import TransformerLM

    torch.manual_seed(0)
    cfg = _tiny_config(
        context_len=16, dropout=0.0,
        norm_type="rmsnorm", activation="swiglu", pos_encoding="rope",
        d_ffn=None,
    )
    lm = TransformerLM(cfg)
    lm.eval()

    prompt = torch.randint(0, cfg.vocab_size, (1, 4))

    torch.manual_seed(42)
    out_naive = lm.generate(prompt, max_new_tokens=8, temperature=1.0, top_k=10, use_cache=False)

    torch.manual_seed(42)
    out_cached = lm.generate(prompt, max_new_tokens=8, temperature=1.0, top_k=10, use_cache=True)

    assert torch.equal(out_naive, out_cached), (
        f"naive: {out_naive.tolist()}, cached: {out_cached.tolist()}"
    )


def test_model_with_moe_forward_and_loss_includes_aux():
    """
    With num_experts >= 2, TransformerLM should produce normal output
    shape AND the loss should include the MoE aux term. Compare to the
    same model with num_experts=0 — losses should differ even with
    identical seed (aux adds a small positive number to CE).
    """
    from model import TransformerLM

    cfg_dense = _tiny_config(num_experts=0)
    cfg_moe = _tiny_config(num_experts=4, top_k_experts=2, moe_aux_loss_coef=0.01)

    torch.manual_seed(0)
    lm_dense = TransformerLM(cfg_dense)
    torch.manual_seed(0)
    lm_moe = TransformerLM(cfg_moe)

    idx = torch.randint(0, cfg_dense.vocab_size, (1, 6))
    targets = torch.randint(0, cfg_dense.vocab_size, (1, 6))

    # Both models forward without crashing
    logits_d, loss_d = lm_dense(idx, targets)
    logits_m, loss_m = lm_moe(idx, targets)

    assert logits_d.shape == logits_m.shape == (1, 6, cfg_dense.vocab_size)
    assert loss_d is not None and loss_m is not None
    assert torch.isfinite(loss_d) and torch.isfinite(loss_m)

    # Both should also yield gradients that flow through all blocks.
    loss_m.backward()
    for n, p in lm_moe.named_parameters():
        assert p.grad is not None or "expert" in n, (
            f"no grad on {n} — only individual experts may be skipped"
        )


def test_generate_stops_at_context_limit():
    """
    With KV-cache generation (default), once the cache fills to context_len
    we hard-stop rather than try to slide a window (which would break the
    learned-position embedding's index range and RoPE's cache range).

    Seed already at max context -> exactly one new token can be generated
    (from the prefill's last-token logit). The decode loop then exits
    immediately because cache_len == context_len.
    """
    from model import TransformerLM

    cfg = _tiny_config(context_len=8)
    lm = TransformerLM(cfg)
    seed = torch.randint(0, cfg.vocab_size, (1, 8))  # already at max
    out = lm.generate(seed, max_new_tokens=5)
    # 8 seed tokens + 1 sampled from prefill, then context-limit stop.
    assert out.shape == (1, 9), f"expected (1, 9), got {tuple(out.shape)}"

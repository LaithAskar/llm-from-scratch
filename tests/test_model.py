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


def test_generate_truncates_long_context():
    from model import TransformerLM

    cfg = _tiny_config(context_len=8)
    lm = TransformerLM(cfg)
    seed = torch.randint(0, cfg.vocab_size, (1, 8))  # already at max
    out = lm.generate(seed, max_new_tokens=5)
    assert out.shape == (1, 13)  # should NOT raise; should truncate internally

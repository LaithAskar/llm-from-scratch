"""
Tests for sft_data.py (mask construction, batch sampling) and sft.py
(end-to-end one-step smoke).

Run: python -m pytest tests/test_sft.py -v
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from sft_data import (
    PREFIX,
    SFTDataset,
    build_split,
    get_sft_batch,
    iter_stories,
)
import tiktoken


def _make_raw(tmp_path: Path, n_stories: int = 30) -> Path:
    """Synthetic TinyStories-style file: stories separated by <|endoftext|>."""
    parts = []
    for i in range(n_stories):
        parts.append(
            f"Once upon a time, there was a cat named Cat{i}. Cat{i} liked to play.\n"
        )
        parts.append("<|endoftext|>\n")
    p = tmp_path / "raw.txt"
    p.write_text("".join(parts), encoding="utf-8")
    return p


def test_iter_stories_splits_on_eot(tmp_path: Path):
    raw = _make_raw(tmp_path, n_stories=5)
    stories = list(iter_stories(raw))
    assert len(stories) == 5
    for i, s in enumerate(stories):
        assert f"Cat{i}" in s
        assert "<|endoftext|>" not in s


def _prepare_split(tmp_path: Path, n: int = 20) -> tuple[Path, Path, int]:
    raw = _make_raw(tmp_path, n_stories=n)
    enc = tiktoken.get_encoding("gpt2")
    prefix_ids = enc.encode_ordinary(PREFIX)
    eot_id = enc.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0]
    tok = tmp_path / "tokens.bin"
    msk = tmp_path / "mask.bin"
    build_split(raw, tok, msk, enc, prefix_ids, eot_id, max_stories=None)
    return tok, msk, len(prefix_ids)


def test_mask_zero_on_prefix_one_elsewhere(tmp_path: Path):
    tok, msk, n_prefix = _prepare_split(tmp_path)
    tokens = np.fromfile(tok, dtype=np.uint16)
    mask = np.fromfile(msk, dtype=np.uint8)
    assert tokens.size == mask.size

    # First n_prefix positions of EACH example must be mask=0; rest mask=1.
    # We can verify by scanning for transitions. With 20 stories, the mask
    # should have exactly 20 "blocks of zeros" each n_prefix long, separated
    # by stretches of ones.
    n_zero_total = int((mask == 0).sum())
    assert n_zero_total == 20 * n_prefix, (
        f"expected {20 * n_prefix} masked-out tokens, got {n_zero_total}"
    )

    # The very first token must be mask=0 (start of first example's prefix).
    assert mask[0] == 0
    # The very last token is the last <|endoftext|>, which is in the completion -> mask=1.
    assert mask[-1] == 1


def test_get_sft_batch_sets_targets_to_minus_100_where_mask_zero(tmp_path: Path):
    tok, msk, _ = _prepare_split(tmp_path, n=40)
    ds = SFTDataset(tok, msk)
    rng = np.random.default_rng(0)
    x, y, m = get_sft_batch(ds, batch_size=4, context_len=32, device="cpu", rng=rng)

    # Wherever mask is False, target must be -100. Wherever True, target must be a real token id (>=0).
    masked_out = (m == False)
    masked_in = (m == True)
    assert torch.all(y[masked_out] == -100)
    assert torch.all(y[masked_in] >= 0)


def test_get_sft_batch_has_some_unmasked_positions(tmp_path: Path):
    """Sanity: if every position got masked, training would do nothing."""
    tok, msk, _ = _prepare_split(tmp_path, n=40)
    ds = SFTDataset(tok, msk)
    rng = np.random.default_rng(0)
    _, _, m = get_sft_batch(ds, batch_size=8, context_len=64, device="cpu", rng=rng)
    frac_unmasked = m.float().mean().item()
    assert frac_unmasked > 0.5, (
        f"only {frac_unmasked:.2%} of positions are unmasked — mostly-prefix "
        f"windows would mean SFT learns nothing"
    )


def test_sft_one_step_finite_loss_and_grad(tmp_path: Path):
    """End-to-end: build a tiny model, load synthetic SFT data, take one step.

    Verifies the loss is finite, gradients flow, and the masking integrates
    cleanly with the existing model.forward path.
    """
    from config import ModelConfig
    from model import TransformerLM

    tok, msk, _ = _prepare_split(tmp_path, n=80)
    ds = SFTDataset(tok, msk)

    cfg = ModelConfig(
        vocab_size=50257,
        n_layer=2, n_head=2, d_model=64, context_len=64,
        dropout=0.0, bias=False,
    )
    model = TransformerLM(cfg)
    model.train()
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4)

    rng = np.random.default_rng(0)
    x, y, _ = get_sft_batch(ds, batch_size=4, context_len=cfg.context_len, device="cpu", rng=rng)

    _, loss = model(x, y)
    assert torch.isfinite(loss).item(), f"loss is not finite: {loss}"
    loss.backward()

    # At least one parameter should have a non-zero gradient.
    has_grad = any(
        p.grad is not None and torch.any(p.grad != 0).item()
        for p in model.parameters()
    )
    assert has_grad, "no parameter received a non-zero gradient"

    optim.step()  # smoke: optimizer doesn't blow up on the gradients

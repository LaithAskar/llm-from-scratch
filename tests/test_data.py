"""
Lightweight smoke tests for data.py — synthetic text, no network.
Run: python -m pytest tests/test_data.py -v
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from data import TokenDataset, get_batch, prepare


def _make_synthetic_corpus(tmp_path: Path, n_chars: int = 50_000) -> Path:
    # Repeating a short text fragment many times gives plenty of tokens for
    # batching while staying trivial to tokenize.
    text = ("Once upon a time, there was a small cat who liked to sit in the sun. " * 1000)[:n_chars]
    src = tmp_path / "source.txt"
    src.write_text(text, encoding="utf-8")
    return src


def test_prepare_and_load(tmp_path: Path):
    src = _make_synthetic_corpus(tmp_path)
    out = tmp_path / "out"
    prepare(str(out), source=str(src))

    train_bin = out / "train.bin"
    val_bin = out / "val.bin"
    assert train_bin.exists() and val_bin.exists()
    assert train_bin.stat().st_size > 0
    assert val_bin.stat().st_size > 0


def test_token_dataset_memmap(tmp_path: Path):
    src = _make_synthetic_corpus(tmp_path)
    out = tmp_path / "out"
    prepare(str(out), source=str(src))

    ds = TokenDataset(out / "train.bin")
    assert len(ds) > 0
    assert ds.data.dtype == np.uint16
    # All token ids must be in valid GPT-2 vocab range.
    assert ds.data.max() < 50257


def test_get_batch_shapes_and_shift(tmp_path: Path):
    src = _make_synthetic_corpus(tmp_path)
    out = tmp_path / "out"
    prepare(str(out), source=str(src))

    ds = TokenDataset(out / "train.bin")
    rng = np.random.default_rng(0)
    x, y = get_batch(ds, batch_size=4, context_len=64, device="cpu", rng=rng)

    assert x.shape == (4, 64)
    assert y.shape == (4, 64)
    assert x.dtype.is_floating_point is False  # int64

    # y must be x shifted by 1: in fact y[:, :-1] == x[:, 1:] (both come from contiguous slices)
    assert (y[:, :-1] == x[:, 1:]).all(), "y is not a shifted view of x"


def test_get_batch_too_small_dataset(tmp_path: Path):
    out = tmp_path / "out"
    out.mkdir()
    bin_path = out / "tiny.bin"
    # Only 10 tokens
    np.arange(10, dtype=np.uint16).tofile(bin_path)
    ds = TokenDataset(bin_path)
    with pytest.raises(ValueError, match="context_len"):
        get_batch(ds, batch_size=1, context_len=64, device="cpu")


def test_missing_bin_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="data.py prepare"):
        TokenDataset(tmp_path / "does_not_exist.bin")

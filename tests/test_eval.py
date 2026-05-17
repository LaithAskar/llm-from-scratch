"""
Smoke tests for eval.py — train a tiny model, then evaluate the resulting checkpoint.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest


# --- Fixture: train a tiny model and yield its ckpt + data dirs ------------


@pytest.fixture
def trained_run(tmp_path: Path):
    from config import Config, ModelConfig, TrainConfig
    from train import train

    data_dir = tmp_path / "data"
    out_dir = tmp_path / "run"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Use real tiktoken vocab so eval.sample's tokenizer is in range.
    vocab_size = 50257
    rng = np.random.default_rng(0)
    train_arr = rng.integers(0, vocab_size, size=4096, dtype=np.uint16)
    val_arr = rng.integers(0, vocab_size, size=1024, dtype=np.uint16)
    train_arr.tofile(data_dir / "train.bin")
    val_arr.tofile(data_dir / "val.bin")

    cfg = Config(
        model=ModelConfig(
            vocab_size=vocab_size,
            n_layer=2, n_head=4, d_model=32, context_len=16, dropout=0.0,
        ),
        train=TrainConfig(
            lr=1e-3, min_lr=1e-4, weight_decay=0.0,
            micro_batch_size=4, grad_accum_steps=1,
            max_steps=3, warmup_steps=1,
            eval_every=2, eval_iters=2,
            ckpt_every=1000, log_every=10,
            out_dir=str(out_dir),
            data_dir=str(data_dir),
            seed=0, dtype="fp32", compile=False,
        ),
    )
    train(cfg)
    return out_dir / "final.pt", data_dir


# --- Tests -----------------------------------------------------------------


def test_load_checkpoint_returns_model_and_config(trained_run):
    from eval import load_checkpoint

    ckpt_path, _ = trained_run
    model, config, device = load_checkpoint(ckpt_path, device="cpu")
    assert model.training is False
    assert config.model.d_model == 32
    assert device == "cpu"


def test_compute_perplexity_returns_positive_float(trained_run):
    from eval import compute_perplexity

    ckpt_path, data_dir = trained_run
    ppl = compute_perplexity(ckpt_path, data_dir, n_iters=3, batch_size=4, device="cpu")
    assert ppl > 0
    assert math.isfinite(ppl)


def test_sample_returns_string(trained_run):
    from eval import sample

    ckpt_path, _ = trained_run
    text = sample(ckpt_path, prompt="hello world", max_tokens=5, top_k=10, device="cpu")
    assert isinstance(text, str)
    assert len(text) > len("hello world")  # added at least something


def test_sample_works_with_empty_prompt(trained_run):
    from eval import sample

    ckpt_path, _ = trained_run
    text = sample(ckpt_path, prompt="", max_tokens=3, device="cpu")
    assert isinstance(text, str)
    assert len(text) > 0

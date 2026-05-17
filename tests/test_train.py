"""
End-to-end smoke test for train.py.

Uses dummy stand-ins for TransformerBlock/RMSNorm so the loop wiring can
be validated before layers.py is fully implemented. Synthesized data, 3
training steps, CPU-only.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
import torch


# --- Helpers ---------------------------------------------------------------


def _make_data(data_dir: Path, vocab_size: int = 257, n_train: int = 4096, n_val: int = 1024):
    data_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    train = rng.integers(0, vocab_size, size=n_train, dtype=np.uint16)
    val = rng.integers(0, vocab_size, size=n_val, dtype=np.uint16)
    train.tofile(data_dir / "train.bin")
    val.tofile(data_dir / "val.bin")


def _make_config(out_dir: Path, data_dir: Path):
    from config import Config, ModelConfig, TrainConfig

    return Config(
        model=ModelConfig(
            vocab_size=257,
            n_layer=2,
            n_head=4,
            d_model=32,
            context_len=16,
            dropout=0.0,
        ),
        train=TrainConfig(
            lr=1e-3,
            min_lr=1e-4,
            weight_decay=0.0,
            micro_batch_size=4,
            grad_accum_steps=1,
            max_steps=3,
            warmup_steps=1,
            eval_every=2,
            eval_iters=2,
            ckpt_every=1000,
            log_every=1,
            out_dir=str(out_dir),
            data_dir=str(data_dir),
            seed=0,
            dtype="fp32",  # avoid CPU autocast platform quirks in tests
            compile=False,
        ),
    )


# --- Tests -----------------------------------------------------------------


def test_get_lr_warmup_and_decay():
    from config import TrainConfig
    from train import get_lr

    tc = TrainConfig(lr=1.0, min_lr=0.1, warmup_steps=10, max_steps=100)
    assert get_lr(0, tc) == pytest.approx(0.1, abs=1e-9)        # step 0 -> 1/10 of lr
    assert get_lr(9, tc) == pytest.approx(1.0, abs=1e-9)        # end of warmup
    # past max_steps -> min_lr
    assert get_lr(200, tc) == pytest.approx(0.1, abs=1e-9)
    # mid-decay should be between min_lr and lr
    mid = get_lr(55, tc)
    assert 0.1 < mid < 1.0


def test_param_groups_split_correctly():
    from config import ModelConfig
    from model import TransformerLM
    from train import make_param_groups

    cfg = ModelConfig(vocab_size=257, n_layer=1, n_head=2, d_model=16, context_len=8)
    lm = TransformerLM(cfg)
    groups = make_param_groups(lm, weight_decay=0.1)
    assert len(groups) == 2
    assert groups[0]["weight_decay"] == 0.1
    assert groups[1]["weight_decay"] == 0.0
    # all 2-D non-emb-non-norm linears go to decay; the rest to no_decay
    n_decay = sum(p.numel() for p in groups[0]["params"])
    n_no_decay = sum(p.numel() for p in groups[1]["params"])
    assert n_decay > 0 and n_no_decay > 0


def test_train_runs_end_to_end(tmp_path: Path):
    from train import train

    data_dir = tmp_path / "data"
    out_dir = tmp_path / "run"
    _make_data(data_dir)
    cfg = _make_config(out_dir, data_dir)

    result = train(cfg)

    # files exist
    assert (out_dir / "config.json").exists()
    assert (out_dir / "log.csv").exists()
    assert (out_dir / "final.pt").exists()

    # we asked for max_steps=3, eval_every=2: an eval should have fired at step 2
    # which means best.pt should exist
    assert (out_dir / "best.pt").exists()

    assert result["final_step"] == 3
    assert np.isfinite(result["best_val"])


def test_train_log_csv_has_expected_columns(tmp_path: Path):
    from train import train

    data_dir = tmp_path / "data"
    out_dir = tmp_path / "run"
    _make_data(data_dir)
    cfg = _make_config(out_dir, data_dir)
    train(cfg)

    with open(out_dir / "log.csv") as f:
        rows = list(csv.reader(f))

    assert rows[0] == ["step", "split", "loss", "lr", "grad_norm", "wallclock_s"]
    # at least one train row + one val row
    splits = {r[1] for r in rows[1:]}
    assert "train" in splits
    assert "val" in splits

    # all loss values must be finite floats
    for r in rows[1:]:
        loss = float(r[2])
        assert np.isfinite(loss), f"non-finite loss in log row: {r}"


def test_train_checkpoint_is_loadable(tmp_path: Path):
    from train import train

    data_dir = tmp_path / "data"
    out_dir = tmp_path / "run"
    _make_data(data_dir)
    cfg = _make_config(out_dir, data_dir)
    train(cfg)

    ckpt = torch.load(out_dir / "final.pt", weights_only=False)
    assert "model" in ckpt and "optim" in ckpt and "config" in ckpt
    assert ckpt["step"] == 3

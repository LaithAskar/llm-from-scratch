"""
Smoke test for ablation.py: run 2 variants × 1 seed × 3 training steps,
verify the summary CSV exists and has all expected rows.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest


# --- Tests -----------------------------------------------------------------


def test_variants_dict_contents():
    from ablation import variants

    v = variants()
    assert set(v.keys()) == {"baseline", "rmsnorm", "rope", "swiglu", "modern"}
    assert v["baseline"].norm_type == "layernorm"
    assert v["rmsnorm"].norm_type == "rmsnorm"
    assert v["rope"].pos_encoding == "rope"
    assert v["swiglu"].activation == "swiglu"
    assert v["modern"].norm_type == "rmsnorm"
    assert v["modern"].activation == "swiglu"
    assert v["modern"].pos_encoding == "rope"

    # d_ffn should auto-resolve to (8/3)*d_model rounded for swiglu variants
    base = v["baseline"]
    assert v["swiglu"].d_ffn != 4 * base.d_model  # different from gelu default
    assert v["modern"].d_ffn == v["swiglu"].d_ffn


def test_run_ablation_smoke(tmp_path: Path):
    from ablation import run_ablation
    from config import ModelConfig, TrainConfig

    data_dir = tmp_path / "data"
    out_root = tmp_path / "abl"
    data_dir.mkdir()

    # Small synthetic data
    vocab_size = 257
    rng = np.random.default_rng(0)
    rng.integers(0, vocab_size, size=4096, dtype=np.uint16).tofile(data_dir / "train.bin")
    rng.integers(0, vocab_size, size=1024, dtype=np.uint16).tofile(data_dir / "val.bin")

    tiny_model = ModelConfig(
        vocab_size=vocab_size, n_layer=2, n_head=4, d_model=32, context_len=16, dropout=0.0,
    )
    tiny_train = TrainConfig(
        lr=1e-3, min_lr=1e-4, weight_decay=0.0,
        micro_batch_size=4, grad_accum_steps=1,
        max_steps=3, warmup_steps=1, eval_every=2, eval_iters=2,
        ckpt_every=1000, log_every=10, dtype="fp32", compile=False,
    )

    summary = run_ablation(
        out_root=out_root,
        data_dir=str(data_dir),
        seeds=[0],
        base_model=tiny_model,
        base_train=tiny_train,
        variant_names=["baseline", "rmsnorm"],
    )

    assert summary.exists()
    with open(summary) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2  # 2 variants × 1 seed

    variants_seen = {r["variant"] for r in rows}
    assert variants_seen == {"baseline", "rmsnorm"}
    for r in rows:
        assert int(r["seed"]) == 0
        assert int(r["final_step"]) == 3
        best_val = float(r["best_val"])
        assert np.isfinite(best_val)

    # Per-variant subdirs exist
    assert (out_root / "baseline" / "seed_0" / "log.csv").exists()
    assert (out_root / "rmsnorm" / "seed_0" / "log.csv").exists()


def test_unknown_variant_raises(tmp_path: Path):
    from ablation import run_ablation

    with pytest.raises(ValueError, match="unknown variants"):
        run_ablation(
            out_root=tmp_path,
            data_dir=str(tmp_path),
            seeds=[0],
            variant_names=["definitely-not-a-variant"],
        )

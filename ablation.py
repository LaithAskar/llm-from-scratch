"""
Ablation runner: trains each variant for N seeds, all sharing identical
training config and model size — only the three architecture switches
differ.

Variants:
    baseline : layernorm + gelu    + learned-pos
    rmsnorm  : rmsnorm   + gelu    + learned-pos
    rope     : layernorm + gelu    + rope
    swiglu   : layernorm + swiglu  + learned-pos
    modern   : rmsnorm   + swiglu  + rope

Outputs:
    {out_root}/{variant}/seed_{n}/
        config.json, log.csv, best.pt, final.pt
    {out_root}/summary.csv
        variant, seed, best_val, final_step, ppl_val, wall_clock_s

CLI:
    python ablation.py --out runs/ablation --data data/tinystories --seeds 0 1 2
    python ablation.py --variants baseline rmsnorm  # subset
    python ablation.py --max-steps 500              # quick smoke run
"""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import replace
from pathlib import Path

from config import Config, ModelConfig, TrainConfig
from eval import compute_perplexity
from train import train


def base_model_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=50257,
        n_layer=4, n_head=6, d_model=192, context_len=256,
        dropout=0.1, bias=False,
        norm_type="layernorm", activation="gelu", pos_encoding="learned",
    )


def base_train_config() -> TrainConfig:
    return TrainConfig(
        lr=3e-4, min_lr=3e-5, weight_decay=0.1,
        micro_batch_size=16, grad_accum_steps=2,
        max_steps=5000, warmup_steps=200,
        eval_every=250, eval_iters=50, ckpt_every=2500, log_every=20,
        dtype="bf16", compile=False,
    )


def variants(base: ModelConfig | None = None) -> dict[str, ModelConfig]:
    """
    Returns the five ablation variants.

    Each variant uses `replace` so the base model size, vocab, context, etc.
    are identical — only the three switches differ. d_ffn is re-resolved to
    None for swiglu variants so __post_init__ applies the PaLM (8/3)*d_model
    matched-FFN sizing.
    """
    b = base or base_model_config()
    return {
        "baseline": replace(b),
        "rmsnorm":  replace(b, norm_type="rmsnorm"),
        "rope":     replace(b, pos_encoding="rope"),
        "swiglu":   replace(b, activation="swiglu", d_ffn=None),
        "modern":   replace(b, norm_type="rmsnorm", activation="swiglu", pos_encoding="rope", d_ffn=None),
    }


def run_ablation(
    out_root: Path,
    data_dir: str,
    seeds: list[int],
    base_model: ModelConfig | None = None,
    base_train: TrainConfig | None = None,
    variant_names: list[str] | None = None,
    train_overrides: dict | None = None,
) -> Path:
    out_root.mkdir(parents=True, exist_ok=True)
    base_model = base_model or base_model_config()
    base_train = base_train or base_train_config()

    all_variants = variants(base_model)
    if variant_names:
        unknown = set(variant_names) - set(all_variants)
        if unknown:
            raise ValueError(f"unknown variants: {sorted(unknown)} (have: {sorted(all_variants)})")
        all_variants = {k: all_variants[k] for k in variant_names}

    rows = []
    for vname, mcfg in all_variants.items():
        for seed in seeds:
            run_dir = out_root / vname / f"seed_{seed}"
            print(f"\n=== variant={vname} seed={seed} ===")

            tcfg = replace(
                base_train,
                out_dir=str(run_dir),
                data_dir=data_dir,
                seed=seed,
            )
            if train_overrides:
                tcfg = replace(tcfg, **train_overrides)

            cfg = Config(model=mcfg, train=tcfg, name=f"{vname}_seed{seed}")
            t0 = time.time()
            result = train(cfg)
            wall = time.time() - t0

            best_ckpt = run_dir / "best.pt"
            ppl = float("nan")
            if best_ckpt.exists():
                ppl = compute_perplexity(
                    best_ckpt, data_dir,
                    n_iters=min(100, tcfg.eval_iters * 2),
                    batch_size=tcfg.micro_batch_size,
                )

            rows.append({
                "variant": vname,
                "seed": seed,
                "best_val": result["best_val"],
                "final_step": result["final_step"],
                "ppl_val": ppl,
                "wall_clock_s": wall,
            })

    summary_path = out_root / "summary.csv"
    fieldnames = ["variant", "seed", "best_val", "final_step", "ppl_val", "wall_clock_s"]
    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"\nWrote summary -> {summary_path}")
    return summary_path


def _main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="runs/ablation")
    p.add_argument("--data", default="data/tinystories")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument(
        "--variants", nargs="+", default=None,
        help=f"Subset of {list(variants().keys())}. Default: all.",
    )
    p.add_argument("--max-steps", type=int, default=None, help="Override TrainConfig.max_steps")
    p.add_argument("--micro-batch-size", type=int, default=None)
    args = p.parse_args()

    overrides = {}
    if args.max_steps is not None:
        overrides["max_steps"] = args.max_steps
    if args.micro_batch_size is not None:
        overrides["micro_batch_size"] = args.micro_batch_size

    run_ablation(
        out_root=Path(args.out),
        data_dir=args.data,
        seeds=args.seeds,
        variant_names=args.variants,
        train_overrides=overrides,
    )


if __name__ == "__main__":
    _main()

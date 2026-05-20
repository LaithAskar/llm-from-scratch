"""
Plot ablation results from runs/ablation/.

Reads per-run log.csv files and the summary.csv, produces:
    figures/loss_curves_train.png   — train-loss curves, one line per variant
    figures/loss_curves_val.png     — val-loss curves, one line per variant
    figures/best_val_bar.png        — best val loss per variant (bar chart)

If multiple seeds are present per variant, shows mean and shades min/max.

CLI:
    python plot_ablation.py                              # default: runs/ablation
    python plot_ablation.py --root runs/ablation_smoke
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _read_log(log_path: Path, split: str) -> tuple[list[int], list[float]]:
    steps, losses = [], []
    with open(log_path) as f:
        r = csv.DictReader(f)
        for row in r:
            if row["split"] != split:
                continue
            steps.append(int(row["step"]))
            losses.append(float(row["loss"]))
    return steps, losses


def _gather_curves(root: Path, split: str) -> dict[str, list[tuple[list[int], list[float]]]]:
    """variant -> list of (steps, losses) across seeds."""
    out: dict[str, list[tuple[list[int], list[float]]]] = defaultdict(list)
    for variant_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for seed_dir in sorted(p for p in variant_dir.iterdir() if p.is_dir() and p.name.startswith("seed_")):
            log = seed_dir / "log.csv"
            if not log.exists():
                continue
            steps, losses = _read_log(log, split)
            if steps:
                out[variant_dir.name].append((steps, losses))
    return out


def _plot_curves(curves: dict[str, list[tuple[list[int], list[float]]]], title: str, out_path: Path):
    fig, ax = plt.subplots(figsize=(8, 5))
    for variant, runs in curves.items():
        if not runs:
            continue
        # Align on common step grid (assume all runs share the same schedule).
        steps = runs[0][0]
        losses_matrix = np.array([r[1] for r in runs if len(r[1]) == len(steps)])
        if losses_matrix.size == 0:
            continue
        mean = losses_matrix.mean(axis=0)
        ax.plot(steps, mean, label=variant, linewidth=1.6)
        if losses_matrix.shape[0] > 1:
            lo, hi = losses_matrix.min(axis=0), losses_matrix.max(axis=0)
            ax.fill_between(steps, lo, hi, alpha=0.15)
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  wrote {out_path}")


def _plot_bar(summary_csv: Path, out_path: Path):
    rows: dict[str, list[float]] = defaultdict(list)
    with open(summary_csv) as f:
        r = csv.DictReader(f)
        for row in r:
            rows[row["variant"]].append(float(row["best_val"]))
    variants = list(rows.keys())
    means = [np.mean(rows[v]) for v in variants]
    mins = [np.min(rows[v]) for v in variants]
    maxs = [np.max(rows[v]) for v in variants]
    err_lo = [m - lo for m, lo in zip(means, mins)]
    err_hi = [hi - m for m, hi in zip(means, maxs)]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(variants, means, yerr=[err_lo, err_hi], capsize=4, color="#4C72B0")
    ax.set_ylabel("best val loss")
    ax.set_title("Best validation loss per variant (lower is better)")
    ax.grid(True, axis="y", alpha=0.3)
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, m, f"{m:.3f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  wrote {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="runs/ablation")
    args = p.parse_args()
    root = Path(args.root)
    if not root.exists():
        raise FileNotFoundError(f"{root} not found — run ablation.py first")

    figdir = root / "figures"
    train_curves = _gather_curves(root, "train")
    val_curves = _gather_curves(root, "val")
    _plot_curves(train_curves, "Train loss by variant", figdir / "loss_curves_train.png")
    _plot_curves(val_curves, "Validation loss by variant", figdir / "loss_curves_val.png")

    summary = root / "summary.csv"
    if summary.exists():
        _plot_bar(summary, figdir / "best_val_bar.png")
    else:
        print(f"  no {summary} found, skipping bar chart")


if __name__ == "__main__":
    main()

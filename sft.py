"""
Supervised fine-tuning: load a pretrained checkpoint, fine-tune at a lower
LR on (prompt, completion) pairs where loss is masked on prompt tokens.

The masking itself happens inside sft_data.get_sft_batch (sets target
positions to -100; cross_entropy then ignores them via its ignore_index).
This file is the training driver.

Run:
    python sft.py \
        --base runs/ablation/modern/seed_1/best.pt \
        --data data/sft \
        --out runs/sft/modern \
        --max-steps 500 \
        --lr 3e-5

Outputs (in out_dir):
    config.json   — frozen snapshot of the SFT Config
    log.csv       — step, split, loss, lr, grad_norm, wallclock_s
    best.pt       — best val-loss checkpoint
    final.pt      — last step
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from config import Config, ModelConfig
from model import TransformerLM
from sft_data import SFTDataset, get_sft_batch

_AMP_DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}


@dataclass
class SFTConfig:
    base_ckpt: str
    data_dir: str = "data/sft"
    out_dir: str = "runs/sft/default"

    # optimization — defaults are SFT-appropriate (lower than pretrain).
    lr: float = 3e-5
    min_lr: float = 3e-6
    weight_decay: float = 0.0  # standard practice for SFT: no decay
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    micro_batch_size: int = 8
    grad_accum_steps: int = 2
    max_steps: int = 500
    warmup_steps: int = 50

    eval_every: int = 100
    eval_iters: int = 20
    log_every: int = 20

    seed: int = 0
    dtype: str = "bf16"

    def to_json(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(__import__("json").dumps(asdict(self), indent=2))


def _autocast(device: str, dtype: torch.dtype):
    if dtype == torch.float32:
        return nullcontext()
    device_type = "cuda" if device.startswith("cuda") else "cpu"
    return torch.amp.autocast(device_type=device_type, dtype=dtype)


def get_sft_lr(step: int, sc: SFTConfig) -> float:
    if step < sc.warmup_steps:
        return sc.lr * (step + 1) / max(1, sc.warmup_steps)
    if step >= sc.max_steps:
        return sc.min_lr
    progress = (step - sc.warmup_steps) / max(1, sc.max_steps - sc.warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return sc.min_lr + coeff * (sc.lr - sc.min_lr)


def load_base(ckpt_path: str, device: str) -> tuple[TransformerLM, Config]:
    blob = torch.load(ckpt_path, map_location=device, weights_only=False)
    base_cfg: Config = blob["config"]
    # Defensive: torch may rehydrate dataclass as dict-like if classes change.
    if not isinstance(base_cfg.model, ModelConfig):
        base_cfg.model = ModelConfig(**base_cfg.model)  # type: ignore[arg-type]
    model = TransformerLM(base_cfg.model).to(device)
    model.load_state_dict(blob["model"])
    return model, base_cfg


@torch.no_grad()
def evaluate_sft(model, dataset, n_iters, batch_size, context_len, device, amp_dtype, rng) -> float:
    was_training = model.training
    model.eval()
    total = 0.0
    for _ in range(n_iters):
        x, y, _ = get_sft_batch(dataset, batch_size, context_len, device, rng=rng)
        with _autocast(device, amp_dtype):
            _, loss = model(x, y)
        total += loss.item()
    if was_training:
        model.train()
    return total / n_iters


def _save(path, model, optim, base_cfg, sft_cfg, step, val_loss):
    torch.save(
        {
            "model": model.state_dict(),
            "optim": optim.state_dict(),
            "step": step,
            "val_loss": val_loss,
            "config": base_cfg,         # same shape as pretrain ckpts: keeps generate_sample.py happy
            "sft_config": asdict(sft_cfg),
        },
        path,
    )


def sft(sc: SFTConfig) -> dict:
    out_dir = Path(sc.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sc.to_json(out_dir / "config.json")

    torch.manual_seed(sc.seed)
    np.random.seed(sc.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(sc.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp_dtype = _AMP_DTYPES[sc.dtype]
    print(f"device: {device}, dtype: {sc.dtype}")

    model, base_cfg = load_base(sc.base_ckpt, device)
    print(f"loaded base from {sc.base_ckpt}")
    print(f"model params (non-emb): {model.num_params(non_embedding=True):,}")

    data_dir = Path(sc.data_dir)
    train_ds = SFTDataset(data_dir / "train_tokens.bin", data_dir / "train_mask.bin")
    val_ds = SFTDataset(data_dir / "val_tokens.bin", data_dir / "val_mask.bin")
    print(f"sft train tokens: {len(train_ds):,}   val tokens: {len(val_ds):,}")

    rng_train = np.random.default_rng(sc.seed)
    rng_val = np.random.default_rng(sc.seed + 1)

    # AdamW without weight decay (SFT standard practice — pretrained weights
    # are already trained with WD; further decay during SFT can damage them).
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=sc.lr,
        betas=(sc.beta1, sc.beta2),
        weight_decay=sc.weight_decay,
        fused=(device == "cuda"),
    )

    log_path = out_dir / "log.csv"
    best_val = float("inf")
    t_start = time.time()
    step = 0

    with open(log_path, "w", newline="") as log_f:
        w = csv.writer(log_f)
        w.writerow(["step", "split", "loss", "lr", "grad_norm", "wallclock_s"])

        model.train()
        while step < sc.max_steps:
            lr = get_sft_lr(step, sc)
            for g in optim.param_groups:
                g["lr"] = lr

            optim.zero_grad(set_to_none=True)
            accum_loss = 0.0
            for _ in range(sc.grad_accum_steps):
                x, y, _ = get_sft_batch(
                    train_ds, sc.micro_batch_size, base_cfg.model.context_len, device, rng=rng_train,
                )
                with _autocast(device, amp_dtype):
                    _, loss = model(x, y)
                    loss = loss / sc.grad_accum_steps
                loss.backward()
                accum_loss += loss.item()

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), sc.grad_clip)
            optim.step()

            if step % sc.log_every == 0:
                wall = time.time() - t_start
                w.writerow([step, "train", f"{accum_loss:.6f}", f"{lr:.6e}",
                            f"{float(grad_norm):.6f}", f"{wall:.2f}"])
                log_f.flush()
                print(f"step {step:5d} | train {accum_loss:.4f} | lr {lr:.2e} "
                      f"| gnorm {float(grad_norm):.3f} | {wall:.1f}s")

            if step > 0 and step % sc.eval_every == 0:
                val_loss = evaluate_sft(
                    model, val_ds, sc.eval_iters, sc.micro_batch_size,
                    base_cfg.model.context_len, device, amp_dtype, rng_val,
                )
                wall = time.time() - t_start
                w.writerow([step, "val", f"{val_loss:.6f}", f"{lr:.6e}",
                            "", f"{wall:.2f}"])
                log_f.flush()
                print(f"step {step:5d} | val   {val_loss:.4f}")
                if val_loss < best_val:
                    best_val = val_loss
                    _save(out_dir / "best.pt", model, optim, base_cfg, sc, step, val_loss)

            step += 1

    _save(out_dir / "final.pt", model, optim, base_cfg, sc, step, None)
    return {"best_val": best_val, "final_step": step}


def _main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True, help="Path to pretrained .pt checkpoint")
    p.add_argument("--data", default="data/sft")
    p.add_argument("--out", default="runs/sft/default")
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--micro-batch-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    sc = SFTConfig(
        base_ckpt=args.base,
        data_dir=args.data,
        out_dir=args.out,
        max_steps=args.max_steps,
        lr=args.lr,
        micro_batch_size=args.micro_batch_size,
        seed=args.seed,
    )
    return sft(sc)


if __name__ == "__main__":
    _main()

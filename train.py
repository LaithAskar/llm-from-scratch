"""
Training loop: AdamW + cosine-with-warmup LR + AMP + gradient accumulation
+ periodic val + checkpoints.

Run:
    python train.py                       # default Config()
    python train.py --config cfg.json     # load a saved Config
    python train.py --out runs/exp1       # override out_dir
    python train.py --data data/foo       # override data_dir

Outputs (in out_dir):
    config.json   — frozen snapshot of the Config used
    log.csv       — step, split, loss, lr, grad_norm, wallclock_s
    best.pt       — best val-loss checkpoint
    step_*.pt     — periodic checkpoints
    final.pt      — last step
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

from config import Config, TrainConfig
from data import TokenDataset, get_batch
from model import TransformerLM

_AMP_DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}


def get_lr(step: int, tc: TrainConfig) -> float:
    """Linear warmup -> cosine decay from lr to min_lr."""
    if step < tc.warmup_steps:
        return tc.lr * (step + 1) / max(1, tc.warmup_steps)
    if step >= tc.max_steps:
        return tc.min_lr
    progress = (step - tc.warmup_steps) / max(1, tc.max_steps - tc.warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return tc.min_lr + coeff * (tc.lr - tc.min_lr)


def make_param_groups(model: torch.nn.Module, weight_decay: float) -> list[dict]:
    """
    AdamW best-practice: decay on 2-D weight matrices, no decay on biases,
    norm parameters, and embeddings.
    Reference: GPT-3 paper §B.1; nanoGPT applies the same heuristic.
    """
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.dim() < 2 or "norm" in name.lower() or "emb" in name.lower():
            no_decay.append(p)
        else:
            decay.append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def _autocast(device: str, dtype: torch.dtype):
    """Build a one-shot autocast context for `with`."""
    if dtype == torch.float32:
        return nullcontext()
    device_type = "cuda" if device.startswith("cuda") else "cpu"
    return torch.amp.autocast(device_type=device_type, dtype=dtype)


@torch.no_grad()
def evaluate(
    model: TransformerLM,
    dataset: TokenDataset,
    n_iters: int,
    batch_size: int,
    context_len: int,
    device: str,
    amp_dtype: torch.dtype,
    rng: np.random.Generator,
) -> float:
    was_training = model.training
    model.eval()
    total = 0.0
    for _ in range(n_iters):
        x, y = get_batch(dataset, batch_size, context_len, device, rng=rng)
        with _autocast(device, amp_dtype):
            _, loss = model(x, y)
        total += loss.item()
    if was_training:
        model.train()
    return total / n_iters


def _save_ckpt(path: Path, model, optim, config: Config, step: int, val_loss):
    torch.save(
        {
            "model": model.state_dict(),
            "optim": optim.state_dict(),
            "step": step,
            "val_loss": val_loss,
            "config": config,
        },
        path,
    )


def train(config: Config) -> dict:
    out_dir = Path(config.train.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config.to_json(out_dir / "config.json")

    torch.manual_seed(config.train.seed)
    np.random.seed(config.train.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.train.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp_dtype = _AMP_DTYPES[config.train.dtype]
    print(f"device: {device}, dtype: {config.train.dtype}")

    data_dir = Path(config.train.data_dir)
    train_ds = TokenDataset(data_dir / "train.bin")
    val_ds = TokenDataset(data_dir / "val.bin")
    print(f"train tokens: {len(train_ds):,}   val tokens: {len(val_ds):,}")

    rng_train = np.random.default_rng(config.train.seed)
    rng_val = np.random.default_rng(config.train.seed + 1)

    model = TransformerLM(config.model).to(device)
    if config.train.compile:
        model = torch.compile(model)  # type: ignore[assignment]
    print(f"model params (non-emb): {model.num_params(non_embedding=True):,}")
    print(f"effective batch: {config.train.effective_batch_size}  "
          f"(micro {config.train.micro_batch_size} × accum {config.train.grad_accum_steps})")

    optim = torch.optim.AdamW(
        make_param_groups(model, config.train.weight_decay),
        lr=config.train.lr,
        betas=(config.train.beta1, config.train.beta2),
        fused=(device == "cuda"),
    )

    use_scaler = (device == "cuda" and amp_dtype == torch.float16)
    scaler = torch.amp.GradScaler("cuda") if use_scaler else None

    log_path = out_dir / "log.csv"
    best_val = float("inf")
    t_start = time.time()
    step = 0

    with open(log_path, "w", newline="") as log_f:
        writer = csv.writer(log_f)
        writer.writerow(["step", "split", "loss", "lr", "grad_norm", "wallclock_s"])

        model.train()
        while step < config.train.max_steps:
            lr = get_lr(step, config.train)
            for g in optim.param_groups:
                g["lr"] = lr

            optim.zero_grad(set_to_none=True)
            accum_loss = 0.0
            for _ in range(config.train.grad_accum_steps):
                x, y = get_batch(
                    train_ds,
                    config.train.micro_batch_size,
                    config.model.context_len,
                    device,
                    rng=rng_train,
                )
                with _autocast(device, amp_dtype):
                    _, loss = model(x, y)
                    loss = loss / config.train.grad_accum_steps
                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                accum_loss += loss.item()

            if scaler is not None:
                scaler.unscale_(optim)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.train.grad_clip
            )
            if scaler is not None:
                scaler.step(optim)
                scaler.update()
            else:
                optim.step()

            if step % config.train.log_every == 0:
                wall = time.time() - t_start
                writer.writerow([step, "train", f"{accum_loss:.6f}", f"{lr:.6e}",
                                 f"{float(grad_norm):.6f}", f"{wall:.2f}"])
                log_f.flush()
                print(f"step {step:5d} | train {accum_loss:.4f} | lr {lr:.2e} "
                      f"| gnorm {float(grad_norm):.3f} | {wall:.1f}s")

            if step > 0 and step % config.train.eval_every == 0:
                val_loss = evaluate(
                    model, val_ds,
                    n_iters=config.train.eval_iters,
                    batch_size=config.train.micro_batch_size,
                    context_len=config.model.context_len,
                    device=device,
                    amp_dtype=amp_dtype,
                    rng=rng_val,
                )
                wall = time.time() - t_start
                writer.writerow([step, "val", f"{val_loss:.6f}", f"{lr:.6e}",
                                 "", f"{wall:.2f}"])
                log_f.flush()
                print(f"step {step:5d} | val   {val_loss:.4f}")
                if val_loss < best_val:
                    best_val = val_loss
                    _save_ckpt(out_dir / "best.pt", model, optim, config, step, val_loss)

            if step > 0 and step % config.train.ckpt_every == 0:
                _save_ckpt(out_dir / f"step_{step:06d}.pt", model, optim, config, step, None)

            step += 1

    _save_ckpt(out_dir / "final.pt", model, optim, config, step, None)
    return {"best_val": best_val, "final_step": step}


def _main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None, help="Path to a saved Config JSON")
    p.add_argument("--out", default=None, help="Override train.out_dir")
    p.add_argument("--data", default=None, help="Override train.data_dir")
    args = p.parse_args()

    config = Config.from_json(args.config) if args.config else Config()
    if args.out:
        config.train.out_dir = args.out
    if args.data:
        config.train.data_dir = args.data
    return train(config)


if __name__ == "__main__":
    _main()

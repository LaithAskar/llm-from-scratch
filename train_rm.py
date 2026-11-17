"""
Reward-model training (Part 7).

Trains a RewardModel on (prompt, chosen, rejected) preference pairs with the
Bradley-Terry loss. The RM trunk is initialised from an SFT checkpoint; only
the scalar head plus a light fine-tune of the trunk are learned.

Metric: preference accuracy = fraction of held-out pairs where the RM scores
the chosen completion above the rejected one. 0.5 is chance.

Run:
    python train_rm.py \
        --base runs/sft/modern/best.pt \
        --prefs data/prefs/pairs.jsonl \
        --out runs/rm/modern \
        --max-steps 300
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

from preference_data import PreferenceDataset, collate_preference_batch
from reward_model import RewardModel, bradley_terry_loss, preference_accuracy

PAD_ID = 0  # value is irrelevant: RM reads the last *real* token; causal
            # attention means later (pad) positions never influence it.


def _autocast(device: str, dtype: torch.dtype):
    if dtype == torch.float32:
        return nullcontext()
    return torch.amp.autocast(
        device_type="cuda" if device.startswith("cuda") else "cpu", dtype=dtype
    )


def _lr(step: int, max_steps: int, warmup: int, peak: float, floor: float) -> float:
    if step < warmup:
        return peak * (step + 1) / max(1, warmup)
    if step >= max_steps:
        return floor
    prog = (step - warmup) / max(1, max_steps - warmup)
    return floor + 0.5 * (1 + math.cos(math.pi * prog)) * (peak - floor)


@torch.no_grad()
def evaluate(rm, dataset, val_idx, batch_size, device, amp_dtype) -> tuple[float, float]:
    """Return (mean BT loss, preference accuracy) over the val split."""
    was_training = rm.training
    rm.eval()
    losses, accs, n = 0.0, 0.0, 0
    for start in range(0, len(val_idx), batch_size):
        idx = val_idx[start : start + batch_size]
        batch = collate_preference_batch(dataset, idx, PAD_ID, device)
        with _autocast(device, amp_dtype):
            r_chosen = rm(batch["chosen_ids"], batch["chosen_len"])
            r_rejected = rm(batch["rejected_ids"], batch["rejected_len"])
            loss = bradley_terry_loss(r_chosen, r_rejected)
        losses += loss.item() * len(idx)
        accs += preference_accuracy(r_chosen, r_rejected) * len(idx)
        n += len(idx)
    if was_training:
        rm.train()
    return losses / n, accs / n


def train_rm(
    base_ckpt: str,
    prefs_path: str,
    out_dir: str,
    max_steps: int,
    lr: float,
    batch_size: int,
    seed: int,
    val_frac: float = 0.1,
) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp_dtype = torch.bfloat16 if device == "cuda" else torch.float32
    torch.manual_seed(seed)
    np.random.seed(seed)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rm = RewardModel.from_sft_checkpoint(base_ckpt, device)
    print(f"device: {device}")
    print(f"RM trunk initialised from {base_ckpt}")
    print(f"reward-model params: {sum(p.numel() for p in rm.parameters()):,}")

    dataset = PreferenceDataset(prefs_path, context_len=rm.config.context_len)
    n = len(dataset)
    perm = np.random.default_rng(seed).permutation(n)
    n_val = max(1, int(n * val_frac))
    val_idx = perm[:n_val].tolist()
    train_idx = perm[n_val:].tolist()
    print(f"preference pairs: {n}  (train {len(train_idx)} / val {len(val_idx)})")

    optim = torch.optim.AdamW(rm.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.0)
    rng = np.random.default_rng(seed + 1)

    log_path = out / "log.csv"
    best_acc = 0.0
    t0 = time.time()
    warmup = min(20, max_steps // 10)

    with open(log_path, "w", newline="") as log_f:
        w = csv.writer(log_f)
        w.writerow(["step", "split", "loss", "acc", "lr", "wallclock_s"])

        rm.train()
        for step in range(max_steps):
            cur_lr = _lr(step, max_steps, warmup, lr, lr * 0.1)
            for g in optim.param_groups:
                g["lr"] = cur_lr

            idx = rng.choice(train_idx, size=min(batch_size, len(train_idx)),
                             replace=False).tolist()
            batch = collate_preference_batch(dataset, idx, PAD_ID, device)

            optim.zero_grad(set_to_none=True)
            with _autocast(device, amp_dtype):
                r_chosen = rm(batch["chosen_ids"], batch["chosen_len"])
                r_rejected = rm(batch["rejected_ids"], batch["rejected_len"])
                loss = bradley_terry_loss(r_chosen, r_rejected)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(rm.parameters(), 1.0)
            optim.step()

            if step % 20 == 0:
                acc = preference_accuracy(r_chosen, r_rejected)
                wall = time.time() - t0
                w.writerow([step, "train", f"{loss.item():.6f}", f"{acc:.4f}",
                            f"{cur_lr:.6e}", f"{wall:.2f}"])
                log_f.flush()
                print(f"step {step:4d} | train loss {loss.item():.4f} | acc {acc:.3f} "
                      f"| lr {cur_lr:.2e}")

            if step > 0 and step % 50 == 0:
                val_loss, val_acc = evaluate(rm, dataset, val_idx, batch_size, device, amp_dtype)
                wall = time.time() - t0
                w.writerow([step, "val", f"{val_loss:.6f}", f"{val_acc:.4f}", "", f"{wall:.2f}"])
                log_f.flush()
                print(f"step {step:4d} | val   loss {val_loss:.4f} | acc {val_acc:.3f}")
                if val_acc > best_acc:
                    best_acc = val_acc
                    torch.save(
                        {"reward_model": rm.state_dict(), "model_config": rm.config,
                         "step": step, "val_acc": val_acc, "base_ckpt": base_ckpt},
                        out / "best.pt",
                    )

    final_loss, final_acc = evaluate(rm, dataset, val_idx, batch_size, device, amp_dtype)
    torch.save(
        {"reward_model": rm.state_dict(), "model_config": rm.config,
         "step": max_steps, "val_acc": final_acc, "base_ckpt": base_ckpt},
        out / "final.pt",
    )
    result = {"best_val_acc": best_acc, "final_val_acc": final_acc, "final_val_loss": final_loss}
    (out / "result.json").write_text(json.dumps(result, indent=2))
    print(f"done | best val acc {best_acc:.3f} | final val acc {final_acc:.3f}")
    return result


def _main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True, help="SFT checkpoint for the RM trunk init")
    p.add_argument("--prefs", default="data/prefs/pairs.jsonl")
    p.add_argument("--out", default="runs/rm/modern")
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    train_rm(args.base, args.prefs, args.out, args.max_steps, args.lr,
             args.batch_size, args.seed)


if __name__ == "__main__":
    _main()

"""
Part 7: reward modeling for preference learning.

The reward model reuses the same TransformerBlock stack as the language model
but replaces the tied LM head with a scalar reward head. It scores the final
non-padding token in each sequence. Training uses the standard pairwise Bradley-
Terry / InstructGPT loss:

    -log sigmoid(r_chosen - r_rejected)

This module implements the mechanics only. A meaningful RM still requires real
preference data (human labels or approved RLAIF judge calls).
"""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ModelConfig
from layers import RMSNorm, TransformerBlock, causal_mask
from preferences import PreferenceDataset, collate_preferences
from train import make_param_groups


def _make_norm(config: ModelConfig) -> nn.Module:
    if config.norm_type == "rmsnorm":
        return RMSNorm(config.d_model, eps=config.norm_eps)
    return nn.LayerNorm(config.d_model, eps=config.norm_eps, bias=config.bias)


class RewardModel(nn.Module):
    """Transformer encoder-style scorer over a token sequence."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.context_len, config.d_model) if config.pos_encoding == "learned" else None
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)])
        self.final_norm = _make_norm(config)
        self.reward_head = nn.Linear(config.d_model, 1)
        self.register_buffer("_causal_mask", causal_mask(config.context_len), persistent=False)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        B, T = idx.shape
        if T > self.config.context_len:
            raise ValueError(f"sequence length {T} exceeds context_len {self.config.context_len}")
        if lengths is None:
            lengths = torch.full((B,), T, device=idx.device, dtype=torch.long)
        lengths = lengths.to(device=idx.device, dtype=torch.long).clamp(min=1, max=T)

        h = self.tok_emb(idx)
        if self.pos_emb is not None:
            pos = torch.arange(T, device=idx.device)
            h = h + self.pos_emb(pos)
        h = self.drop(h)
        mask = self._causal_mask[:T, :T]
        for block in self.blocks:
            h = block(h, mask=mask)
        h = self.final_norm(h)

        last_idx = (lengths - 1).view(B, 1, 1).expand(B, 1, h.size(-1))
        last_h = h.gather(1, last_idx).squeeze(1)
        return self.reward_head(last_h).squeeze(-1)


def pairwise_reward_loss(chosen_rewards: torch.Tensor, rejected_rewards: torch.Tensor, margin: float = 0.0) -> torch.Tensor:
    """Bradley-Terry pairwise loss; lower when chosen scores exceed rejected."""
    return -F.logsigmoid(chosen_rewards - rejected_rewards - margin).mean()


@dataclass
class RewardTrainConfig:
    data_path: str
    out_dir: str = "runs/reward_model/default"
    lr: float = 1e-4
    weight_decay: float = 0.0
    batch_size: int = 8
    max_steps: int = 500
    log_every: int = 20
    eval_every: int = 100
    seed: int = 0
    pad_id: int = 0


def reward_train_step(model: RewardModel, batch: dict[str, torch.Tensor], optim: torch.optim.Optimizer) -> float:
    model.train()
    optim.zero_grad(set_to_none=True)
    chosen = model(batch["chosen"], batch["chosen_lengths"])
    rejected = model(batch["rejected"], batch["rejected_lengths"])
    loss = pairwise_reward_loss(chosen, rejected)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optim.step()
    return float(loss.item())


def train_reward_model(model_cfg: ModelConfig, train_cfg: RewardTrainConfig) -> dict:
    """Tiny reference trainer for JSONL preference data."""
    out = Path(train_cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(__import__("json").dumps({"model": asdict(model_cfg), "reward": asdict(train_cfg)}, indent=2))

    torch.manual_seed(train_cfg.seed)
    dataset = PreferenceDataset(train_cfg.data_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = RewardModel(model_cfg).to(device)
    optim = torch.optim.AdamW(make_param_groups(model, train_cfg.weight_decay), lr=train_cfg.lr)

    log_path = out / "log.csv"
    t0 = time.time()
    losses: list[float] = []
    with open(log_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "loss", "wallclock_s"])
        for step in range(train_cfg.max_steps):
            # deterministic cycling is enough for this small educational trainer.
            rows = [dataset[(step * train_cfg.batch_size + i) % len(dataset)] for i in range(train_cfg.batch_size)]
            batch = collate_preferences(rows, pad_id=train_cfg.pad_id, max_len=model_cfg.context_len)
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = reward_train_step(model, batch, optim)
            losses.append(loss)
            if step % train_cfg.log_every == 0:
                w.writerow([step, f"{loss:.6f}", f"{time.time() - t0:.2f}"])
                f.flush()
    torch.save({"model": model.state_dict(), "config": model_cfg, "reward_config": asdict(train_cfg)}, out / "final.pt")
    return {"final_loss": losses[-1], "steps": train_cfg.max_steps}


def _main():
    p = argparse.ArgumentParser(description="Train a tiny reward model from preference JSONL")
    p.add_argument("--data", required=True, help="JSONL preferences")
    p.add_argument("--out", default="runs/reward_model/default")
    p.add_argument("--max-steps", type=int, default=500)
    args = p.parse_args()
    cfg = ModelConfig()
    rcfg = RewardTrainConfig(data_path=args.data, out_dir=args.out, max_steps=args.max_steps)
    train_reward_model(cfg, rcfg)


if __name__ == "__main__":
    _main()

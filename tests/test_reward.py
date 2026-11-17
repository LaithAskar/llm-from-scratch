"""
Tests for the reward model (reward_model.py) and the preference pipeline
(preference_data.py) — Part 7.

Run: python -m pytest tests/test_reward.py -v
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

from config import ModelConfig
from reward_model import RewardModel, bradley_terry_loss, preference_accuracy
from preference_data import (
    HeuristicOracle,
    PreferenceDataset,
    collate_preference_batch,
    quality_score,
)


def _tiny_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=50257, n_layer=2, n_head=2, d_model=64,
        context_len=64, dropout=0.0, bias=False,
    )


# --------------------------------------------------------------------------
# RewardModel
# --------------------------------------------------------------------------

def test_reward_model_output_shape():
    rm = RewardModel(_tiny_config())
    idx = torch.randint(0, 50257, (5, 30))
    lengths = torch.full((5,), 30, dtype=torch.long)
    r = rm(idx, lengths)
    assert r.shape == (5,)
    assert r.dtype == torch.float32


def test_reward_model_rejects_bad_lengths():
    rm = RewardModel(_tiny_config())
    idx = torch.randint(0, 50257, (2, 20))
    with pytest.raises(ValueError):
        rm(idx, torch.tensor([0, 20]))          # length 0 is illegal
    with pytest.raises(ValueError):
        rm(idx, torch.tensor([21, 20]))         # length > T is illegal


def test_reward_ignores_padding_after_last_real_token():
    """The score must depend only on tokens up to `lengths-1`. Causal
    attention guarantees padding after that point cannot change it."""
    torch.manual_seed(0)
    rm = RewardModel(_tiny_config())
    rm.eval()

    real = torch.randint(0, 50257, (1, 12))
    lengths = torch.tensor([12])

    # Same 12 real tokens, padded to width 40 with two different pad values.
    pad_a = torch.cat([real, torch.zeros(1, 28, dtype=torch.long)], dim=1)
    pad_b = torch.cat([real, torch.full((1, 28), 999, dtype=torch.long)], dim=1)

    with torch.no_grad():
        ra = rm(pad_a, lengths)
        rb = rm(pad_b, lengths)
    assert torch.allclose(ra, rb, atol=1e-5), (
        "reward changed with padding value — score is leaking from pad tokens"
    )


# --------------------------------------------------------------------------
# Bradley-Terry loss
# --------------------------------------------------------------------------

def test_bt_loss_at_equal_rewards_is_log2():
    """r_chosen == r_rejected -> sigmoid(0)=0.5 -> loss = -log(0.5) = log 2."""
    r = torch.zeros(8)
    loss = bradley_terry_loss(r, r.clone())
    assert abs(loss.item() - math.log(2)) < 1e-5


def test_bt_loss_decreases_as_margin_grows():
    rejected = torch.zeros(4)
    small = bradley_terry_loss(torch.full((4,), 0.5), rejected)
    large = bradley_terry_loss(torch.full((4,), 5.0), rejected)
    assert large < small, "loss should fall as the chosen-vs-rejected margin grows"
    assert large.item() > 0, "loss is a negative log; always positive"


def test_bt_loss_gradient_pushes_rewards_apart():
    r_chosen = torch.tensor([0.0], requires_grad=True)
    r_rejected = torch.tensor([0.0], requires_grad=True)
    bradley_terry_loss(r_chosen, r_rejected).backward()
    # Minimising loss should raise r_chosen (neg grad) and lower r_rejected (pos grad).
    assert r_chosen.grad.item() < 0
    assert r_rejected.grad.item() > 0


def test_preference_accuracy():
    chosen = torch.tensor([1.0, 2.0, 0.0, 5.0])
    rejected = torch.tensor([0.0, 1.5, 0.5, 1.0])  # chosen wins on 3 of 4 (loses pair 3)
    assert preference_accuracy(chosen, rejected) == pytest.approx(0.75)


# --------------------------------------------------------------------------
# Heuristic quality score + oracle
# --------------------------------------------------------------------------

def test_quality_score_penalises_repetition():
    varied = [10, 11, 12, 13, 14, 15, 16, 17]
    repeated = [10, 11, 10, 11, 10, 11, 10, 11]
    assert quality_score(varied) > quality_score(repeated)


def test_quality_score_penalises_stray_eot():
    eot = 50256
    clean = [10, 11, 12, 13, 14, 15]
    with_stray = [10, 11, eot, 12, 13, 14]   # eot before the final position
    assert quality_score(clean) > quality_score(with_stray)


def test_heuristic_oracle_prefers_higher_quality():
    oracle = HeuristicOracle()
    prompt = [1, 2, 3]
    varied = [10, 11, 12, 13, 14, 15, 16, 17]
    repeated = [10, 11, 10, 11, 10, 11, 10, 11]
    assert oracle.prefer(prompt, varied, repeated) == 0      # A (varied) wins
    assert oracle.prefer(prompt, repeated, varied) == 1      # B (varied) wins


def test_heuristic_oracle_skips_ties():
    oracle = HeuristicOracle()
    same = [10, 11, 12, 13]
    assert oracle.prefer([1], same, list(same)) == -1


# --------------------------------------------------------------------------
# Preference dataset + collation
# --------------------------------------------------------------------------

def _write_prefs(tmp_path: Path) -> Path:
    rows = [
        {"prompt": [1, 2], "chosen": [3, 4, 5], "rejected": [6, 7]},
        {"prompt": [1, 2], "chosen": [8, 9, 10, 11], "rejected": [12]},
    ]
    p = tmp_path / "pairs.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return p


def test_preference_dataset_loads_and_concatenates(tmp_path: Path):
    ds = PreferenceDataset(_write_prefs(tmp_path), context_len=64)
    assert len(ds) == 2
    # _seq concatenates prompt + completion.
    assert ds._seq(ds.examples[0], "chosen") == [1, 2, 3, 4, 5]
    assert ds._seq(ds.examples[0], "rejected") == [1, 2, 6, 7]


def test_preference_dataset_truncates_to_context_len(tmp_path: Path):
    ds = PreferenceDataset(_write_prefs(tmp_path), context_len=3)
    assert ds._seq(ds.examples[0], "chosen") == [1, 2, 3]   # 5 tokens -> 3


def test_collate_pads_and_records_lengths(tmp_path: Path):
    ds = PreferenceDataset(_write_prefs(tmp_path), context_len=64)
    batch = collate_preference_batch(ds, [0, 1], pad_id=0, device="cpu")
    # chosen seqs: [1,2,3,4,5] (len 5) and [1,2,8,9,10,11] (len 6) -> pad to 6.
    assert batch["chosen_ids"].shape == (2, 6)
    assert batch["chosen_len"].tolist() == [5, 6]
    # padded row 0 has a trailing pad token at position 5.
    assert batch["chosen_ids"][0, 5].item() == 0
    assert batch["chosen_ids"][0, 4].item() == 5   # last real token preserved


def test_rm_one_step_finite_loss_and_grad(tmp_path: Path):
    """End-to-end: collate a preference batch, score both sides, BT loss,
    backward — loss finite and gradient flows to the reward head."""
    ds = PreferenceDataset(_write_prefs(tmp_path), context_len=64)
    rm = RewardModel(_tiny_config())
    rm.train()
    batch = collate_preference_batch(ds, [0, 1], pad_id=0, device="cpu")

    r_chosen = rm(batch["chosen_ids"], batch["chosen_len"])
    r_rejected = rm(batch["rejected_ids"], batch["rejected_len"])
    loss = bradley_terry_loss(r_chosen, r_rejected)
    assert torch.isfinite(loss).item()
    loss.backward()
    assert rm.reward_head.weight.grad is not None
    assert torch.any(rm.reward_head.weight.grad != 0).item()

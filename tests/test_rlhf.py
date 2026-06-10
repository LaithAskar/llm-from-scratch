"""
Tests for Parts 7-9: reward modeling/RLAIF-style preference learning, PPO,
and GRPO helpers.

These are intentionally tiny CPU tests. They verify the learning mechanics
without claiming that a useful reward model or aligned policy has been trained.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from config import ModelConfig
from model import TransformerLM


def test_pairwise_reward_loss_prefers_chosen_over_rejected():
    from reward_model import pairwise_reward_loss

    chosen = torch.tensor([2.0, 0.5, -0.2])
    rejected = torch.tensor([0.0, 0.1, -1.0])
    good_loss = pairwise_reward_loss(chosen, rejected)
    bad_loss = pairwise_reward_loss(rejected, chosen)

    assert good_loss.item() < bad_loss.item()
    assert torch.isfinite(good_loss).item()


def test_reward_model_scores_one_scalar_per_sequence_and_trains_one_step():
    from reward_model import RewardModel, pairwise_reward_loss

    cfg = ModelConfig(
        vocab_size=128,
        n_layer=1,
        n_head=2,
        d_model=32,
        context_len=16,
        dropout=0.0,
        bias=False,
    )
    model = RewardModel(cfg)
    chosen = torch.randint(0, cfg.vocab_size, (4, cfg.context_len))
    rejected = torch.randint(0, cfg.vocab_size, (4, cfg.context_len))
    lengths = torch.full((4,), cfg.context_len, dtype=torch.long)

    chosen_scores = model(chosen, lengths)
    rejected_scores = model(rejected, lengths)
    assert chosen_scores.shape == (4,)
    assert rejected_scores.shape == (4,)

    loss = pairwise_reward_loss(chosen_scores, rejected_scores)
    loss.backward()
    assert any(p.grad is not None and torch.any(p.grad != 0) for p in model.parameters())


def test_load_preference_jsonl_and_collate(tmp_path: Path):
    from preferences import PreferenceDataset, collate_preferences

    path = tmp_path / "prefs.jsonl"
    rows = [
        {"prompt_ids": [1, 2], "chosen_ids": [3, 4, 5], "rejected_ids": [6]},
        {"prompt_ids": [7], "chosen_ids": [8], "rejected_ids": [9, 10]},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows))

    ds = PreferenceDataset(path)
    batch = collate_preferences([ds[0], ds[1]], pad_id=0, max_len=6)

    assert batch["chosen"].shape == (2, 6)
    assert batch["rejected"].shape == (2, 6)
    assert batch["chosen_lengths"].tolist() == [5, 2]
    assert batch["rejected_lengths"].tolist() == [3, 3]
    assert batch["chosen"][0, :5].tolist() == [1, 2, 3, 4, 5]


def test_sequence_logprobs_match_manual_cross_entropy():
    from ppo import sequence_logprobs

    cfg = ModelConfig(
        vocab_size=64,
        n_layer=1,
        n_head=2,
        d_model=32,
        context_len=8,
        dropout=0.0,
        bias=False,
    )
    policy = TransformerLM(cfg)
    tokens = torch.randint(0, cfg.vocab_size, (3, cfg.context_len))
    prompt_len = 3

    got = sequence_logprobs(policy, tokens, prompt_len=prompt_len)
    logits, _ = policy(tokens[:, :-1])
    logp = F.log_softmax(logits, dim=-1)
    target = tokens[:, 1:]
    token_logp = logp.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    # prompt_len=3 means response targets start at target column prompt_len-1.
    expected = token_logp[:, prompt_len - 1 :].sum(dim=-1)

    assert torch.allclose(got, expected, atol=1e-5)


def test_ppo_clipped_policy_loss_uses_lower_surrogate_when_ratio_too_large():
    from ppo import ppo_clipped_policy_loss

    old = torch.zeros(3)
    new = torch.log(torch.tensor([1.0, 1.5, 0.5]))
    adv = torch.tensor([1.0, 1.0, -1.0])

    loss, stats = ppo_clipped_policy_loss(new, old, adv, clip_eps=0.2)
    # Sample 2 has positive advantage and ratio 1.5, so clipped objective uses 1.2.
    expected_obj = torch.tensor([1.0, 1.2, -0.8]).mean()

    assert torch.allclose(loss, -expected_obj)
    assert stats["clip_frac"] > 0


def test_grpo_group_advantages_normalize_per_prompt_group():
    from grpo import group_advantages

    rewards = torch.tensor([1.0, 2.0, 3.0, 10.0, 10.0, 10.0])
    adv = group_advantages(rewards, group_size=3)

    assert adv.shape == rewards.shape
    assert torch.allclose(adv[:3].mean(), torch.tensor(0.0), atol=1e-6)
    assert torch.allclose(adv[:3].std(unbiased=False), torch.tensor(1.0), atol=1e-5)
    # Equal-reward group should not NaN; it should become all zeros.
    assert torch.allclose(adv[3:], torch.zeros(3))


def test_grpo_policy_loss_matches_ppo_with_group_normalized_advantages():
    from grpo import grpo_policy_loss, group_advantages
    from ppo import ppo_clipped_policy_loss

    old = torch.zeros(6)
    new = torch.log(torch.tensor([1.0, 1.1, 0.9, 0.8, 1.3, 1.0]))
    rewards = torch.tensor([1.0, 2.0, 3.0, -1.0, 0.0, 1.0])

    loss, stats = grpo_policy_loss(new, old, rewards, group_size=3, clip_eps=0.2)
    expected, _ = ppo_clipped_policy_loss(new, old, group_advantages(rewards, 3), clip_eps=0.2)

    assert torch.allclose(loss, expected)
    assert "mean_group_reward" in stats

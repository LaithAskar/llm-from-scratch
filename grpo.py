"""
Part 9: GRPO helpers.

GRPO (Group Relative Policy Optimization) removes the learned value critic used
by PPO. For each prompt, sample a group of completions, score them with the
reward model, normalize rewards within that group, and use those normalized
relative scores as advantages in a PPO-like clipped policy objective.
"""

from __future__ import annotations

import torch

from ppo import ppo_clipped_policy_loss


def group_advantages(rewards: torch.Tensor, group_size: int, eps: float = 1e-8) -> torch.Tensor:
    """Normalize rewards within each prompt group.

    Input is flat: [prompt0_sample0, prompt0_sample1, ..., prompt1_sample0, ...]
    Returns same shape. Equal-reward groups become zeros rather than NaNs.
    """
    if rewards.ndim != 1:
        raise ValueError("rewards must be flat (num_prompts * group_size,)")
    if group_size < 2:
        raise ValueError("group_size must be >= 2")
    if rewards.numel() % group_size != 0:
        raise ValueError("number of rewards must be divisible by group_size")
    grouped = rewards.view(-1, group_size)
    mean = grouped.mean(dim=1, keepdim=True)
    std = grouped.std(dim=1, unbiased=False, keepdim=True)
    adv = (grouped - mean) / (std + eps)
    adv = torch.where(std > eps, adv, torch.zeros_like(adv))
    return adv.reshape_as(rewards)


def grpo_policy_loss(
    new_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    rewards: torch.Tensor,
    group_size: int,
    clip_eps: float = 0.2,
) -> tuple[torch.Tensor, dict[str, float]]:
    """GRPO clipped policy objective using group-normalized rewards."""
    advantages = group_advantages(rewards, group_size=group_size)
    loss, stats = ppo_clipped_policy_loss(new_logprobs, old_logprobs, advantages, clip_eps)
    stats.update({
        "mean_group_reward": float(rewards.view(-1, group_size).mean(dim=1).mean().detach()),
        "adv_std": float(advantages.std(unbiased=False).detach()),
    })
    return loss, stats


def grpo_kl_penalty(new_logprobs: torch.Tensor, ref_logprobs: torch.Tensor, beta: float = 0.02) -> torch.Tensor:
    """Simple sequence-level KL penalty used to keep the policy near reference."""
    return beta * (new_logprobs - ref_logprobs).mean()

"""
Part 8: PPO helpers for RL fine-tuning a language model.

This is the small, testable core of PPO rather than a full production trainer:
sequence log-prob extraction, clipped policy objective, value loss, KL penalty,
and GAE. A full run needs a trained reward model from Part 7 and sampled policy
rollouts.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from model import TransformerLM


def sequence_logprobs(policy: TransformerLM, sequences: torch.Tensor, prompt_len: int) -> torch.Tensor:
    """Sum log p(response tokens | prefix) for each sequence.

    `sequences` contains prompt+response token IDs. If prompt_len is the number
    of prompt tokens, response targets begin at target column prompt_len-1
    because logits[:, t] predicts sequences[:, t+1].
    """
    if sequences.ndim != 2:
        raise ValueError("sequences must be (batch, time)")
    if not (1 <= prompt_len < sequences.size(1)):
        raise ValueError("prompt_len must leave at least one response token")
    logits, _ = policy(sequences[:, :-1])
    logp = F.log_softmax(logits, dim=-1)
    targets = sequences[:, 1:]
    token_logp = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return token_logp[:, prompt_len - 1 :].sum(dim=-1)


def approx_kl(new_logprobs: torch.Tensor, ref_logprobs: torch.Tensor) -> torch.Tensor:
    """Monte-Carlo KL estimate over sampled sequences."""
    return (new_logprobs - ref_logprobs).mean()


def ppo_clipped_policy_loss(
    new_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    clip_eps: float = 0.2,
) -> tuple[torch.Tensor, dict[str, float]]:
    """PPO clipped surrogate loss for already-summed response log-probs."""
    if new_logprobs.shape != old_logprobs.shape or new_logprobs.shape != advantages.shape:
        raise ValueError("new_logprobs, old_logprobs, and advantages must have same shape")
    ratio = torch.exp(new_logprobs - old_logprobs)
    clipped = ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps)
    objective = torch.minimum(ratio * advantages, clipped * advantages)
    loss = -objective.mean()
    stats = {
        "ratio_mean": float(ratio.mean().detach()),
        "clip_frac": float(((ratio - clipped).abs() > 0).float().mean().detach()),
        "adv_mean": float(advantages.mean().detach()),
    }
    return loss, stats


def value_loss(values: torch.Tensor, returns: torch.Tensor, clip_range: float | None = None, old_values: torch.Tensor | None = None) -> torch.Tensor:
    """MSE value loss, optionally with PPO-style value clipping."""
    if clip_range is None:
        return 0.5 * (values - returns).pow(2).mean()
    if old_values is None:
        raise ValueError("old_values required when clip_range is set")
    clipped = old_values + (values - old_values).clamp(-clip_range, clip_range)
    loss_unclipped = (values - returns).pow(2)
    loss_clipped = (clipped - returns).pow(2)
    return 0.5 * torch.maximum(loss_unclipped, loss_clipped).mean()


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generalized Advantage Estimation for one batch of trajectories.

    Shapes: rewards=(B,T), values=(B,T+1), dones=(B,T) where done=1 cuts bootstrapping.
    Returns (advantages, returns), each (B,T).
    """
    if values.size(1) != rewards.size(1) + 1:
        raise ValueError("values must have one bootstrap column: T+1")
    adv = torch.zeros_like(rewards)
    lastgaelam = torch.zeros(rewards.size(0), device=rewards.device, dtype=rewards.dtype)
    for t in reversed(range(rewards.size(1))):
        nonterminal = 1.0 - dones[:, t]
        delta = rewards[:, t] + gamma * values[:, t + 1] * nonterminal - values[:, t]
        lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
        adv[:, t] = lastgaelam
    returns = adv + values[:, :-1]
    return adv, returns


def ppo_total_loss(
    new_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    values: torch.Tensor,
    returns: torch.Tensor,
    kl: torch.Tensor | None = None,
    kl_coef: float = 0.02,
    value_coef: float = 0.5,
    clip_eps: float = 0.2,
) -> tuple[torch.Tensor, dict[str, float]]:
    policy_loss, stats = ppo_clipped_policy_loss(new_logprobs, old_logprobs, advantages, clip_eps)
    v_loss = value_loss(values, returns)
    total = policy_loss + value_coef * v_loss
    if kl is not None:
        total = total + kl_coef * kl
        stats["kl"] = float(kl.detach())
    stats["policy_loss"] = float(policy_loss.detach())
    stats["value_loss"] = float(v_loss.detach())
    return total, stats

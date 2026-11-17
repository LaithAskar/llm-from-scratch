"""
Reward model (Part 7).

A RewardModel wraps a TransformerLM trunk and adds a scalar reward head.
It maps a token sequence to a single number r(sequence). It is trained on
*pairwise preference data* with the Bradley-Terry loss:

    loss = -log sigmoid(r(chosen) - r(rejected))

i.e. the model is pushed to assign a higher scalar to the preferred
completion. Only the *difference* of rewards is supervised, so the
absolute scale of r is unidentifiable — that's expected and fine; PPO
(Part 8) only ever uses reward differences / advantages.

Design notes
------------
- The RM is initialised from an SFT checkpoint (standard RLHF practice):
  the trunk already has a useful representation of TinyStories text, and
  only the scalar head is learned from scratch.
- The reward is read from the hidden state at the *last real token* of
  each sequence (the token just before padding). Because attention is
  causal, padding tokens after that position cannot influence the score,
  so the padding value is irrelevant.
- The trunk's `lm_head` is inherited from TransformerLM and simply unused
  here. It is weight-tied to the embedding so it costs no extra params.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ModelConfig
from model import TransformerLM


class RewardModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.transformer = TransformerLM(config)
        # Scalar head: hidden state -> reward. Small init so initial rewards
        # sit near 0 and the BT loss starts in its well-conditioned region.
        self.reward_head = nn.Linear(config.d_model, 1, bias=False)
        nn.init.normal_(self.reward_head.weight, mean=0.0, std=0.02)

    @classmethod
    def from_sft_checkpoint(cls, ckpt_path: str, device: str) -> "RewardModel":
        """Build an RM whose trunk is initialised from an SFT (or pretrain)
        checkpoint. The reward head is fresh."""
        blob = torch.load(ckpt_path, map_location=device, weights_only=False)
        base_cfg = blob["config"]
        model_cfg = base_cfg.model
        if not isinstance(model_cfg, ModelConfig):
            model_cfg = ModelConfig(**model_cfg)
        rm = cls(model_cfg)
        rm.transformer.load_state_dict(blob["model"])
        return rm.to(device)

    def forward(self, idx: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        idx     : (B, T) int64 token ids, right-padded.
        lengths : (B,) int64, number of real (non-pad) tokens per row, >= 1.
        returns : (B,) float reward, one scalar per sequence.
        """
        if torch.any(lengths < 1):
            raise ValueError("every sequence must have at least 1 real token")
        if torch.any(lengths > idx.size(1)):
            raise ValueError("lengths cannot exceed the padded sequence length")

        h = self.transformer.hidden_states(idx)        # (B, T, C)
        # Gather the hidden state at the last real token of each row.
        last_idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, h.size(-1))  # (B,1,C)
        last_h = h.gather(dim=1, index=last_idx).squeeze(1)                 # (B, C)
        return self.reward_head(last_h).squeeze(-1)                         # (B,)


def bradley_terry_loss(
    r_chosen: torch.Tensor,    # (B,) reward of the preferred completion
    r_rejected: torch.Tensor,  # (B,) reward of the dispreferred completion
) -> torch.Tensor:
    """
    Pairwise preference loss: -log sigmoid(r_chosen - r_rejected), averaged.

    F.logsigmoid is numerically stable across the full range of the
    reward difference, so no manual clamping is needed.
    """
    return -F.logsigmoid(r_chosen - r_rejected).mean()


@torch.no_grad()
def preference_accuracy(r_chosen: torch.Tensor, r_rejected: torch.Tensor) -> float:
    """Fraction of pairs where the RM scores the chosen completion higher.
    0.5 is chance; 1.0 is perfect agreement with the oracle's labels."""
    return (r_chosen > r_rejected).float().mean().item()

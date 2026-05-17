"""
TransformerLM: token embedding + (optional learned) position embedding +
n TransformerBlocks + final norm + tied LM head.

Architecture decisions (norm type, FFN activation, RoPE vs learned-pos,
pre-norm vs post-norm) live inside TransformerBlock in layers.py.
This file is plumbing only — it stacks blocks and wires up the I/O.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ModelConfig
from layers import RMSNorm, TransformerBlock, causal_mask


def _make_norm(config: ModelConfig) -> nn.Module:
    """Final norm — must match the norm type used inside blocks."""
    if config.norm_type == "rmsnorm":
        return RMSNorm(config.d_model, eps=config.norm_eps)
    return nn.LayerNorm(config.d_model, eps=config.norm_eps, bias=config.bias)


class TransformerLM(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)

        # Learned-absolute position embedding lives here. RoPE is applied
        # inside attention and doesn't appear at this level.
        if config.pos_encoding == "learned":
            self.pos_emb: Optional[nn.Embedding] = nn.Embedding(config.context_len, config.d_model)
        else:
            self.pos_emb = None

        self.drop = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)])
        self.final_norm = _make_norm(config)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Causal mask: pre-compute once at max context length, slice per forward.
        mask = causal_mask(config.context_len)
        self.register_buffer("_causal_mask", mask, persistent=False)

        # Init weights, then tie LM head <-> token embedding.
        self.apply(self._init_weights)

        # GPT-2 trick: scale residual projection inits by 1/sqrt(2N).
        # If your Block uses different names than `out_proj` / `down_proj`,
        # update these suffixes (or remove the scaling — it's a stability nicety,
        # not load-bearing for correctness).
        for name, p in self.named_parameters():
            if name.endswith("out_proj.weight") or name.endswith("down_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        # Tie weights AFTER init so lm_head's Linear init doesn't clobber
        # the embedding's init (they end up pointing at the same tensor).
        self.lm_head.weight = self.tok_emb.weight

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self, non_embedding: bool = True) -> int:
        """Param count. Embeddings excluded by default (they dominate at this scale)."""
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.tok_emb.weight.numel()
            if self.pos_emb is not None:
                n -= self.pos_emb.weight.numel()
            # lm_head is tied to tok_emb, already excluded above.
        return n

    def forward(
        self,
        idx: torch.Tensor,                       # (B, T) int64
        targets: Optional[torch.Tensor] = None,  # (B, T) int64 or None
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = idx.shape
        if T > self.config.context_len:
            raise ValueError(
                f"sequence length {T} exceeds context_len {self.config.context_len}"
            )

        h = self.tok_emb(idx)                              # (B, T, C)
        if self.pos_emb is not None:
            pos = torch.arange(T, device=idx.device)        # (T,)
            h = h + self.pos_emb(pos)                       # broadcast -> (B, T, C)
        h = self.drop(h)

        mask = self._causal_mask[:T, :T]                    # (T, T) bool

        for block in self.blocks:
            h = block(h, mask=mask)

        h = self.final_norm(h)
        logits = self.lm_head(h)                            # (B, T, V)

        loss: Optional[torch.Tensor] = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100,
            )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,                  # (B, T) int64 seed
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """Naive sampling — no KV cache (that's a Part 3 add-on)."""
        was_training = self.training
        self.eval()
        try:
            for _ in range(max_new_tokens):
                idx_cond = (
                    idx
                    if idx.size(1) <= self.config.context_len
                    else idx[:, -self.config.context_len:]
                )
                logits, _ = self(idx_cond)
                logits = logits[:, -1, :] / max(temperature, 1e-8)
                if top_k is not None:
                    k = min(top_k, logits.size(-1))
                    v, _ = torch.topk(logits, k)
                    logits[logits < v[:, [-1]]] = -float("inf")
                probs = F.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)
                idx = torch.cat((idx, next_id), dim=1)
        finally:
            if was_training:
                self.train()
        return idx

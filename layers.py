"""
Transformer building blocks, implemented from scratch.

Part 1 starts here: Multi-Head Attention. Re-derive each line; do not copy from
the upstream reference repo.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    """
    Scaled dot-product multi-head self-attention.

    Reference: Vaswani et al., "Attention Is All You Need" (2017), section 3.2.

    Shapes
    ------
    Input  x:      (batch, seq_len, embed_dim)
    Output:        (batch, seq_len, embed_dim)
    Optional mask: (seq_len, seq_len) or (batch, 1, seq_len, seq_len);
                   `True` positions are kept, `False` positions are masked out.
                   For a decoder-style causal mask, this is the lower-triangular
                   matrix.

    Args
    ----
    embed_dim : total embedding dimension (must be divisible by num_heads).
    num_heads : number of attention heads. head_dim = embed_dim // num_heads.
    dropout   : dropout prob applied to attention weights post-softmax.

    Interview anchors to be able to answer after writing this:
    - why divide by sqrt(head_dim)?
    - why softmax along the last dim of the (B, H, T, T) score tensor?
    - what shape does the mask have and why fill with -inf, not 0?
    - what's the memory cost of the (T, T) score matrix and why does that
      motivate Flash Attention / KV cache?
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        # TODO: assert embed_dim % num_heads == 0 with a helpful error message.

        # TODO: store embed_dim, num_heads, head_dim as instance attributes.

        # TODO: create three nn.Linear projections for Q, K, V.
        #       Each: embed_dim -> embed_dim. Bias is optional (decide and
        #       justify your choice in a comment — Vaswani uses bias=True;
        #       many modern LLMs use bias=False).

        # TODO: create the output projection nn.Linear(embed_dim, embed_dim).

        # TODO: create nn.Dropout for attention weights.

        raise NotImplementedError("Implement __init__ for MultiHeadAttention.")

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Expected: x.shape == (B, T, C) where C == self.embed_dim.

        # Step 1: linear projections.
        # TODO: compute q, k, v from x via your three projections.
        #       Each result has shape (B, T, C).

        # Step 2: split into heads.
        # TODO: reshape each of q, k, v to (B, T, H, head_dim) and then
        #       transpose to (B, H, T, head_dim) so attention is per-head.

        # Step 3: scaled dot-product scores.
        # TODO: scores = q @ k.transpose(-2, -1) / sqrt(head_dim)
        #       shape: (B, H, T, T).

        # Step 4: apply mask (if provided).
        # TODO: scores = scores.masked_fill(mask == 0, float('-inf')) or
        #       equivalent. Be careful about mask broadcasting shape.

        # Step 5: softmax + dropout.
        # TODO: attn = softmax over the last dim. attn = self.dropout(attn).

        # Step 6: weighted sum of values.
        # TODO: out = attn @ v, shape (B, H, T, head_dim).

        # Step 7: merge heads.
        # TODO: transpose back to (B, T, H, head_dim), then reshape to
        #       (B, T, C). Use .contiguous() before reshape if needed.

        # Step 8: output projection.
        # TODO: out = self.out_proj(out). Return out.

        raise NotImplementedError("Implement forward for MultiHeadAttention.")


def causal_mask(seq_len: int, device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Lower-triangular boolean mask of shape (seq_len, seq_len).
    True == attend, False == mask out.

    For decoder-style autoregressive attention, this prevents position i from
    looking at positions j > i. Useful smoke-test ingredient for forward().
    """
    # TODO: return torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))
    raise NotImplementedError("Implement causal_mask.")


class RMSNorm(nn.Module):
    """
    Root-Mean-Square LayerNorm (Zhang & Sennrich, 2019).

    Drops LayerNorm's mean-subtraction and bias. Cheaper per-token (one fewer
    reduction) and empirically as good or better on transformer LMs (LLaMA,
    PaLM, Mistral all use it).

    y = x / sqrt(mean(x^2) + eps) * gain

    Args
    ----
    dim : last-dim size (typically d_model).
    eps : numerical stabilizer inside the sqrt.

    Interview anchors:
    - why drop the mean term — what does it cost / save?
    - why is the `gain` (weight) parameter d-dim and not scalar?
    - what dtype should you compute the reduction in if x is bf16? (hint:
      promote to fp32 to avoid catastrophic precision loss in the mean.)
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()

        # TODO: store eps. create self.weight as a learnable d-dim parameter
        # initialized to ones (nn.Parameter(torch.ones(dim))).

        raise NotImplementedError("Implement __init__ for RMSNorm.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (..., dim)

        # TODO: compute mean(x^2) along the last dim, keepdim=True.
        #       promote to fp32 first if x is fp16/bf16 to keep the mean stable,
        #       then cast back. (See LLaMA reference impl.)

        # TODO: rms = sqrt(mean_sq + eps), then x_norm = x / rms.

        # TODO: return x_norm * self.weight.

        raise NotImplementedError("Implement forward for RMSNorm.")


class TransformerBlock(nn.Module):
    """
    One transformer decoder block.

    Architectural choices (you decide and document each in a comment):
    - pre-norm vs post-norm? (modern: pre-norm. Why?)
    - which norm: LayerNorm vs RMSNorm — read from config.
    - which activation: GELU vs SwiGLU FFN — read from config.
    - position encoding: learned-absolute (handled at LM level), or RoPE
      (handled inside attention) — read from config.

    The constructor takes a ModelConfig so all switches live in one place.
    The model assembly code in model.py just stacks n_layer of these.

    Suggested forward:
        h = x + self.attn(self.norm1(x), mask)
        h = h + self.ffn(self.norm2(h))
        return h

    But you implement it — the suggested form is pre-norm; if you pick
    post-norm, justify it.

    Interview anchors:
    - why pre-norm beats post-norm at depth (training stability, gradient
      flow without warmup tricks). See Xiong et al., "On Layer Normalization
      in the Transformer Architecture" (2020).
    - why two norms per block, not one or three?
    - why is the FFN dim usually 4 * d_model (or (8/3) * d_model for SwiGLU)?
    """

    def __init__(self, config):  # config: ModelConfig — avoiding circular import in type hint
        super().__init__()

        # TODO: build self.norm1, self.attn, self.norm2, self.ffn.
        # TODO: switch on config.norm_type to pick LayerNorm or RMSNorm.
        # TODO: switch on config.activation to pick the FFN variant.
        # TODO: pass config.pos_encoding to MHA if you handle RoPE inside it.

        raise NotImplementedError("Implement __init__ for TransformerBlock.")

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # TODO: pre-norm + residual for attention.
        # TODO: pre-norm + residual for FFN.
        # TODO: return.

        raise NotImplementedError("Implement forward for TransformerBlock.")


if __name__ == "__main__":
    # Smoke test. Run with: python layers.py
    # This will fail with NotImplementedError until you implement the class.
    # Once implemented, it asserts shape correctness on a tiny random input.

    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch, seq_len, embed_dim, num_heads = 2, 16, 64, 8
    mha = MultiHeadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=0.0).to(device)
    x = torch.randn(batch, seq_len, embed_dim, device=device)

    # --- Test 1: no mask ---
    out = mha(x)
    assert out.shape == (batch, seq_len, embed_dim), (
        f"shape mismatch: expected {(batch, seq_len, embed_dim)}, got {tuple(out.shape)}"
    )

    # --- Test 2: causal mask ---
    mask = causal_mask(seq_len, device=device)
    out_masked = mha(x, mask=mask)
    assert out_masked.shape == (batch, seq_len, embed_dim), "causal-masked shape wrong"

    # --- Test 3: gradients flow ---
    loss = out_masked.sum()
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in mha.parameters()), (
        "no gradients reached MHA parameters"
    )

    print(f"MultiHeadAttention smoke test passed on {device}.")
    print(f"  input shape:  {tuple(x.shape)}")
    print(f"  output shape: {tuple(out.shape)}")
    print(f"  param count:  {sum(p.numel() for p in mha.parameters()):,}")

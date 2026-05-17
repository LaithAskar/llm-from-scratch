"""
Transformer building blocks, implemented from scratch.

Part 1 starts here: Multi-Head Attention. Re-derive each line; do not copy from
the upstream reference repo.
"""

from __future__ import annotations

import math
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


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
        bias: bool = False,
        rotary: Optional[Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]] = None,
    ):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
            )
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        # bias=False matches LLaMA / PaLM convention; the project's
        # config.bias plumbs through here from TransformerBlock.
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.attn_dropout = nn.Dropout(dropout)
        # If set, called as (q, k) -> (q_rot, k_rot) between head-split and
        # scores. None = no positional rotation (learned-absolute pos is
        # handled in model.py at the embedding level).
        self.rotary = rotary

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, C = x.shape
        H, D = self.num_heads, self.head_dim

        # Project and split into heads in one expression per tensor.
        # view(B, T, H, D) puts heads on a new axis; transpose(1, 2) moves
        # heads in front of time so attention is a batched per-head matmul.
        q = self.q_proj(x).view(B, T, H, D).transpose(1, 2)  # (B, H, T, D)
        k = self.k_proj(x).view(B, T, H, D).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, D).transpose(1, 2)

        if self.rotary is not None:
            q, k = self.rotary(q, k)

        # Scaled dot-product scores. (B, H, T, D) @ (B, H, D, T) -> (B, H, T, T).
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(D)

        if mask is not None:
            # Convention: mask True == attend, False == mask out (project
            # convention, set by causal_mask). masked_fill writes where the
            # predicate is True, so we invert.
            # Shape (T, T) bool broadcasts across (B, H) automatically.
            scores = scores.masked_fill(~mask, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)

        # (B, H, T, T) @ (B, H, T, D) -> (B, H, T, D).
        out = attn @ v

        # Merge heads back: transpose breaks contiguity, so .contiguous()
        # before view() to avoid a stride error.
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)


def causal_mask(seq_len: int, device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Lower-triangular boolean mask of shape (seq_len, seq_len).
    True == attend, False == mask out.

    For decoder-style autoregressive attention, this prevents position i from
    looking at positions j > i.
    """
    return torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))


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
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Promote to fp32 for the reduction. bf16/fp16 mean-of-squares loses
        # precision catastrophically across the d_model-sized sum; this is
        # the standard LLaMA-reference fix.
        orig_dtype = x.dtype
        x_f32 = x.to(torch.float32)
        rms = torch.rsqrt(x_f32.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x_f32 * rms).to(orig_dtype) * self.weight


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (Su et al., 2021).

    Instead of *adding* a position vector to the token embedding, *rotate*
    each Q and K vector by a position-dependent angle. The dot product of
    two rotated vectors depends only on their *relative* position offset
    (not their absolute positions), which is why RoPE extrapolates better
    to unseen sequence lengths than learned-absolute pos.

    Convention: half-split (LLaMA / Hugging Face style). Pair element `i`
    with element `i + head_dim/2`. The first half holds "real" components,
    the second half "imaginary." Each pair is rotated by `m * inv_freq[i]`
    where `m` is the position.

    Cached cos/sin tables are non-persistent buffers — they regenerate
    automatically if you change device or seq len above max_seq_len.

    Call signature: `rope(q, k) -> (q_rotated, k_rotated)`. V is NOT rotated
    (V carries content, position should only influence attention weights).
    """

    def __init__(self, head_dim: int, max_seq_len: int, base: float = 10000.0):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim ({head_dim}) must be even for RoPE")
        self.head_dim = head_dim
        # Inverse frequencies: shape (head_dim/2,). Computed in fp32 for
        # precision — the cached cos/sin are also fp32, downstream matmuls
        # cast as needed.
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        positions = torch.arange(max_seq_len, dtype=torch.float32)
        # Outer product: (max_seq_len, head_dim/2). Each (m, i) entry is
        # the angle m * inv_freq[i].
        freqs = torch.outer(positions, inv_freq)
        # Non-persistent: regenerate on load instead of bloating checkpoints.
        self.register_buffer("cos_cached", freqs.cos(), persistent=False)
        self.register_buffer("sin_cached", freqs.sin(), persistent=False)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # q, k expected shape: (B, H, T, D), with D == self.head_dim.
        T = q.size(-2)
        if T > self.cos_cached.size(0):
            raise ValueError(
                f"sequence length {T} exceeds RoPE max_seq_len {self.cos_cached.size(0)}"
            )
        cos = self.cos_cached[:T]  # (T, D/2)
        sin = self.sin_cached[:T]
        return _apply_rotary(q, cos, sin), _apply_rotary(k, cos, sin)


def _apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    Apply rotary embedding to x using cached cos/sin.

    Half-split convention:
        x = [x1 | x2]  where x1, x2 each have shape (..., D/2)
        x_rotated = [x1*cos - x2*sin | x1*sin + x2*cos]

    Shapes:
        x:   (..., T, D)
        cos: (T, D/2)   — broadcasts against the leading dims of x
        sin: (T, D/2)
        out: (..., T, D)
    """
    D = x.size(-1)
    half = D // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    # Cast cos/sin to x's dtype for AMP. The cache is fp32; without this,
    # PyTorch promotes the whole result to fp32 and you lose AMP's bf16 path.
    cos_x = cos.to(x.dtype)
    sin_x = sin.to(x.dtype)
    out1 = x1 * cos_x - x2 * sin_x
    out2 = x1 * sin_x + x2 * cos_x
    return torch.cat((out1, out2), dim=-1)


class GeluFFN(nn.Module):
    """
    Classic two-matrix FFN: y = down_proj(GELU(up_proj(x))).

    With d_ffn = 4 * d_model (Vaswani default), this contributes 8 * d_model²
    params per block — roughly 2x the attention param count, and the bulk of
    each layer's representational capacity.

    The second linear is named `down_proj` so that model.py's _init_weights
    can apply GPT-2 style residual scaling via name suffix match.
    """

    def __init__(self, d_model: int, d_ffn: int, bias: bool = False, dropout: float = 0.0):
        super().__init__()
        self.up_proj = nn.Linear(d_model, d_ffn, bias=bias)
        self.down_proj = nn.Linear(d_ffn, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down_proj(F.gelu(self.up_proj(x))))


class SwiGLUFFN(nn.Module):
    """
    Gated FFN with SiLU activation (Shazeer 2020):

        y = down_proj( silu(gate_proj(x)) * up_proj(x) )

    Three matrices instead of two. The element-wise product is the "gate":
    silu(gate_proj(x)) decides per-channel how much of up_proj(x) flows
    through. With d_ffn = (8/3) * d_model (PaLM convention, applied in
    ModelConfig.__post_init__), total FFN params match a GELU-FFN with
    d_ffn = 4*d_model — controlled comparison in the ablation.

    Naming: gate_proj, up_proj, down_proj follow the LLaMA convention.
    down_proj triggers the residual init scaling in model.py.
    """

    def __init__(self, d_model: int, d_ffn: int, bias: bool = False, dropout: float = 0.0):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ffn, bias=bias)
        self.up_proj = nn.Linear(d_model, d_ffn, bias=bias)
        self.down_proj = nn.Linear(d_ffn, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


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

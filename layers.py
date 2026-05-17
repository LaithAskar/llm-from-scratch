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
        kv_cache: Optional[dict] = None,
    ) -> torch.Tensor | tuple[torch.Tensor, dict]:
        """
        kv_cache: optional inference-time cache.
          - None  -> training / non-cached path; returns tensor only.
          - dict (possibly empty, possibly populated with "k" and "v") ->
            cached path; returns (out, updated_cache).

        Cached path semantics:
          - If kv_cache is empty: prefill mode. x is the full prompt.
          - If kv_cache has "k"/"v": decode mode. x is typically (B, 1, C);
            the new K/V are concatenated to the cached history before scoring.

        The caller is responsible for mask shape — during prefill, pass a
        (T_prompt, T_prompt) causal mask; during decode, pass None (single
        new token can attend to all history by construction).
        """
        B, T, C = x.shape
        H, D = self.num_heads, self.head_dim

        q = self.q_proj(x).view(B, T, H, D).transpose(1, 2)  # (B, H, T, D)
        k = self.k_proj(x).view(B, T, H, D).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, D).transpose(1, 2)

        # Position offset for RoPE: how many tokens are already in the cache.
        # 0 in the non-cached or prefill-with-empty-cache case.
        pos_offset = 0
        if kv_cache is not None and "k" in kv_cache:
            pos_offset = kv_cache["k"].size(-2)

        if self.rotary is not None:
            q, k = self.rotary(q, k, position_offset=pos_offset)

        # Concat with cached K/V if any. Store ALREADY-ROTATED K so we don't
        # re-rotate on future calls.
        if kv_cache is not None:
            if "k" in kv_cache:
                k = torch.cat([kv_cache["k"], k], dim=-2)
                v = torch.cat([kv_cache["v"], v], dim=-2)
            new_kv_cache = {"k": k, "v": v}

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(D)

        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)

        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.out_proj(out)

        if kv_cache is not None:
            return out, new_kv_cache
        return out


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

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        position_offset: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # q, k expected shape: (B, H, T, D), with D == self.head_dim.
        # position_offset: starting position for these tokens. 0 for the
        # normal training/forward case. >0 during cached generation, where
        # the new tokens sit at positions [cache_len, cache_len+T).
        T = q.size(-2)
        end = position_offset + T
        if end > self.cos_cached.size(0):
            raise ValueError(
                f"position {end} exceeds RoPE max_seq_len {self.cos_cached.size(0)}"
            )
        cos = self.cos_cached[position_offset:end]  # (T, D/2)
        sin = self.sin_cached[position_offset:end]
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


class MoEFFN(nn.Module):
    """
    Mixture-of-Experts feed-forward (Shazeer et al. 2017 / Switch Transformer
    style). Each token is routed to top-k of N experts; the output is a
    convex combination of the chosen experts' outputs (weights = renormalized
    router probabilities).

    Param budget: with N experts each at inner dim d_ffn_per_expert, total
    expert params are N * 2 * d_model * d_ffn_per_expert (two linears per
    expert, bias=False). To match a baseline GeluFFN with d_ffn = 4*d_model,
    set d_ffn_per_expert = 4*d_model / N (rounded). For N=4, d_model=128:
    d_ffn_per_expert = 128 — matches the GELU baseline's 131K FFN params
    within a few hundred (router adds N*d_model).

    Load-balancing aux loss (Switch Transformer eq. 4):
        aux = N * sum_i (f_i * P_i)
    where f_i is the fraction of tokens routed to expert i (any of top-k),
    and P_i is the mean router probability for expert i. Perfectly balanced
    -> aux = 1. Collapsed (one expert) -> aux = N. Multiply by a small
    coefficient (default 0.01) and add to the main CE loss.

    Stored as self.last_aux_loss for the model to pick up after forward.
    Set to None until first forward.

    For inference, aux loss is still computed but the caller can ignore it.
    """

    def __init__(
        self,
        d_model: int,
        d_ffn_per_expert: int,
        num_experts: int,
        top_k: int = 2,
        bias: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        if top_k > num_experts:
            raise ValueError(f"top_k ({top_k}) > num_experts ({num_experts})")
        self.num_experts = num_experts
        self.top_k = top_k
        self.router = nn.Linear(d_model, num_experts, bias=False)
        # Each expert is a small GELU FFN. SwiGLU would also work; using
        # GELU here so the MoE variant is clearly "baseline + MoE" rather
        # than confounded with the activation switch.
        self.experts = nn.ModuleList([
            GeluFFN(d_model=d_model, d_ffn=d_ffn_per_expert, bias=bias, dropout=dropout)
            for _ in range(num_experts)
        ])
        self.last_aux_loss: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        x_flat = x.reshape(B * T, C)

        # Router. Computed in fp32 to keep softmax + load-balancing math stable.
        router_logits = self.router(x_flat.float())               # (B*T, N)
        router_probs = F.softmax(router_logits, dim=-1)           # (B*T, N)

        # Top-k selection. Renormalize so each token's top-k weights sum to 1.
        top_probs, top_idx = router_probs.topk(self.top_k, dim=-1)   # both (B*T, k)
        top_probs = top_probs / top_probs.sum(dim=-1, keepdim=True)

        # Aux loss (computed before any reduction by expert routing).
        # f_i: fraction of tokens with expert i in their top-k.
        N = self.num_experts
        token_count = float(B * T)
        # one-hot over top-k selections: (B*T, k, N)
        one_hot = F.one_hot(top_idx, num_classes=N).float()
        # Any-of-top-k assignment per (token, expert): (B*T, N)
        assigned = one_hot.sum(dim=1).clamp(max=1.0)
        f_i = assigned.sum(dim=0) / token_count                   # (N,)
        P_i = router_probs.mean(dim=0)                            # (N,)
        self.last_aux_loss = N * (f_i * P_i).sum()

        # Combine expert outputs. Loop over experts (small N), gather
        # assigned tokens, compute, scatter back with the routing weight.
        out_flat = torch.zeros_like(x_flat)
        for e in range(N):
            # Mask of tokens routed to this expert (in any top-k slot).
            slot_mask = (top_idx == e)                            # (B*T, k) bool
            token_mask = slot_mask.any(dim=-1)                    # (B*T,) bool
            if not token_mask.any():
                continue
            # Per-token weight for this expert = sum over its top-k slots
            # of the renormalized probability mass on this expert.
            weight = (top_probs * slot_mask.to(top_probs.dtype)).sum(dim=-1)  # (B*T,)
            # Compute the expert's output only on its assigned tokens.
            expert_in = x_flat[token_mask]
            expert_out = self.experts[e](expert_in)
            # Add the weighted contribution back into the output buffer.
            out_flat[token_mask] = out_flat[token_mask] + weight[token_mask].unsqueeze(-1) * expert_out

        return out_flat.reshape(B, T, C)


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

        # Norm picker (used twice — once before attn, once before ffn).
        def make_norm():
            if config.norm_type == "rmsnorm":
                return RMSNorm(config.d_model, eps=config.norm_eps)
            return nn.LayerNorm(config.d_model, eps=config.norm_eps, bias=config.bias)

        self.norm1 = make_norm()
        self.norm2 = make_norm()

        # Optional RoPE — handed to MHA as its `rotary` callable. None for
        # learned-absolute pos (handled at the LM embedding level in model.py).
        rotary: Optional[RotaryEmbedding] = None
        if config.pos_encoding == "rope":
            rotary = RotaryEmbedding(
                head_dim=config.head_dim,
                max_seq_len=config.context_len,
                base=config.rope_base,
            )

        self.attn = MultiHeadAttention(
            embed_dim=config.d_model,
            num_heads=config.n_head,
            dropout=config.dropout,
            bias=config.bias,
            rotary=rotary,
        )

        # FFN picker — MoE takes precedence if enabled, else the activation switch.
        if getattr(config, "num_experts", 0) >= 2:
            # d_ffn_per_expert: param-matched to baseline (4*d_model)/N.
            # config.d_ffn was resolved at config-construction time to 4*d_model
            # (GELU default) or (8/3)*d_model (SwiGLU). We use the GELU figure
            # divided by N so MoE matches the GELU-FFN baseline at the same N.
            d_ffn_per_expert = max(1, (4 * config.d_model) // config.num_experts)
            self.ffn = MoEFFN(
                d_model=config.d_model,
                d_ffn_per_expert=d_ffn_per_expert,
                num_experts=config.num_experts,
                top_k=config.top_k_experts,
                bias=config.bias,
                dropout=config.dropout,
            )
        elif config.activation == "swiglu":
            self.ffn = SwiGLUFFN(
                d_model=config.d_model,
                d_ffn=config.d_ffn,
                bias=config.bias,
                dropout=config.dropout,
            )
        else:
            self.ffn = GeluFFN(
                d_model=config.d_model,
                d_ffn=config.d_ffn,
                bias=config.bias,
                dropout=config.dropout,
            )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[dict] = None,
    ) -> torch.Tensor | tuple[torch.Tensor, dict]:
        # Pre-norm: residual stream stays a clean identity path; norms sit
        # *inside* each sublayer branch (Xiong et al. 2020). Empirically
        # stable at depth without warmup gymnastics.
        if kv_cache is None:
            h = x + self.attn(self.norm1(x), mask=mask)
            h = h + self.ffn(self.norm2(h))
            return h
        else:
            attn_out, new_kv = self.attn(self.norm1(x), mask=mask, kv_cache=kv_cache)
            h = x + attn_out
            h = h + self.ffn(self.norm2(h))
            return h, new_kv


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

# `layers.py` cheatsheet — reference, not implementation

Study reference for implementing `layers.py` from scratch. **Prose, equations,
and tensor-shape transformations only.** No pasteable PyTorch code — the
translation from "compute Q, K, V" to `q = self.q_proj(x)` is the exercise.

Use this to avoid re-learning theory while implementing. Read the section,
then close it and write the code.

**Notation:**
- `B` = batch size, `T` = sequence length, `C = d_model` = embedding dim,
  `H` = num_heads, `D = C / H` = head_dim, `V` = vocab_size, `F` = FFN inner dim.
- Tensor shapes written as `(B, T, C)` etc. Last dim is contiguous in memory.

**Order of attack (dependency order):**
1. `causal_mask` (trivial — 1 line)
2. `RMSNorm` (small, self-contained, tests it independently)
3. `MultiHeadAttention` (the big one)
4. FFN helpers — `GELU FFN`, `SwiGLU FFN` (small)
5. RoPE helper (medium, math-heavy)
6. `TransformerBlock` (just wires the above together)

---

## 1. `causal_mask(seq_len, device)`

**What it does:** returns a `(T, T)` boolean lower-triangular matrix.
`mask[i, j] = True` means position `i` may attend to position `j`.
For autoregressive decoding, this blocks attending to future positions.

**Math:** `mask[i, j] = (j ≤ i)`.

**Implementation:** one call to a torch function that builds a lower-triangular
matrix of ones, cast to bool, on the requested device. Look up `torch.tril`.

**Interview Qs:**
- *Why bool and not float?* — Saves memory (1 byte vs 4) and the downstream
  `masked_fill` op takes a bool mask directly.
- *Why pre-compute and slice at forward time vs build per call?* — Avoids
  per-step allocation. `model.py` registers it as a buffer at `context_len`
  and slices `[:T, :T]` per forward.

---

## 2. `RMSNorm(dim, eps)`

**What it does:** scales each token's activation vector to have unit
root-mean-square, then applies a learned per-channel gain. Drops LayerNorm's
mean-subtraction and bias.

**Forward equation:**

```
rms(x) = sqrt(mean(x², axis=-1) + eps)
y      = (x / rms(x)) * weight
```

where `weight ∈ ℝ^dim`, initialized to ones (so it's identity at init).

**Shapes:**

| Tensor       | Shape         |
|--------------|---------------|
| `x` (input)  | `(..., dim)`  |
| `mean(x²)`   | `(..., 1)`    |
| `rms(x)`     | `(..., 1)`    |
| `y` (output) | `(..., dim)`  |

**Implementation steps (prose):**

1. In `__init__`: store `eps`, create a learnable parameter `weight` of
   shape `(dim,)` initialized to ones.
2. In `forward`:
   1. Remember the input dtype.
   2. Promote `x` to fp32 *before* computing mean-of-squares — see pitfall below.
   3. Compute mean of `x²` over the last dim, keeping dim for broadcasting.
   4. Compute the reciprocal-sqrt (look up `torch.rsqrt`) of `mean_sq + eps`.
   5. Multiply `x * rsqrt(...)` — this is the normalization.
   6. Cast back to the original dtype.
   7. Multiply by `weight` (broadcasts over leading dims).
   8. Return.

**Pitfall — bf16/fp16:** if `x` is bf16, computing `x²` in bf16 loses precision
because of the limited mantissa. The mean across hundreds of squared values
underflows or accumulates catastrophic error. **Promote to fp32 for the
reduction, cast back after.** LLaMA's reference impl does exactly this; nanoGPT
mirrors it. This is a real bug if skipped, not theoretical.

**Why no bias?** Empirically not load-bearing for transformer LMs. LayerNorm's
bias was inherited from BatchNorm and removed in RMSNorm without quality loss.

**Interview Qs:**
- *Why drop the mean?* — Cheaper (one fewer reduction over the last dim), and
  empirically as good or better on transformer LMs. The mean-subtraction was
  inherited from BatchNorm; transformer hidden states don't actually need it.
- *Why is `weight` per-channel (d-dim) and not scalar?* — Different channels
  carry different feature scales; a scalar would force them all to the same
  post-norm magnitude, losing expressivity. The per-channel gain lets the
  model re-amplify the channels it cares about.
- *Why fp32 for the reduction?* — See pitfall above. Numerical stability.
- *Why `rsqrt` not `1/sqrt`?* — Single fused op on GPU, slightly faster and
  more numerically stable than divide-after-sqrt.

**Reference:** Zhang & Sennrich, *Root Mean Square Layer Normalization* (2019).
LLaMA reference: `model.py` → `class RMSNorm`.

---

## 3. `MultiHeadAttention(embed_dim, num_heads, dropout)`

The centerpiece. Think in *shapes first* — most of the work is reshaping
tensors so that one batched matmul does H heads' worth of attention in
parallel.

### Forward equation (per head h)

```
Q_h = x · W_Q^h          shape: (B, T, D)
K_h = x · W_K^h          shape: (B, T, D)
V_h = x · W_V^h          shape: (B, T, D)

scores_h = (Q_h · K_h^T) / sqrt(D)       shape: (B, T, T)
scores_h = scores_h + mask_bias          (additive, see step 4)
attn_h   = softmax(scores_h, axis=-1)    shape: (B, T, T)
out_h    = attn_h · V_h                  shape: (B, T, D)
```

Concat across heads, then project:

```
out = concat([out_0, ..., out_{H-1}], axis=-1) · W_O      shape: (B, T, C)
```

In practice you don't loop over heads. You stack the H projections into one
weight matrix of shape `(C, C)`, then reshape the output to expose the head
dim. **Everything below is just a way to do all H heads as one batched matmul.**

### `__init__` steps (prose)

1. Assert `embed_dim % num_heads == 0` with a clear error message — head_dim
   must be integral.
2. Store `embed_dim`, `num_heads`, `head_dim = embed_dim // num_heads`.
3. Create three `Linear` projections (Q, K, V), each `embed_dim → embed_dim`.
   - Decide on bias. Modern LLMs (LLaMA, PaLM) use `bias=False` for these.
     Vaswani used `bias=True`. Either is defensible; cite the choice. `bias`
     comes from `config.bias` in your project — read it.
4. Create the output projection `Linear(embed_dim → embed_dim)`.
5. Create `nn.Dropout(dropout)` for attention weights.

(Optional alt: a single Q/K/V combined `Linear(C → 3*C)` then split. Faster.
Don't bother for the first pass.)

### `forward(x, mask)` steps (prose, shape-annotated)

Input: `x` of shape `(B, T, C)`. Optional `mask` of shape `(T, T)` (or
broadcastable to `(B, 1, T, T)`).

1. **Project Q, K, V.** Apply the three Linear layers. Each result is `(B, T, C)`.
2. **Expose head dim.** Reshape each from `(B, T, C)` to `(B, T, H, D)`, then
   transpose dims 1 and 2 to get `(B, H, T, D)`. Now heads sit on a batch-like
   axis — every head computes its own attention independently as part of one
   batched matmul.
3. **Scaled scores.** Compute `Q · K^T` over the last two dims of K. Result
   is `(B, H, T, T)`. Divide by `sqrt(D)`.
4. **Apply mask.** If a mask is provided, use `masked_fill` to set the
   positions where `mask == False` (or `0`) to `-inf`. Be careful about
   broadcasting: a `(T, T)` mask broadcasts to `(B, H, T, T)` automatically
   if you don't add a leading dim explicitly.
5. **Softmax.** Apply softmax over the last dim (the "keys" dim). Each row
   of the `(T, T)` block now sums to 1.
6. **Dropout the weights.** Apply `self.attn_drop` to the softmaxed weights.
7. **Weighted sum of values.** Compute `attn · V`, shape `(B, H, T, D)`.
8. **Merge heads.** Transpose back to `(B, T, H, D)`, call `.contiguous()`
   (because transpose breaks contiguity, and the next `.view()` requires it),
   then reshape to `(B, T, C)`.
9. **Output projection.** Apply `self.out_proj`, shape `(B, T, C)`. Return.

### Shape table (cheat reference)

| Step | Tensor          | Shape           |
|------|-----------------|-----------------|
| 1    | Q / K / V       | `(B, T, C)`     |
| 2    | Q / K / V       | `(B, H, T, D)`  |
| 3    | scores          | `(B, H, T, T)`  |
| 4    | scores (masked) | `(B, H, T, T)`  |
| 5    | attn weights    | `(B, H, T, T)`  |
| 7    | out             | `(B, H, T, D)`  |
| 8    | out             | `(B, T, C)`     |
| 9    | out             | `(B, T, C)`     |

### Interview Qs (these are likely)

- **Why divide by `sqrt(D)`?**
  Q and K have variance ~1 per element at init (Gaussian). Their dot product
  is a sum of D iid products, so it has variance ~D and stdev ~`sqrt(D)`.
  Without the scaling, softmax inputs grow with D, pushing it into the
  saturated regime where one logit dominates and gradients vanish. Scaling
  by `sqrt(D)` keeps the score variance ~1 regardless of head dim.

- **Why softmax over the last dim?**
  In a `(B, H, T, T)` score tensor, the last dim is the "keys" axis —
  for each query position, you're computing a distribution *over which
  positions to attend to*. The distribution must sum to 1, so softmax goes
  over the keys dim. Softmax over the queries dim would be wrong (it'd
  normalize across queries, not produce per-query attention weights).

- **Why `-inf` for masked positions (not 0)?**
  Softmax is `exp(x) / Σ exp(x)`. Setting masked logits to `-inf` makes
  `exp(-inf) = 0`, so masked positions contribute zero probability and don't
  appear in the weighted sum. Setting masked logits to 0 would give them
  `exp(0) = 1`, i.e., as much weight as an unmasked logit of 0 — totally wrong.

- **Why does the `(T, T)` score matrix motivate Flash Attention / KV cache?**
  Score memory scales as `O(B · H · T²)`. At T=2048, B=8, H=32, fp16: that's
  ~2 GB just for one layer's attention scores. Flash Attention recomputes
  scores in tile-sized chunks that fit in SRAM and never materializes the
  full `T²` matrix in HBM. KV cache addresses a different problem (avoiding
  re-encoding past tokens during autoregressive decoding) but is also
  motivated by the same scaling.

- **Why bias=False on the QKV linears?**
  Empirically doesn't matter; saves a tiny number of params. Some argue the
  attention computation is invariant to the bias (it'd just shift Q and K
  uniformly, which subtracts out after softmax up to numerical effects).
  LLaMA/PaLM both use bias=False. Vaswani used bias=True; not a load-bearing
  difference.

### Pitfalls

- **Forgetting `.contiguous()` before `.view()`** after transpose. PyTorch
  will give an unhelpful error. Use `.reshape()` if you want it to do the
  contiguous-or-copy decision implicitly.
- **Mask broadcasting.** A `(T, T)` mask broadcasts to `(B, H, T, T)` if you
  don't pre-unsqueeze. A `(B, T, T)` mask does NOT — it'd try to broadcast
  against H and fail. Sanity-check shapes on a tiny case.
- **Mask True/False convention.** The project's `causal_mask` uses
  `True == attend, False == mask out`. Your `masked_fill` call therefore
  needs to mask where `mask == False`, not where `mask == True`. Easy to
  invert by mistake — write a one-line test that checks position 0 cannot
  attend to position 1 after applying the causal mask.
- **Dropout in eval mode.** `nn.Dropout` is a no-op in eval mode automatically.
  Don't manually disable it.

**Reference:** Vaswani et al., *Attention Is All You Need* (2017), §3.2.
nanoGPT's `model.py` → `class CausalSelfAttention` is the cleanest reference
implementation; read it AFTER you've written yours.

---

## 4. FFN helpers

### 4a. `GELU FFN` (the classic)

**Forward equation:**

```
y = W2 · GELU(W1 · x)
```

Two linears, one nonlinearity in between. `W1: C → F`, `W2: F → C`.

For `d_model = C`, the standard inner dim is `F = 4 * C`. So the FFN has
`2 * C * 4C = 8C²` params (vs attention's `4C²`), making it the bigger
chunk of model params per block.

**Implementation steps (prose):**
1. `__init__`: two `Linear` layers (`C → F` and `F → C`), one `nn.GELU()`.
2. `forward`: linear → gelu → linear → return.

**Naming convention** for `model.py`'s residual-init scaling to work: the
second linear should be an attribute called `down_proj` (matches the suffix
pattern in `model.py:_init_weights`).

### 4b. `SwiGLU FFN` (the modern one)

**Forward equation:**

```
y = W_down · ( silu(W_gate · x) ⊙ (W_up · x) )
```

Three matrices, not two. `W_gate: C → F`, `W_up: C → F`, `W_down: F → C`.
`silu(z) = z * sigmoid(z)` (also called SiLU or Swish-1).

**Param matching:** the project's `ModelConfig.__post_init__` sets
`F = (8/3) * C` rounded to a multiple of 64 for SwiGLU, so total FFN params
match a GELU-FFN with `F = 4C`. **This is the whole point of the
`(8/3) * d_model` rule** — controlled comparison in your ablation.

**Implementation steps (prose):**
1. `__init__`: three `Linear` layers (gate, up, down). One `nn.SiLU()`.
2. `forward`: compute `silu(gate(x)) * up(x)`, then `down_proj` of that.
3. Same naming: call the C→F gate matrix `gate_proj`, the C→F up matrix
   `up_proj`, the F→C output `down_proj`.

**Interview Qs:**
- *Why three matrices?* — The element-wise product `silu(gate(x)) * up(x)`
  is the "gated linear unit" — a learned gate decides per-channel how much
  of `up(x)` to let through. Two-matrix FFNs can't do this kind of
  multiplicative gating.
- *Why SiLU and not GELU?* — Mostly empirical. PaLM and LLaMA found SwiGLU >
  GeGLU > GLU on perplexity at matched params. SiLU is smoother near 0
  than ReLU, gradient-friendlier than GELU.
- *Why `(8/3) * d_model`?* — Param-matching against a GELU FFN with `4 * d_model`.
  Two matrices of size `C × 4C` = `8C²`. Three matrices of size `C × F` = `3CF`.
  Setting `3CF = 8C²` gives `F = (8/3) * C`. Round to multiple of 64 for GPU
  alignment.

**Reference:** Shazeer, *GLU Variants Improve Transformer* (2020). PaLM §3
for the matched-FFN rule.

---

## 5. RoPE — Rotary Position Embedding

The most math-heavy part. Skip first pass if needed; learned-positional in
`model.py` works without it. Required for the `rope` and `modern` ablation
variants.

**Intuition:** instead of *adding* a position vector to the embedding
(absolute pos-emb), *rotate* the Q and K vectors by a position-dependent
angle before computing attention. Because attention scores are dot products
and rotations preserve dot products *within a head*, the resulting score
depends only on the *relative* offset between positions — `score(Q_m, K_n)`
becomes a function of `m - n`, not `m` and `n` separately. Better
extrapolation to unseen lengths and no extra parameters.

### Setup (do once, cache)

For head dim `D` (which must be even), define D/2 frequencies:

```
θ_i = base^(-2i / D)        for i = 0, 1, ..., D/2 - 1
```

where `base = 10000` typically (config's `rope_base`).

For each position `m`, the per-frequency angle is `m * θ_i`. Precompute
`cos(m θ_i)` and `sin(m θ_i)` for all `m ∈ [0, max_seq_len)` and all `i`.
Cache them as buffers (non-persistent) on the module.

Cached shape: `(max_seq_len, D/2)` for both cos and sin.

### Applying RoPE to Q (and K)

Q has shape `(B, H, T, D)`. Treat the last dim as `D/2` pairs of consecutive
elements `(q_0, q_1), (q_2, q_3), ..., (q_{D-2}, q_{D-1})`. Each pair gets
rotated by angle `m θ_i` where `m` is its position and `i` is which pair:

```
[q'_{2i}  ]   [cos(mθ_i)  -sin(mθ_i)] [q_{2i}  ]
[q'_{2i+1}] = [sin(mθ_i)   cos(mθ_i)] [q_{2i+1}]
```

Which is equivalent to:

```
q'_{2i}   = q_{2i}   * cos(mθ_i) - q_{2i+1} * sin(mθ_i)
q'_{2i+1} = q_{2i}   * sin(mθ_i) + q_{2i+1} * cos(mθ_i)
```

Same transform for K. Do NOT rotate V.

**Two common indexing conventions** — pick one and stick to it:
- *Interleaved:* `(q_0, q_1, q_2, q_3, ...)` are pairs `(0,1), (2,3), ...`.
- *Half-split:* first half is "real", second half is "imag" — pairs are
  `(q_0, q_{D/2}), (q_1, q_{D/2+1}), ...`. nanoGPT and LLaMA reference
  use this — it's slightly faster (one slice instead of strided indexing).

The half-split version is what the LLaMA reference and the HuggingFace
implementation use. Either works mathematically; the model only needs the
same convention at Q and K.

### Implementation steps (prose)

1. `__init__(self, head_dim, max_seq_len, base=10000)`:
   1. Assert `head_dim` is even.
   2. Compute the `D/2` inverse frequencies as `1.0 / (base ** (arange(0, D, 2) / D))`.
   3. Build a `(max_seq_len,)` arange of positions.
   4. Outer product → `(max_seq_len, D/2)` of angles.
   5. Compute and register cos/sin as non-persistent buffers.

2. `apply_rotary(q, k, seq_len)`:
   1. Slice the cached cos/sin to the current seq_len.
   2. Rotate q using the cos/sin (vectorized — no python loop over positions).
   3. Same for k.
   4. Return rotated q, k.

3. In `MultiHeadAttention.forward`, between steps 2 (head-split) and 3
   (scores), apply rotary to q and k *only if* the config says so. Pass
   `pos_encoding` and an instantiated rope module via `config` or via
   constructor.

### Interview Qs

- *Why does RoPE produce relative positions?*
  Dot product of two rotated vectors `R(mθ)·q` and `R(nθ)·k` equals
  `q · R((m-n)θ) · k`. Rotations commute through the inner product as a
  difference of angles — score depends only on `m - n`.
- *Why even head_dim?* — RoPE pairs adjacent elements; odd dim has a leftover.
- *Why base=10000?* — Inherited from Transformer's sinusoidal positional
  encoding. Controls the wavelength range across pair indices. Larger base
  → longer wavelengths → better long-context extrapolation. LLaMA-2 increased
  it to 500000 for longer context.
- *Why no parameters?* — All angles are deterministic from position and
  pair index. Saves params and is a strong inductive bias.
- *Why don't we rotate V?* — V carries content, not position. We want
  position to influence *which* values to mix (via the score), not the
  values themselves.

**Reference:** Su et al., *RoFormer: Enhanced Transformer with Rotary Position
Embedding* (2021). LLaMA reference impl: `apply_rotary_emb`.

---

## 6. `TransformerBlock(config)`

The plumbing block. Architectural choices come from `config`; this just
wires them up.

### Forward equation (pre-norm)

```
h = x + attn(norm1(x), mask)
y = h + ffn(norm2(h))
return y
```

Pre-norm = "normalize *before* the sublayer." Post-norm = "normalize *after*
adding the residual." Modern transformers use pre-norm. The difference matters
at depth — see interview Q below.

### `__init__` steps (prose)

1. Look up `config.norm_type` → instantiate `RMSNorm` or `nn.LayerNorm` twice
   (norm1, norm2). For LayerNorm, use `eps=config.norm_eps` and
   `bias=config.bias`.
2. Instantiate `MultiHeadAttention(embed_dim=d_model, num_heads=n_head,
   dropout=dropout)`. If RoPE is selected and you handle it inside MHA, pass
   the rope module / config too.
3. Look up `config.activation` → instantiate `GELU FFN` or `SwiGLU FFN` with
   inner dim `config.d_ffn`.

### `forward(x, mask)` steps (prose)

1. Compute attention sublayer: `h = x + attn(norm1(x), mask=mask)`. Residual
   is added; norm is applied to the input *before* attention.
2. Compute FFN sublayer: `y = h + ffn(norm2(h))`. Same pattern.
3. Return `y`.

### Interview Qs

- **Why pre-norm beats post-norm at depth.**
  Post-norm sits on the residual stream itself, so gradients have to flow
  through every norm on their way back. At depth, this requires careful
  learning-rate warmup (Vaswani used 4000 warmup steps for a 6-layer model)
  and is unstable past ~12 layers without tricks.

  Pre-norm puts the norm *inside* the sublayer branch, so the residual stream
  is a clean identity path. Gradients flow straight back via the residual,
  bypassing every norm. Empirically trains stably at 100+ layers without
  warmup gymnastics. See Xiong et al., *On Layer Normalization in the
  Transformer Architecture* (2020) for the analysis.

- **Why two norms per block, not one or three?**
  One norm per sublayer (attn and FFN are two sublayers, hence two norms).
  Each sublayer's input distribution needs stabilization independently.
  Variants exist: NormFormer adds a third norm post-FFN; GPT-J merges
  some norms; T5 uses RMSNorm without center. The two-norm pattern is the
  vanilla baseline.

- **Why FFN dim = 4 * d_model (or 8/3 for SwiGLU)?**
  4× is empirical from Vaswani — chosen to give the FFN ~2x the param count
  of attention, which is where most representational capacity sits.
  Sweeping it shows a broad optimum around 4×. The `(8/3) * d_model` for
  SwiGLU is the param-matched equivalent (see SwiGLU FFN section).

### Pitfall

If `norm_type == "rmsnorm"`, the *final* norm in `model.py` is also RMSNorm
— make sure your block uses the same norm type. The project's `model.py`
already handles this via `_make_norm`. Just don't hardcode `nn.LayerNorm`
inside the block.

---

## Sanity tests you can run mentally before pytest

For each component, before running the suite, ask:

- **MHA:** Does `attention(x) when x is random` produce output of the same
  shape as input? Does gradient flow back to all parameters? With causal
  mask, is position 0's output independent of position 1's input?
- **RMSNorm:** Does `rms_norm(x).pow(2).mean(-1)` ≈ 1.0 at init (since
  weight=1)? Does dtype round-trip preserve dtype?
- **TransformerBlock:** Does the block satisfy `block(0) == 0` after init
  (since residuals add zero through identity-initialized sublayers — not
  exactly, but close). Does stacking 2 blocks preserve shape?

If any of these fail, debug *before* running the full pytest suite — narrow
the surface area.

---

## When `layers.py` is done

1. Run `python layers.py` — there's a smoke test in `__main__` for MHA.
2. Delete `tests/_dummies.py` and the `patch_layers` fixture in
   `tests/conftest.py`. The 28-test suite now exercises your real code.
3. Run `pytest tests/ -v` — should still be 28 passing.
4. Run `python data.py prepare` to download TinyStories (multi-GB, takes a
   while).
5. Smoke training: `python ablation.py --max-steps 500 --variants baseline`
   — should produce a `runs/ablation/baseline/seed_0/log.csv` with
   decreasing train loss.
6. Full ablation: drop `--max-steps`, let it run all 5 variants × seeds.

---

## What this cheatsheet deliberately does not give you

- **Pasteable PyTorch lines.** The translation from prose to `nn.Linear(...)`
  is the interview-defensible exercise. If a recruiter asks "walk me through
  your MHA," you need the conversion of these steps to code to live in
  *your* head, not on this page.
- **Code review of your implementation.** Ask for that explicitly when you
  have a draft; I'll review math, shapes, naming, and edge cases.
- **The decision of which optional features to do.** RoPE and SwiGLU are
  required only if you want the `rope` / `swiglu` / `modern` ablation
  variants to run. You could ship Parts 1-3 + ablation with only the
  `baseline` and `rmsnorm` variants if time is short.

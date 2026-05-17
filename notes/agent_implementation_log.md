# `layers.py` implementation log

Per-component walkthrough of code written by the AI agent. Laith is PM/reviewer;
this log is the load-bearing artifact for that role. Each entry is structured so
that reading it in order leaves you able to defend any function in a code-defense
round.

**How to read this:** for each component, read the *Design* section first
(WHY the code is the way it is), then open the actual file and walk the code
with the *Walkthrough* section beside it.

---

## 1. `causal_mask(seq_len, device)`

**File:** `layers.py:108-117`
**Tests:** `tests/test_layers.py:18-67` (6 cases)
**Patch status:** real implementation; monkey-patch removed from `conftest.py`.

### Design

A causal mask for autoregressive decoders blocks position `i` from attending
to position `j > i`. The mask is applied to attention *scores* (the
`(B, H, T, T)` matmul of Q with K^T) via `masked_fill`, replacing blocked
positions with `-inf` so that softmax assigns them exactly zero probability.

The shape is `(T, T)` — not `(B, H, T, T)` — because every batch item and
every head share the same causal structure (every model.py forward sees the
same context-length, every head looks at the same positions). Broadcasting
expands it to `(B, H, T, T)` automatically at the `masked_fill` site without
allocating the larger tensor.

### Walkthrough

```python
return torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))
```

One line. Three things to defend:

1. **`torch.ones(seq_len, seq_len, ..., device=device)`** builds an all-ones
   matrix of the requested size on the requested device. Device is forwarded
   from the caller so the mask lives where Q/K/V live — no implicit CPU→GPU
   copy at attention time.

2. **`dtype=torch.bool`** uses 1 byte per element instead of 4 (fp32). For
   a `context_len=256` model this is `256² = 65,536` bytes vs 256 KB — small
   absolute savings, but right convention. `masked_fill` accepts a bool mask
   natively.

3. **`torch.tril(...)`** zeroes the strict upper triangle, leaving the
   lower-triangular part (including the main diagonal) as ones. After bool
   cast: `True` at `j ≤ i`, `False` at `j > i`. Convention used in this
   project: `True == attend`.

### Why this convention (True = attend, not the opposite)

Two opposite conventions exist in the wild:
- This project / Hugging Face: `True == attend`, `False == mask out`.
- PyTorch `nn.Transformer`: `True == mask out`, `False == attend`.

They're equivalent under negation, but the chosen convention determines the
sign of the `masked_fill` predicate. In `MultiHeadAttention.forward` we'll
write `scores.masked_fill(~mask, float("-inf"))` — note the bitwise NOT,
because `masked_fill` fills where the predicate is `True`.

The "True == attend" convention reads more naturally as English ("this token
attends to that one"), which is why it was chosen.

### Interview Qs you should be able to answer

- **Why `-inf` and not 0 in `masked_fill`?**
  Softmax is `exp(score) / Σ exp(score)`. `exp(-inf) = 0` → zero probability
  on masked positions. `exp(0) = 1` → masked positions get the same
  unnormalized weight as a real unmasked logit of 0. That would be a bug.

- **Why bool dtype and not int8 or float?**
  `masked_fill` and `torch.where` both accept bool predicates natively.
  Bool is 1 byte (matches int8) and the type signature makes the intent
  explicit. Float would work but adds implicit casts.

- **Why is the mask `(T, T)` instead of `(B, H, T, T)`?**
  Causal structure is identical across batch and heads — broadcasting
  handles the rest. Allocating `(B, H, T, T)` would multiply memory by
  `B*H` for no information gain.

- **Why does `model.py` register this as a buffer at `context_len` once
  and slice per forward, instead of recomputing per call?**
  Avoids per-step allocation in the hot path. Also keeps the mask on the
  same device as the rest of the model via `register_buffer`'s
  `model.to(device)` semantics.

### Test coverage

`tests/test_layers.py` (6 cases):
- `test_causal_mask_shape_and_dtype` — `(T, T)`, bool.
- `test_causal_mask_lower_triangular` — `mask[i,j] == (j <= i)` for every pair.
- `test_causal_mask_diagonal_is_attendable` — self-attention works (`mask[i,i] == True`).
- `test_causal_mask_blocks_future` — strict upper triangle is uniformly `False`.
- `test_causal_mask_works_with_masked_fill` — integration: with `~mask` predicate,
  masked positions become `-inf` and unmasked positions stay finite.
- `test_causal_mask_respects_device` — CPU device round-trips.

CUDA-device path isn't unit tested (CI runs on CPU), but it's covered indirectly
by `model.py`'s `register_buffer + .to(device)` flow in the `test_model.py` cases.

### Diff vs the stub

The stub raised `NotImplementedError`. The TODO comment in the stub already
specified the exact one-liner — no design freedom here, this is mechanical.
The interesting thing isn't *what* the code is, it's *why* the conventions
(bool dtype, True=attend, `(T,T)` shape, device-forwarded) are right for
how the function will be used downstream.

---

## 2. `RMSNorm(dim, eps)`

**File:** `layers.py:142-156`
**Tests:** `tests/test_layers.py:73-148` (7 cases)
**Patch status:** real implementation; monkey-patch removed from `conftest.py`.

### Design

Root Mean Square LayerNorm (Zhang & Sennrich, 2019). Drops LayerNorm's
mean-subtraction and bias term. The per-token computation is:

```
rms(x) = sqrt(mean(x², axis=-1) + eps)
y      = (x / rms(x)) * weight
```

`weight` is a learned per-channel gain of shape `(dim,)` initialized to ones.
At init this means the forward pass is essentially "rescale each token to
unit RMS" — the gain only starts to differentiate channels via gradient
updates.

**Why drop the mean?** Empirically as good or better on transformer LMs
(LLaMA, PaLM, Mistral all use RMSNorm) and ~30% fewer FLOPs per token
(no second reduction, no mean-subtraction, no bias add).

**Why per-channel gain (not scalar)?** Different channels of `d_model`
carry features at different scales. A scalar gain would force the entire
hidden state to a single post-norm magnitude; per-channel gain lets the
model amplify channels it cares about.

### Walkthrough

```python
def __init__(self, dim: int, eps: float = 1e-5):
    super().__init__()
    self.eps = eps
    self.weight = nn.Parameter(torch.ones(dim))
```

Standard `nn.Module` init. `self.weight` is registered as a learnable
parameter via `nn.Parameter` — this makes it appear in
`model.parameters()`, get gradient updates, move with `.to(device)`, and
serialize in `state_dict`. Init to ones means RMSNorm is *identity gain*
at step 0; the model only diverges from identity-scale via training.

`eps` is stored as a plain attribute (not a parameter or buffer) because
it's a constant float, not a tensor to be moved or updated.

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    orig_dtype = x.dtype
    x_f32 = x.to(torch.float32)
    rms = torch.rsqrt(x_f32.pow(2).mean(dim=-1, keepdim=True) + self.eps)
    return (x_f32 * rms).to(orig_dtype) * self.weight
```

Five things to defend:

1. **`orig_dtype = x.dtype` + `x.to(torch.float32)`.** This is the
   load-bearing precision fix. Under AMP (the default in this project,
   `dtype="bf16"` in `TrainConfig`), inputs arrive as bf16. bf16 has only
   ~3 decimal digits of mantissa. Computing `x²` and then averaging across
   `d_model` (e.g. 192-1024 elements) in bf16 produces a mean that's off by
   several percent, which then propagates through the rsqrt into the output.
   Promoting to fp32 for the reduction is what LLaMA's reference impl does;
   skipping this is one of the most common silent-bug RMSNorm
   implementations. `test_rmsnorm_promotes_bf16_input_to_fp32_for_stability`
   explicitly fails a naive bf16 implementation.

2. **`x_f32.pow(2).mean(dim=-1, keepdim=True)`.** Mean of squares over the
   last dim (which is `dim`). `keepdim=True` preserves the dim for clean
   broadcasting against `x`, so the result is shape `(..., 1)`.

3. **`torch.rsqrt(... + self.eps)`.** `rsqrt(z) = 1/sqrt(z)` as a single
   fused op. Two reasons to prefer it over `1.0 / torch.sqrt(z)`:
   (a) fused = one kernel launch instead of two,
   (b) numerically more stable near small z (the divide and sqrt error
   compose poorly otherwise). `eps` is added *inside* the sqrt to prevent
   `rsqrt(0)` blowups on all-zero inputs
   (`test_rmsnorm_eps_prevents_division_by_zero`).

4. **`(x_f32 * rms).to(orig_dtype)`.** Normalize in fp32, then cast back to
   the input dtype. After this line the tensor matches the rest of the
   network's dtype convention.

5. **`* self.weight`.** Per-channel gain applied last, in original dtype.
   At init weight=1 so this is a no-op; during training it lets the model
   re-amplify per-channel.

### Interview Qs you should be able to answer

- **Why no bias term?**
  Two reasons: (a) empirically not load-bearing for transformer LMs (the
  centered hidden states already have ~zero mean after the residual+norm
  loops), and (b) the bias was inherited from BatchNorm in the first
  LayerNorm paper without strong evidence it was necessary in the
  transformer context. Removing it saves `dim` parameters per norm.

- **Why fp32 for the reduction, when the rest of the model runs in bf16?**
  bf16's 3-digit mantissa loses precision catastrophically when you sum
  many squared values across `d_model`. The mean accumulates ~`d_model`
  rounding errors. fp32 has ~7 digits, which is enough. The cast back to
  bf16 at the end means downstream ops still get bf16 — only the small
  internal reduction runs in fp32.

- **Why `rsqrt` instead of `1/sqrt`?**
  One fused GPU kernel vs two. Also `rsqrt` is intrinsically more stable
  near small inputs.

- **Why `eps` *inside* the sqrt, not added to `rms` after?**
  Identical for large `mean_sq`, but on all-zero inputs `mean_sq=0` and
  `sqrt(0) = 0`, then `1/sqrt(0) = inf`. With `eps` inside: `sqrt(eps)`
  is small but finite, `rsqrt` gives a large but finite value.
  `test_rmsnorm_eps_prevents_division_by_zero` is the regression test.

- **What's the difference between RMSNorm and LayerNorm in one sentence?**
  RMSNorm drops LayerNorm's mean-subtraction and bias; you get unit RMS
  per token instead of unit variance per token, with one fewer reduction
  and `dim` fewer parameters.

- **What's the difference between RMSNorm and BatchNorm?**
  BatchNorm normalizes per-channel *across the batch*. RMSNorm normalizes
  per-token *across channels*. Batch statistics are unstable at small
  batches (problem for LMs with long sequences), and BatchNorm requires
  tracking running statistics for eval. RMSNorm has neither problem.

### Test coverage

`tests/test_layers.py` (7 cases):
- `test_rmsnorm_preserves_shape` — `(B, T, dim)` in, same out.
- `test_rmsnorm_weight_is_parameter_of_correct_shape` — `(dim,)` Parameter,
  init to ones.
- `test_rmsnorm_output_has_unit_rms_at_init` — per-token RMS of output is
  ~1 when weight=1, regardless of input scale.
- `test_rmsnorm_gradient_flows_to_input_and_weight` — both `x.grad` and
  `weight.grad` populate and are finite.
- `test_rmsnorm_promotes_bf16_input_to_fp32_for_stability` — the
  regression test for the fp32-promotion fix. Would fail on naive bf16.
- `test_rmsnorm_eps_prevents_division_by_zero` — all-zero input gives
  finite output.
- `test_rmsnorm_weight_scales_output` — weight=2 produces output 2× weight=1.

Plus the indirect coverage via `test_rmsnorm_branch_selects_rmsnorm` in
`tests/test_model.py:148-153`, which now asserts `isinstance(lm.final_norm,
RMSNorm)` against the real implementation (was previously against
`DummyRMSNorm`).

### Diff vs the stub

The stub had detailed TODO comments specifying every step. No design
freedom on the core algorithm. The only judgment call was:
- *Cast back to orig_dtype before or after the weight multiply?* The
  LLaMA reference casts back *before* weight, so weight*output happens in
  the model's working dtype. Followed that convention here.

---

## 3. `MultiHeadAttention(embed_dim, num_heads, dropout, bias, rotary)`

**File:** `layers.py:46-112`
**Tests:** `tests/test_layers.py:153-289` (8 cases)
**Patch status:** real implementation; MHA was never patched in conftest
(the patch was on `TransformerBlock`, which constructs MHA — MHA is still
unreachable through model.py until TransformerBlock lands).

### Design

Scaled dot-product multi-head attention, Vaswani et al. 2017 §3.2. The
whole implementation is ~25 lines because virtually all the work is
*reshaping tensors* so that one batched matmul computes H heads' attention
simultaneously.

**API choices that differ from the stub:**

1. **Added `bias: bool = False` constructor param.** The stub didn't take
   bias but the project's `ModelConfig.bias` exists and needs to plumb
   somewhere — TransformerBlock will pass `config.bias`. Defaulting to
   `False` matches LLaMA / PaLM convention. The stub's TODO explicitly
   flagged this as a design call.

2. **Added `rotary: Optional[Callable]` constructor param.** RoPE has to
   happen *inside* MHA, between head-split and the scores matmul, because
   it rotates Q and K. Rather than coupling MHA to the project's Config
   (`if config.pos_encoding == "rope":` inside MHA), I made it a callable
   slot. TransformerBlock will construct an `apply_rotary(q, k)` and pass
   it. `None` means no rotation (learned-absolute pos lives at the
   embedding level in model.py).

### Walkthrough (forward)

```python
def forward(self, x, mask=None):
    B, T, C = x.shape
    H, D = self.num_heads, self.head_dim
```

Unpack input shape and head dims. `C` (input embed_dim) equals `H * D`
(num_heads × head_dim) by construction.

```python
    q = self.q_proj(x).view(B, T, H, D).transpose(1, 2)  # (B, H, T, D)
    k = self.k_proj(x).view(B, T, H, D).transpose(1, 2)
    v = self.v_proj(x).view(B, T, H, D).transpose(1, 2)
```

Project x through three independent Linear layers (each `C → C`), then
reshape to expose the head dim and transpose to put heads in front of time.

- `view(B, T, H, D)` reinterprets the contiguous `(B, T, C)` tensor as
  `(B, T, H, D)` without copy. `view` requires contiguity, which `q_proj(x)`
  output has by default.
- `transpose(1, 2)` swaps the T and H axes, giving `(B, H, T, D)`. Heads
  now sit on a batch-like axis, so the next matmul broadcasts across them.

**Note:** transpose returns a non-contiguous view. That's fine for matmul
(which doesn't require contiguity) but matters later when we have to
re-merge heads — see the `.contiguous()` call below.

```python
    if self.rotary is not None:
        q, k = self.rotary(q, k)
```

The RoPE hook. No-op when not provided. When TransformerBlock wires this
up with `pos_encoding="rope"`, a Rotary module is passed that rotates each
Q and K vector by a position-dependent angle. V is *not* rotated — V
carries content, not position.

```python
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(D)
```

The attention scores matmul. `k.transpose(-2, -1)` swaps the last two dims
of K, so K becomes `(B, H, D, T)`. Then `q @ k_T`:

`(B, H, T, D) @ (B, H, D, T) -> (B, H, T, T)`

Each `(T, T)` block in the result is `score[q_pos, k_pos]` = dot product
of the query at `q_pos` with the key at `k_pos`. Divided by `sqrt(D)` to
keep score variance ~1 regardless of head dim (otherwise softmax
saturates as D grows; see interview Qs).

```python
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
```

The mask is `(T, T)` bool, project convention "True == attend." `masked_fill`
fills *where the predicate is True*, so we invert with `~` to fill where
the mask says False. `-inf` becomes `exp(-inf) = 0` after softmax, giving
masked positions zero attention weight.

Broadcasting: `(T, T)` against `(B, H, T, T)`. PyTorch right-aligns dims
and broadcasts singleton or missing dims, so the `(T, T)` mask is
implicitly `(1, 1, T, T)`. **The same mask applies to every batch and
every head**, which is correct for causal attention.

```python
    attn = F.softmax(scores, dim=-1)
    attn = self.attn_dropout(attn)
```

Softmax along the *keys* dim (last). For each query position, this gives
a probability distribution over which key positions to attend to. Dropout
applied to the attention weights themselves (standard practice; this is
sometimes called "attention dropout" vs the input dropout in
TransformerBlock).

```python
    out = attn @ v
```

`(B, H, T, T) @ (B, H, T, D) -> (B, H, T, D)`. Each output position is a
weighted sum of the value vectors using the per-position attention weights.

```python
    out = out.transpose(1, 2).contiguous().view(B, T, C)
    return self.out_proj(out)
```

Merge heads. Transpose back to `(B, T, H, D)`. `.contiguous()` is required
here because the transpose left the tensor non-contiguous; `view(B, T, C)`
needs contiguity to reinterpret the flat memory. Without `.contiguous()`,
PyTorch raises a stride error. (Alt: `.reshape(B, T, C)` would handle the
contiguity decision implicitly, at the cost of being slightly less explicit
about the cost.)

Final `out_proj` is a learned linear mixing across heads. This is what
turns the concatenated per-head outputs back into a single embed_dim
vector.

### Interview Qs you should be able to answer

- **Why divide scores by `sqrt(D)`?**
  Q and K elements are roughly N(0, 1) at init (Gaussian via `_init_weights`).
  Their dot product over D elements is a sum of D iid products, so it has
  variance ~D and standard deviation ~sqrt(D). Without rescaling, the
  softmax inputs grow as D grows, eventually pushing into the saturated
  regime where one logit dominates everything. Then `d(softmax)/d(input) → 0`
  and gradients vanish. Dividing by sqrt(D) keeps the score distribution
  at variance ~1 regardless of head dim, which keeps softmax in its
  responsive range.

- **Why softmax over the last dim (the keys axis)?**
  For each query position, we're computing "given this query, how much
  weight on each key position?" — that's a distribution over keys, so
  softmax goes over the keys axis. Softmax over the queries axis would
  normalize across queries, producing per-key attention weights, which is
  a wrong semantics.

- **Why `-inf` for masked positions instead of 0?**
  Softmax: `exp(score) / Σ exp(score)`. `exp(-inf) = 0` (zero probability).
  `exp(0) = 1` (same weight as an unmasked logit of 0). The 0-fill would
  give masked positions exactly as much weight as a real low-score position,
  which is a correctness bug, not a perf issue.

- **What's the memory cost of the `(T, T)` score matrix and why does it
  motivate Flash Attention?**
  Per layer: `B * H * T * T * sizeof(dtype)`. At `B=8, H=32, T=2048, fp16`
  that's ~2 GB *just for one layer's intermediate scores*, and it has to
  fit in HBM. Flash Attention restructures the computation so scores are
  computed in tile-sized chunks that fit in SRAM (much faster than HBM
  reads/writes) and the full `T*T` matrix never materializes. Trade: ~2x
  fewer FLOPs visible to PyTorch, ~3-9x faster wall-clock in practice.

- **Why bias=False on the Q/K/V linears?**
  Empirically not load-bearing. There's a structural argument too: a bias
  on Q is equivalent to shifting Q by a constant per position; if that
  same shift were also on K (it'd have its own bias), the dot product
  `Q·K^T = (q+b_q)·(k+b_k)^T` adds bias-dependent terms that softmax
  partially normalizes out. Most modern LLMs (LLaMA, PaLM, Mistral) use
  bias=False; the param savings are tiny but the principle of "remove
  what doesn't help" applies. Vaswani used bias=True; the original choice
  wasn't ablated, and later work removed it without quality loss.

- **What does `.contiguous()` do and why is it needed before `view`?**
  PyTorch tensors have a `data` buffer and a `stride` tuple that maps
  multi-dim indices to flat-buffer offsets. Most ops produce strides that
  match a row-major layout (contiguous). `transpose` swaps two strides
  without copying memory, leaving the tensor non-contiguous. `view`
  requires contiguity because it reinterprets the underlying flat memory
  with new strides — it can't reorder bytes. `.contiguous()` copies the
  tensor into a new buffer with row-major layout. Skipping it on a
  post-transpose tensor raises `view size is not compatible with input
  tensor's size and stride`.

- **What's the FLOPs breakdown per attention call?**
  Three projections: 3 × B × T × C² ≈ 3BTC². Scores matmul:
  B × H × T² × D = BT²C. Attn-V matmul: same, BT²C. Output projection:
  BTC². Total: 4BTC² + 2BT²C. The T² terms dominate at long context —
  which is the other reason Flash Attention matters.

### Test coverage

`tests/test_layers.py` (8 cases):
- `test_mha_output_shape_matches_input` — `(B, T, C)` in, `(B, T, C)` out.
- `test_mha_rejects_non_divisible_dims` — `ValueError` on
  `embed_dim % num_heads != 0`.
- `test_mha_gradient_flows_to_all_params` — every named parameter gets a
  finite gradient.
- **`test_mha_causal_mask_blocks_future_information`** — the load-bearing
  correctness test. Perturbing position 1's input must NOT change
  position 0's output (it's masked from seeing it), but position 1's
  output WILL change. Catches: inverted mask convention, missing mask
  broadcast, mask dtype/shape mismatches. If this passes, the mask plumbing
  is correct.
- `test_mha_no_mask_mixes_all_positions` — without a mask, position 0
  responds to perturbations at position 3. Confirms attention actually
  attends (not just an MLP in disguise).
- `test_mha_param_count_bias_off_vs_on` — `bias=True` adds exactly
  `4 * embed_dim` params (4 linears, embed_dim bias each).
- `test_mha_dropout_is_noop_in_eval_mode` — `nn.Dropout` is automatically
  identity in `eval()`, two forwards give identical outputs.
- `test_mha_calls_rotary_hook_if_provided` — a no-op rotary gives output
  identical to no rotary; a scaling rotary still produces correct-shape
  output. Verifies the hook is actually wired without testing RoPE's
  semantics (that's RoPE's test).

### Common bugs this implementation avoids

- **Inverted mask convention.** `causal_mask` returns True-means-attend,
  but `masked_fill` writes where the predicate is True — these are opposite,
  so `~mask` is required. Forgetting this would mask exactly the positions
  you wanted to keep. Caught by `test_mha_causal_mask_blocks_future_information`.
- **Missing `.contiguous()` after transpose.** Would error out on the `view`
  call. Caught by every shape test.
- **Wrong axis for softmax.** Caught by `test_mha_no_mask_mixes_all_positions`
  (wrong axis would still produce shape-correct output but with broken
  attention semantics).
- **head_dim not stored.** Would force re-derivation in forward; not a
  correctness bug but a code-smell. Stored in `__init__`.

### Why no integration with model.py yet

MHA is constructed by `TransformerBlock`, which is still a stub. The
`DummyBlock` in `tests/_dummies.py` replaces TransformerBlock entirely
during model.py tests, so the real MHA isn't exercised through the model
yet. That happens in step 6 (TransformerBlock implementation).

---

## 4. `GeluFFN(d_model, d_ffn, bias, dropout)`

**File:** `layers.py:165-184`
**Tests:** `tests/test_layers.py:294-320` (4 cases)
**Patch status:** N/A — wasn't stubbed; new class added.

### Design

Classic two-matrix feed-forward: `y = down_proj(GELU(up_proj(x)))`. With
`d_ffn = 4 * d_model` (Vaswani convention, set as the GELU default in
`ModelConfig.__post_init__`), this layer carries 8 · d_model² params —
roughly 2× the attention block's param count, and the dominant chunk of
each transformer block's representational capacity.

**Naming is load-bearing.** `model.py:_init_weights` applies GPT-2's
residual init scaling (`std = 0.02 / sqrt(2*n_layer)`) to any parameter
whose name ends in `out_proj.weight` or `down_proj.weight`. Calling the
second linear `down_proj` is how GeluFFN opts in. Renaming it would
silently disable the scaling, hurting training stability at depth.

### Walkthrough

```python
def __init__(self, d_model, d_ffn, bias=False, dropout=0.0):
    super().__init__()
    self.up_proj = nn.Linear(d_model, d_ffn, bias=bias)
    self.down_proj = nn.Linear(d_ffn, d_model, bias=bias)
    self.dropout = nn.Dropout(dropout)

def forward(self, x):
    return self.dropout(self.down_proj(F.gelu(self.up_proj(x))))
```

Two linears with a GELU between. Dropout *after* the down-projection,
before the residual add in TransformerBlock. Standard pattern.

### Interview Qs

- **Why is FFN dim 4×d_model?** Empirical from Vaswani 2017. Sweeps show a
  broad optimum around 4×. Smaller hurts capacity; larger gives diminishing
  returns and bloats VRAM. Modern frontier models use 4× (or its PaLM-style
  equivalent `(8/3)×` for SwiGLU).
- **Why GELU instead of ReLU?** Smoother around 0; small negative inputs
  pass through with attenuation rather than being fully zeroed. Empirically
  ~0.2-0.5 PPL improvement over ReLU on transformer LMs. GPT-2, BERT,
  GPT-3 all use GELU.
- **Why no normalization inside the FFN?** Pre-norm in TransformerBlock
  normalizes the *input* to the FFN. A norm after the second linear
  would double-normalize and tends to hurt — the residual stream wants to
  be the only normalized object per sublayer.

### Test coverage

`tests/test_layers.py` (4 cases):
- `test_gelu_ffn_preserves_outer_shape` — input/output shape match.
- `test_gelu_ffn_param_count` — exactly `2 * d_model * d_ffn` params at bias=False.
- `test_gelu_ffn_has_down_proj_name` — regression on the residual-init contract.
- `test_gelu_ffn_gradient_flows` — grads to input + every param, finite.

---

## 5. `SwiGLUFFN(d_model, d_ffn, bias, dropout)`

**File:** `layers.py:187-211`
**Tests:** `tests/test_layers.py:325-371` (5 cases)
**Patch status:** N/A — new class.

### Design

Gated linear unit with SiLU activation. Three matrices instead of two:

```
y = down_proj( silu(gate_proj(x)) * up_proj(x) )
```

The element-wise product `silu(gate_proj(x)) * up_proj(x)` is the gate.
For each output channel, the gate (a learned function of x) decides how
much of `up_proj(x)` to let through. Two-matrix FFNs can't represent this
kind of multiplicative interaction.

**Why three matrices but matched param count?** Two matrices of shape
`C × 4C` = `8C²` params. Three matrices of shape `C × F` = `3CF`. Setting
`3CF = 8C²` gives `F = (8/3) · C`. The project's `ModelConfig.__post_init__`
does this calculation and rounds to a multiple of 64 (for GPU alignment).
The point is the *controlled comparison* in the ablation: GELU vs SwiGLU
at matched params — any quality difference is attributable to the
architecture choice, not param count.

**Naming:** `gate_proj`, `up_proj`, `down_proj` follow LLaMA convention.
`down_proj.weight` triggers the residual scaling in model.py (same
contract as GeluFFN).

### Walkthrough

```python
def __init__(self, d_model, d_ffn, bias=False, dropout=0.0):
    super().__init__()
    self.gate_proj = nn.Linear(d_model, d_ffn, bias=bias)
    self.up_proj = nn.Linear(d_model, d_ffn, bias=bias)
    self.down_proj = nn.Linear(d_ffn, d_model, bias=bias)
    self.dropout = nn.Dropout(dropout)

def forward(self, x):
    return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))
```

`F.silu(z) = z * sigmoid(z)`. Smooth, non-monotonic (slight bump in
negative region), and gradient-friendly near zero. Has been called Swish-1
elsewhere.

### Interview Qs

- **Why three matrices?** Multiplicative gating. The element-wise product
  lets the model selectively pass or attenuate channels based on a learned
  gate that's a function of the same input. Empirically (Shazeer 2020,
  PaLM, LLaMA), this outperforms plain two-matrix FFNs at matched params.
- **Why SiLU and not GELU inside the gate?** Empirical. Shazeer ablated
  GLU variants (GeGLU, SwiGLU, ReGLU) and SwiGLU won at matched-params on
  T5 perplexity. Practical reason: SiLU is cheaper than GELU (no `erf` or
  tanh approximation needed).
- **Why `(8/3) * d_model`?** Param-matching. Two matrices of C×4C =
  8C² params. Three matrices of C×F = 3CF. Set equal → F = (8/3)C. The
  rounding-to-64 is for tensor-core alignment, costs ~1% in param count.
- **Where's the activation on `up_proj`?** There isn't one. `up_proj(x)`
  passes through linearly; the only nonlinearity is `silu(gate_proj(x))`.
  This is the GLU structure — one branch activated, one branch linear,
  multiplied. (If both branches were activated you'd have a different
  architecture, e.g., "double-SwiGLU.")

### Test coverage

`tests/test_layers.py` (5 cases):
- `test_swiglu_ffn_preserves_outer_shape` — shape preserved.
- `test_swiglu_ffn_param_count` — exactly `3 * d_model * d_ffn` params at bias=False.
- `test_swiglu_param_match_to_gelu_via_palm_sizing` — at d_model=192,
  PaLM-sized SwiGLU has within 5% of GELU(4*d_model)'s param count.
  Regression on the matched-comparison invariant.
- `test_swiglu_ffn_has_three_named_projections` — `gate_proj`, `up_proj`,
  `down_proj` names present. Regression on the model.py init contract
  and the LLaMA naming convention.
- `test_swiglu_ffn_gradient_flows` — grads to input + every param, finite.

---

<!-- Next entries (RoPE, TransformerBlock) appended as components land. -->

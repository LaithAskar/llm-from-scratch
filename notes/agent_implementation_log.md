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

## 6. `RotaryEmbedding(head_dim, max_seq_len, base)` + `_apply_rotary`

**File:** `layers.py:165-228`
**Tests:** `tests/test_layers.py:376-490` (8 cases)
**Patch status:** N/A — new class.

### Design

Su et al. 2021. Instead of adding a position vector to the token embedding,
*rotate* each Q and K vector by a position-dependent angle. The dot product
of two rotated vectors depends only on their *relative* offset:

```
dot( R(mθ)·q, R(nθ)·k ) = q · R((n-m)θ) · k
```

(Rotations compose through dot products as the difference of their angles.)
This means `score[m, n]` only depends on `m - n`, not on `m` and `n`
individually — an inductive bias toward relative position that learned-
absolute pos doesn't have.

**Half-split convention** (LLaMA / Hugging Face). For head_dim `D`, pair
element `i` with element `i + D/2`. The first half is "real components,"
the second half is "imaginary components." Each pair is rotated by
`m * inv_freq[i]` where:

```
inv_freq[i] = 1 / base^(2i / D)        for i = 0, ..., D/2 - 1
```

`base = 10000` is inherited from Transformer's sinusoidal positional
encoding. Higher `base` → longer wavelengths in the frequency tail →
better extrapolation to unseen long contexts. LLaMA-2 raised it to 500000
for the same reason.

**Alternative: interleaved convention.** Some implementations pair adjacent
elements `(q[0], q[1]), (q[2], q[3]), ...` instead of half-split. Equivalent
math, different memory access pattern. Half-split is one slice per half
(contiguous), interleaved needs strided indexing. Half-split is slightly
faster on most GPUs. **Both conventions need Q and K to agree** — if MHA's
Q used half-split and K used interleaved, the rotations would be incompatible.

### Walkthrough

```python
def __init__(self, head_dim, max_seq_len, base=10000.0):
    super().__init__()
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim ({head_dim}) must be even for RoPE")
    self.head_dim = head_dim
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    positions = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)
    self.register_buffer("cos_cached", freqs.cos(), persistent=False)
    self.register_buffer("sin_cached", freqs.sin(), persistent=False)
```

Precompute everything in `__init__`:

1. **Validate even head_dim.** Pairs require it.
2. **Compute inv_freqs** of shape `(D/2,)`. The exponent `2i/D` over `i = 0,
   ..., D/2-1` means `i=0` gives `inv_freq = 1` (high frequency, fast
   rotation per position) and `i = D/2-1` gives `~1/base` (low frequency,
   slow rotation per position). The geometric spread across pair indices
   is what gives RoPE its multi-scale position encoding.
3. **Outer product positions × inv_freqs.** Result shape `(max_seq_len, D/2)`.
   Element `(m, i)` is `m * inv_freq[i]` — the rotation angle for position
   `m` at pair `i`.
4. **Register cos/sin as non-persistent buffers.** Non-persistent means they
   don't go into the state_dict — every load regenerates them. This avoids
   bloating checkpoints with derivable data. They still move with
   `.to(device)` because they're registered buffers.

```python
def forward(self, q, k):
    T = q.size(-2)
    if T > self.cos_cached.size(0):
        raise ValueError(...)
    cos = self.cos_cached[:T]
    sin = self.sin_cached[:T]
    return _apply_rotary(q, cos, sin), _apply_rotary(k, cos, sin)
```

Slice the cache to actual sequence length, apply rotation to both Q and K,
return. V is NOT rotated — content vectors shouldn't be position-dependent;
only the attention weights (computed from Q·K^T) should be.

### `_apply_rotary` walkthrough

```python
def _apply_rotary(x, cos, sin):
    D = x.size(-1)
    half = D // 2
    x1 = x[..., :half]                       # (..., D/2)
    x2 = x[..., half:]                       # (..., D/2)
    cos_x = cos.to(x.dtype)
    sin_x = sin.to(x.dtype)
    out1 = x1 * cos_x - x2 * sin_x
    out2 = x1 * sin_x + x2 * cos_x
    return torch.cat((out1, out2), dim=-1)
```

The half-split rotation. Each pair `(x1[..., i], x2[..., i])` becomes
`(x1*cos - x2*sin, x1*sin + x2*cos)` — that's a 2D rotation matrix applied
to each pair.

**Dtype cast.** The cache is fp32 (for precision when generating cos/sin).
Without `cos.to(x.dtype)`, multiplying `x: bf16` by `cos: fp32` promotes
the whole result to fp32, breaking AMP. Casting cos/sin to x's dtype keeps
the computation in the model's working dtype.

**Broadcasting.** `cos` and `sin` are shape `(T, D/2)`. `x1, x2` are shape
`(B, H, T, D/2)`. PyTorch right-aligns dims: `(T, D/2)` broadcasts as
`(1, 1, T, D/2)` against `(B, H, T, D/2)`. The same cos/sin applies to
every batch and every head — exactly what we want, since RoPE is purely
positional.

### Interview Qs

- **Why does RoPE give *relative* positions?**
  Inner product: `(R(mθ)·q) · (R(nθ)·k) = q · R(-mθ)·R(nθ) · k = q · R((n-m)θ) · k`.
  Rotations commute through the inner product as a difference. The score
  depends only on `n - m`.

- **Why even head_dim?**
  RoPE pairs elements. Odd dim has a leftover element with no pair partner.

- **Why `base = 10000`?**
  Inherited from the original Transformer's sinusoidal positional encoding,
  where it was chosen so wavelengths span from 2π to 10000·2π — covering
  position-scale variation from token-level to sentence-level. RoPE
  recycles the same convention. Higher base → longer wavelengths in the
  tail → smaller per-position rotation at low-frequency pairs → better
  long-context behavior. LLaMA-2 raised base to 500000 to extend the
  context window without retraining the rotation cache.

- **Why precompute and cache cos/sin?**
  Computing `sin(m * inv_freq)` per forward would be wasteful. They depend
  only on position and frequency, not on the input — compute once at
  `__init__`. Non-persistent buffer means they regenerate on load instead
  of inflating checkpoint size by `2 * max_seq_len * head_dim/2 * 4 bytes`.

- **Why fp32 for the cache, then cast to input dtype at use?**
  Precision: at large positions (`m = 2000+`), `m * inv_freq` accumulates
  enough that bf16's small mantissa starts losing digits in the
  trigonometry. Computing in fp32 then casting at the use site gets the
  best of both — precise angles, dtype-consistent multiplications.

- **Why is V not rotated?**
  V is the value being mixed; rotating it would entangle position with
  content. The desired effect — making attention scores position-aware —
  is achieved entirely by rotating Q and K. After softmax, the weighted
  sum of (un-rotated) Vs gives the right answer.

- **Half-split vs interleaved — does the math change?**
  No, the rotation is the same set of 2D rotations applied to the same
  pairs of elements. Only the indexing convention differs (which element
  is "the partner" of element `i`). Hugging Face and LLaMA both use
  half-split; some older implementations use interleaved. The two are
  not interoperable at the *weights* level (a model trained with one
  convention can't load weights from the other without re-ordering), but
  both produce valid RoPE.

### Test coverage

`tests/test_layers.py` (8 cases):
- `test_rope_rejects_odd_head_dim` — ValueError on odd dim.
- `test_rope_output_shape_matches_input` — shape preserved.
- `test_rope_is_identity_at_position_zero` — `cos(0)=1, sin(0)=0`,
  so position 0's rotation is identity. Sanity floor.
- **`test_rope_relative_position_invariance`** — the load-bearing test.
  Fixed (q, k) at offset (0,1) and at offset (5,6) give identical scores.
  Catches: half-split inversion, cos/sin swap, wrong pair indexing,
  forgetting the negative sign on the sin term, broadcasting bugs.
  If this passes, the rotation math is correct.
- `test_rope_different_offsets_give_different_scores` — inverse sanity:
  different relative offsets give different scores (else RoPE is doing
  nothing).
- `test_rope_rejects_overlong_sequence` — clear error message instead of
  silent indexing OOB.
- `test_rope_gradient_flows` — q and k get gradient. (RoPE has no params,
  so we just check the call doesn't break autograd.)
- `test_rope_integrates_with_mha` — end-to-end: hand MHA a RotaryEmbedding
  via constructor, run forward, output shape matches. Confirms the
  callable contract.

### Why no parameters?

RoPE is a pure function of position and pair index. All angles are
deterministic; nothing learnable. This is a strong inductive bias and a
small param savings over learned-absolute pos (which adds `context_len *
d_model` params). Empirically the bias is helpful — RoPE outperforms
learned-absolute on most benchmarks even at matched effective param count.

---

## 7. `TransformerBlock(config)`

**File:** `layers.py:281-340`
**Tests:** `tests/test_layers.py:495-595` (10 cases)
**Patch status:** real implementation; **`DummyBlock` patch fully removed**
from `conftest.py`. All 28 model/train/eval/ablation tests now exercise the
real layers end-to-end.

### Design

The plumbing block. Wires norm + attn + norm + ffn into the standard
pre-norm pattern, with all architectural switches read from `config`:

- `config.norm_type`: `LayerNorm` or `RMSNorm` for both norms.
- `config.activation`: `GeluFFN` or `SwiGLUFFN` for the FFN.
- `config.pos_encoding`: `RotaryEmbedding` constructed and handed to MHA
  via the `rotary=` hook (vs `None` for learned-absolute, handled in
  `model.py` at the embedding level).

This is exactly the layer of indirection the ablation needs: flip one
config field, swap one component, every other variable held constant.

### Walkthrough

```python
def __init__(self, config):
    super().__init__()

    def make_norm():
        if config.norm_type == "rmsnorm":
            return RMSNorm(config.d_model, eps=config.norm_eps)
        return nn.LayerNorm(config.d_model, eps=config.norm_eps, bias=config.bias)

    self.norm1 = make_norm()
    self.norm2 = make_norm()
```

A nested `make_norm` closure builds the right norm type — used twice (once
before attention, once before the FFN). Could have been a free function;
nested gives it private scope and prevents accidental reuse outside the
block. `nn.LayerNorm`'s `bias` argument was added in PyTorch 2.1.

```python
    rotary = None
    if config.pos_encoding == "rope":
        rotary = RotaryEmbedding(
            head_dim=config.head_dim,
            max_seq_len=config.context_len,
            base=config.rope_base,
        )
```

Construct RoPE only if needed. **Key choice:** `max_seq_len = config.context_len`,
not larger. RoPE's cache size is `O(max_seq_len * head_dim)` — sizing it
beyond what the model can process is wasted memory. `context_len` is the
right ceiling.

```python
    self.attn = MultiHeadAttention(
        embed_dim=config.d_model,
        num_heads=config.n_head,
        dropout=config.dropout,
        bias=config.bias,
        rotary=rotary,
    )
```

Build the attention sublayer with all switches plumbed through.

```python
    if config.activation == "swiglu":
        self.ffn = SwiGLUFFN(
            d_model=config.d_model, d_ffn=config.d_ffn,
            bias=config.bias, dropout=config.dropout,
        )
    else:
        self.ffn = GeluFFN(
            d_model=config.d_model, d_ffn=config.d_ffn,
            bias=config.bias, dropout=config.dropout,
        )
```

FFN picker. `config.d_ffn` was resolved at config-construction time —
`(8/3) * d_model` for SwiGLU, `4 * d_model` for GELU. Don't recompute here.

```python
def forward(self, x, mask=None):
    h = x + self.attn(self.norm1(x), mask=mask)
    h = h + self.ffn(self.norm2(h))
    return h
```

Pre-norm residual pattern. Two lines, the whole architecture in two
expressions. Read literally:

1. Normalize x, run attention on the normalized version, add to original x.
   Norm sits *inside* the sublayer branch; residual stream stays clean.
2. Same pattern for FFN: normalize, transform, add.

Mask passed through to attention; FFN doesn't need it (FFNs operate
per-position).

### Interview Qs

- **Why pre-norm beats post-norm at depth.**
  Post-norm sits *on the residual stream itself*: `y = LayerNorm(x +
  sublayer(x))`. Gradients flowing back have to pass through every norm,
  which controls the magnitude of the backward signal. At depth this
  requires careful warmup (Vaswani used 4000 steps for a 6-layer model).
  Pre-norm: `y = x + sublayer(LayerNorm(x))`. The residual stream is a
  pure identity path; norms only affect the sublayer branch. Gradient
  flow is exponentially better. Training is stable at 100+ layers without
  warmup gymnastics. See Xiong et al. 2020 for the spectral analysis.

- **Why two norms per block, not one or three?**
  One norm per sublayer. Two sublayers (attention, FFN) ⇒ two norms.
  Each sublayer's input distribution needs stabilization independently
  — attention output has different scale and skew than FFN input. NormFormer
  proposes a third norm post-FFN with marginal gains; not standard.

- **Why is the norm picker a closure, not a class-level method?**
  Used twice, only inside `__init__`. Closure keeps it local — no
  pollution of the class API, no need to remember to pass `self.config`
  to it. Same code without the closure would be a repeated `if` block
  for norm1 and norm2. (Pythonically equivalent; this is style.)

- **Why does the block construct RoPE itself instead of receiving it from
  the LM?**
  Each block needs its own RoPE module so the buffer lives on the same
  device as the rest of the block's parameters (moved via
  `model.to(device)`). Could share one RoPE across blocks at the LM
  level — same cache values — but the bookkeeping cost is higher than
  the duplication cost: cache is only `O(context_len * head_dim / 2 *
  4 bytes)` per block ≈ a few KB.

- **Why does the FFN dropout default to `config.dropout` here but the
  attention also gets `config.dropout`?**
  In this project, one dropout knob controls all. nanoGPT uses the same
  convention. Some impls (e.g., HF's GPT-2) separate `attn_pdrop` and
  `resid_pdrop`; that's overfitting the API for marginal ablation value.

- **Why doesn't `forward` do anything with the mask in the FFN step?**
  FFN is per-position — each token's FFN output depends only on its own
  input. No cross-position interaction, so no mask needed. Only the
  attention sublayer mixes information across positions, and that's where
  the mask is applied.

### Test coverage

`tests/test_layers.py` (10 cases):
- `test_block_forward_shape_preserved` — `(B, T, C)` in/out.
- `test_block_norm_switch_layernorm` / `..._rmsnorm` — config drives norm
  type; both norm1 and norm2 are the same type.
- `test_block_activation_switch_gelu` / `..._swiglu` — config drives FFN
  type.
- `test_block_rope_switch_on` — `pos_encoding="rope"` puts a
  `RotaryEmbedding` on the MHA's `rotary` slot.
- `test_block_rope_switch_off` — `pos_encoding="learned"` keeps `rotary=None`.
- `test_block_modern_variant_all_switches` — the most-complex ablation
  variant ("modern": rmsnorm + swiglu + rope) constructs and forwards
  end-to-end.
- `test_block_gradient_flows_to_input_and_all_params` — gradient flow,
  finite gradients on every named param.
- `test_block_residual_pattern_attn_then_ffn` — parameter accounting:
  all of the block's params are in `{norm1, norm2, attn, ffn}`. Regression
  if someone accidentally adds a stray learnable parameter to the block.

### Indirect coverage via `tests/test_model.py`, `test_train.py`, etc.

Removing the `DummyBlock` patch from `conftest.py` means the existing 28
infrastructure tests now exercise *real* layers:
- `test_forward_with_targets_returns_loss` — real model output now goes
  through real attention + real FFN; CE loss is real, ~ln(vocab_size) at
  init as the test expected.
- `test_grad_flows_to_all_params` — gradient flows through real attention,
  real norms, real FFN.
- `test_train_runs_end_to_end` — 3 real training steps with real layers.
- `test_run_ablation_smoke` — 2 variants × 1 seed × 3 steps with real
  layers, including real RMSNorm in the rmsnorm variant.

All pass on first attempt — the layers' shapes, gradients, init, and
config-switching are correct as observed by the model/train/eval/ablation
stack.

### What's next (out of layers.py scope)

- `python data.py prepare` to download TinyStories (~2 GB).
- `python ablation.py --max-steps 500 --variants baseline rmsnorm rope swiglu modern --seeds 0`
  for a short smoke run that produces a real `summary.csv`.
- Plotting / analysis script over `summary.csv` once seeds finish.
- README with honest authorship attribution (separately tracked).

---

## 8. KV cache (Part 3* — inference optimization)

**Files modified:**
- `layers.py:RotaryEmbedding.forward` — added `position_offset` arg.
- `layers.py:MultiHeadAttention.forward` — added optional `kv_cache` dict;
  conditional tuple return.
- `layers.py:TransformerBlock.forward` — added optional `kv_cache`; passes through.
- `model.py:TransformerLM.generate` — rewritten as prefill-then-decode with
  `use_cache=True` (default). `use_cache=False` keeps the recompute path
  for correctness testing.
- `model.py:_sample_next` — new module-level helper used by both generate paths.

**Tests:** `tests/test_layers.py` (+7 cases), `tests/test_model.py` (+2 cases).
Total 85 passing.

### Design

KV cache is a *generation-time* optimization. During autoregressive
sampling, every new token needs to attend to all previous K and V. Without
caching, each step recomputes K and V from scratch for the entire history
— O(T²) attention FLOPs across T new tokens. With caching, we store K and
V from past steps and only project the new token — O(T) per step.

For 100 generated tokens at context_len=128: ~50× fewer attention FLOPs.

**Why it's training-neutral.** The cache only activates when `kv_cache` is
passed to MHA/Block. Default is `None`, so training (which calls `forward`
without cache) is unchanged. Same code path, same gradients, same losses.

### API choices

1. **Conditional return type on MHA/Block.** When `kv_cache=None`, return
   just the output tensor (existing API preserved). When `kv_cache` is a
   dict, return `(out, updated_cache)`. The alternative — always returning
   a tuple — would churn every existing caller; the conditional shape is
   documented in the docstring.

2. **Empty `{}` means "prefill mode."** Caller passes `kv_cache={}` for the
   first call (prompt processing). MHA distinguishes "fresh cache" vs
   "populated cache" by checking `"k" in kv_cache`.

3. **Store ALREADY-ROTATED K in the cache.** When RoPE is active, K is
   rotated by its position angle before going into the cache. On the next
   step, the cached K isn't re-rotated — it's already correctly positioned.
   New K gets rotated using `position_offset=cache_length`.

4. **Hard-stop at `context_len` instead of sliding the window.** When the
   cache fills to `context_len`, `generate` returns the tokens it has
   rather than evicting old cache entries. Sliding would break
   learned-absolute position embeddings (positions would exceed
   `context_len`) and RoPE (cached angles would be wrong post-eviction).
   Sliding-window generation is a real technique but adds complexity
   disproportionate to a toy LLM.

### Walkthrough — `MultiHeadAttention.forward` with cache

```python
pos_offset = 0
if kv_cache is not None and "k" in kv_cache:
    pos_offset = kv_cache["k"].size(-2)

if self.rotary is not None:
    q, k = self.rotary(q, k, position_offset=pos_offset)
```

Determine where the new tokens sit in absolute position space. On first
call (empty cache), they start at 0. On the Nth decode step (cache has N
already-rotated tokens), they start at N.

```python
if kv_cache is not None:
    if "k" in kv_cache:
        k = torch.cat([kv_cache["k"], k], dim=-2)
        v = torch.cat([kv_cache["v"], v], dim=-2)
    new_kv_cache = {"k": k, "v": v}
```

Concat new K/V onto history. The score matmul then has shape
`(B, H, T_new, T_total)` because k has T_total tokens but q only has T_new.

### Walkthrough — `TransformerLM._generate_cached`

```python
caches: list[dict] = [{} for _ in self.blocks]

# Prefill
h = self.tok_emb(idx) + (pos_emb if learned else 0)
for i, block in enumerate(self.blocks):
    h, caches[i] = block(h, mask=mask, kv_cache=caches[i])
next_id = sample(self.final_norm(h[:, -1:, :]))

# Decode loop
for _ in range(max_new_tokens - 1):
    if caches[0]["k"].size(-2) >= context_len: break
    new_x = self.tok_emb(next_id) + pos_emb_at_current_length
    for i, block in enumerate(self.blocks):
        new_x, caches[i] = block(new_x, mask=None, kv_cache=caches[i])
    next_id = sample(self.final_norm(new_x))
```

Two key tricks:
- **Sample first token from prefill's last position.** Prefill computes
  hidden states for every prompt position; we only need the last to
  predict the next token.
- **No mask during decode.** Single new token (T_q=1) attends to all of
  history; masking would be a no-op anyway.

### Interview Qs you should be able to answer

- **Why is KV cache only useful at generation time, not training?**
  Training does a single parallel forward over the whole sequence — each
  position's K, V is computed once and used once, in parallel. There's no
  reuse across steps to cache. Generation is autoregressive: every new
  step re-attends to all previous K and V; without cache, those get
  re-projected from the embedding stream each time.

- **Why don't we cache the queries?**
  Q for position N is used only at the attention step at position N. K
  and V from position N are read by every step after N. Cache is only
  useful for tensors that are *read across multiple decoder steps.*

- **What's the memory cost of the KV cache?**
  Per layer: `2 (K and V) * B * H * T * D * bytes`. Toy scale
  (`B=1, H=4, T=128, D=32, bf16`): 64 KB per layer per batch — negligible.
  GPT-3 scale (`H=96, T=2048, D=128, n_layer=96, bf16`): ~9 GB per batch.
  This is why long-context inference is memory-bound, not compute-bound.

- **Why does the cached path's first call need a causal mask but subsequent
  calls don't?**
  Prefill processes the full prompt in one forward — all prompt positions
  exist in the same q tensor, so the causal mask prevents each from
  peeking at later prompt positions. Decode calls have only one new
  position; T_q=1 against T_kv=T_total broadcasts to "attend to
  everything," which is correct because everything in the cache is "the
  past" from the new token's perspective.

- **What does sliding-window or rolling KV cache do, and why didn't we
  implement it?**
  Slide the cache: evict oldest entries when adding new ones, so
  generation can continue past context_len indefinitely. Used in streaming
  chat. Not done here because (a) it adds significant complexity
  (eviction logic, position-index translation for learned-abs pos, RoPE
  cache index translation), (b) for a toy LLM the test cases don't need
  it, and (c) the alternative — "you ran out of context, stop" — is
  honest and explicit.

### Test coverage

`tests/test_layers.py` (+7):
- `test_rope_position_offset_equals_absolute_position`
- **`test_mha_cached_prefill_plus_decode_equals_full`** — bit-equality
- `test_mha_cached_decode_step_by_step_equals_full`
- `test_mha_cached_with_rope_equals_full`
- `test_block_cached_equals_full`
- `test_block_cached_with_rope_equals_full`
- `test_block_cached_with_rmsnorm_swiglu_modern_stack_equals_full`

`tests/test_model.py` (+2):
- **`test_generate_cached_matches_naive_path`** — `generate(use_cache=True)`
  and `generate(use_cache=False)` produce bit-identical token sequences
  given the same RNG seed. End-to-end correctness gate.
- `test_generate_cached_matches_naive_with_rope` — same with modern stack.

### Side-effect on prior test

`test_generate_stops_at_context_limit` (formerly
`test_generate_truncates_long_context`) was updated. The old naive path
silently slid a context window and kept appending; the new cached path
hard-stops when the cache fills. Honest API > quietly-degrading API.

---

## 9. BPE training (Part 4)

**File:** `bpe.py` (new module)
**Tests:** `tests/test_bpe.py` (15 cases)

### Design

Standard byte-pair encoding (Sennrich et al. 2015). The algorithm:

1. **Pre-tokenize** text into "words" using GPT-2's regex (which keeps
   leading whitespace attached to the following word, separates punctuation
   from letters, etc.). This prevents merges from crossing natural word
   boundaries.
2. **Initialize** vocab with all 256 single-byte tokens.
3. **Repeat** until vocab_size is reached:
   - Count adjacent token-pair frequencies across all words (weighted by
     word frequency).
   - Find the most frequent pair.
   - Add the merged token to the vocab, record the merge in the merge list.
   - Replace every occurrence in every word.

The ablation matrix still uses tiktoken's GPT-2 BPE so vocab is held
constant across variants. This module is a **standalone artifact**
demonstrating the algorithm, not a drop-in replacement for tiktoken.

### Pre-tokenization regex (GPT-2 convention)

```
'(?:[sdmt]|ll|ve|re) | ?\p{L}+ | ?\p{N}+ | ?[^\s\p{L}\p{N}]+ | \s+(?!\S) | \s+
```

What each branch matches:
- `'(?:[sdmt]|ll|ve|re)` — English contractions ('s, 'd, 'm, 't, 'll, 've, 're)
- ` ?\p{L}+` — letters with optional leading space (so " hello" is one token)
- ` ?\p{N}+` — numbers with optional leading space (" 2026")
- ` ?[^\s\p{L}\p{N}]+` — punctuation runs with optional leading space
- `\s+(?!\S)` — trailing whitespace (used to keep final spaces from being orphaned)
- `\s+` — interior whitespace runs

Requires the `regex` package (stdlib `re` doesn't support `\p{L}` /
`\p{N}` as of Python 3.13).

### Encoding strategy

Given a trained merge list, encoding "hello" works by:
1. Pre-tokenize into pre-tokens (here: `b"hello"`).
2. Split each pre-token into single bytes (`[b"h", b"e", b"l", b"l", b"o"]`).
3. Repeatedly find the pair with the **lowest merge index** (= the
   earliest-learned merge) and apply it. Repeat until no learned merges
   remain.
4. Look up each remaining token in the vocab to get its integer ID.

The "lowest-merge-first" priority matches tiktoken and Hugging Face's
GPT-2 BPE. Alternative strategies (highest-frequency-first, greedy-longest
match) exist but produce different encodings — same vocab, different IDs
for the same text. This implementation uses GPT-2's convention.

### Walkthrough — `train_bpe`

```python
words = pretokenize(corpus)
word_freqs = collections.Counter(words)
word_freqs_split = {
    tuple(bytes([b]) for b in word): freq
    for word, freq in word_freqs.items()
}
vocab = {bytes([i]): i for i in range(256)}
merges = []
```

Initialize: count word frequencies, split each word into byte-tokens, set
up the initial 256-byte vocab.

```python
for step in range(vocab_size - 256):
    pair_counts = _count_pairs(word_freqs_split)
    if not pair_counts: break
    best_pair, best_count = max(pair_counts.items(), key=lambda kv: kv[1])
    merged = best_pair[0] + best_pair[1]
    vocab[merged] = len(vocab)
    merges.append(best_pair)
    word_freqs_split = {
        _apply_merge(word, best_pair): freq
        for word, freq in word_freqs_split.items()
    }
```

The merge loop. Each iteration: count pairs across all words, pick the
most-frequent, add the merged token (with the next-available integer ID)
to the vocab, record the merge, apply it to every word.

**Efficiency note.** This implementation re-counts all pairs from scratch
on every iteration — O(N * V) where N is total tokens and V is merges.
A production impl maintains an incremental pair-count table and only
updates around the affected positions — O(K) per merge where K is the
number of occurrences of the merged pair. For toy-scale corpora (< 100 MB)
the simple approach is fine and ~5x easier to verify by hand.

### Walkthrough — `encode`

```python
merge_rank = {pair: i for i, pair in enumerate(merges)}
for word in pretokenize(text):
    tokens = [bytes([b]) for b in word]
    while len(tokens) > 1:
        best_rank = len(merges); best_pos = -1
        for i in range(len(tokens) - 1):
            rank = merge_rank.get((tokens[i], tokens[i + 1]), len(merges))
            if rank < best_rank:
                best_rank = rank
                best_pos = i
        if best_pos < 0: break
        tokens = tokens[:best_pos] + [tokens[best_pos] + tokens[best_pos+1]] + tokens[best_pos+2:]
    out.extend(vocab[t] for t in tokens)
```

Greedy lowest-merge-rank application. The inner loop scans all adjacent
pairs and picks the one with the smallest merge index. Apply it; repeat
until no pair is in `merge_rank`. Then look up each remaining token's ID.

The `merge_rank.get(pair, len(merges))` trick treats unknown pairs as
"infinity rank" — they're never picked because any known pair has rank
< len(merges).

### Interview Qs you should be able to answer

- **Why pre-tokenize before BPE?**
  Without pre-tokenization, BPE could merge across word boundaries —
  things like "the dog" → "thedog" as a single token. Pre-tokenization
  caps merges at the word level, which both (a) gives more interpretable
  tokens and (b) prevents combinatorial explosion in the merge space.
  GPT-2's regex is empirically tuned to give good English-language
  coverage.

- **Why bytes (256 base vocab) instead of unicode characters?**
  Unicode-character base would mean ~1M code points → huge initial vocab.
  Most LLMs never see most code points. Byte-level (256 fixed initial
  vocab) is universal — every text is encodable, BPE figures out which
  byte sequences are worth merging based on the actual training corpus.

- **What's the difference between BPE and WordPiece (BERT) or Unigram (T5)?**
  BPE: greedy merge of most-frequent pair. Deterministic given the
  corpus.
  WordPiece: BPE variant that picks the merge maximizing likelihood of
  the corpus under the resulting tokenization (slightly different
  objective; usually similar results).
  Unigram (SentencePiece): start with a large vocab, prune least-useful
  tokens iteratively until target size. Better at handling languages
  without natural word boundaries.

- **Why "lowest merge index" for encoding, not "highest frequency"?**
  Because the merge order *records the algorithm's own training-time
  decisions*. If at training step 3 the algorithm chose to merge ('e',
  'd'), it did so because at that point in the iterative merging, 'ed'
  was the right choice — even if later merges made other pairs more
  frequent. Replaying the same priority order at encode time gives
  consistent tokenization with what the algorithm would produce if you
  re-trained on the same text.

- **What's the compression ratio you'd expect on English?**
  GPT-2 BPE (50K vocab) is ~4 bytes/token on English text. A 4K vocab
  trained from scratch on TinyStories gets ~3-3.5 bytes/token (less
  efficient because of the smaller vocab, more efficient on
  TinyStories's narrow distribution than GPT-2 BPE is on it).
  Single-byte (no merges, just bytes) is 1 byte/token by definition.

- **Why isn't this BPE compatible with GPT-2's BPE?**
  Different training corpus, different merge order, different final
  vocab. Even with identical algorithms, different data produces
  different tokenizers. To be GPT-2 BPE compatible, you'd need GPT-2's
  exact training data — which is proprietary.

### Test coverage

`tests/test_bpe.py` (15 cases):

Pretokenize:
- `test_pretokenize_splits_words_and_keeps_spaces`
- `test_pretokenize_separates_punctuation`
- `test_pretokenize_handles_numbers`

Train:
- `test_train_bpe_initial_vocab_is_all_bytes` — vocab_size=256 → no merges.
- `test_train_bpe_rejects_small_vocab` — ValueError on < 256.
- `test_train_bpe_learns_repeated_pair` — `'ababab'` → merge ('a', 'b') first.
- `test_train_bpe_learns_multiple_merges_in_order`
- `test_train_bpe_stops_when_no_more_pairs`

Encode/decode:
- `test_encode_decode_round_trip_after_training`
- `test_encode_decode_handles_unseen_text` — byte fallback for unseen chars.
- `test_encode_reduces_token_count_vs_bytes` — actual compression check.
- `test_encode_uses_lowest_merge_first` — priority order regression.

Serialization:
- `test_save_load_round_trip` — bytes ↔ hex JSON round-trip.
- `test_save_load_then_encode_matches`.

Non-ASCII:
- `test_bpe_handles_non_ascii_via_utf8_bytes` — "café résumé naïve"
  encodes and decodes correctly.

### CLI

```bash
python bpe.py train --corpus data/tinystories/train.txt --vocab-size 4096 --out tiny_bpe.json
python bpe.py encode --tokenizer tiny_bpe.json --text "hello world"
python bpe.py decode --tokenizer tiny_bpe.json --ids 264,1234,99
```

Training on TinyStories train set (~2 GB) at vocab_size=4096 takes ~10-30
minutes with the naive implementation. With `--max-chars 1_000_000` it's
under a minute and produces a usable demonstration tokenizer.

---

## 10. Mixture of Experts (Part 5 — as ablation variant)

**Files modified:**
- `layers.py:MoEFFN` — new class (router + N experts + load-balancing aux loss).
- `layers.py:TransformerBlock.__init__` — FFN picker now checks
  `config.num_experts >= 2` first.
- `config.py:ModelConfig` — new fields: `num_experts`, `top_k_experts`,
  `moe_aux_loss_coef`.
- `model.py:TransformerLM.forward` — accumulate per-block MoE aux loss
  and add `coef * total_aux` to the CE loss when training.
- `ablation.py:variants()` — new `"moe"` variant (num_experts=4, top_k=2).

**Tests:** `tests/test_layers.py` (+8 cases), `tests/test_model.py`
(+1 case), `tests/test_ablation.py` updated. Total 109 passing.

### Design

Switch Transformer-style MoE (Fedus et al. 2022). Each token is routed to
top-k of N expert FFNs; output is a weighted combination using router
probabilities. A load-balancing auxiliary loss prevents the common
failure mode where the router collapses to always picking the same
expert.

**Why MoE in this ablation matrix?** The "Parts 4-9" scoping question
naturally raised: can MoE be done at this scale? Answer: yes as an
ablation variant. At 10M-class scale the *finding* is interesting on its
own: MoE probably doesn't pay versus a matched-param dense baseline,
because the design point that justifies MoE (skip experts to save
compute on huge dense models) doesn't apply here. A clean negative
result is itself the contribution.

**Param matching.** Baseline GELU FFN at d_ffn=4·d_model has 8·d_model²
FFN params. N-expert MoE at d_ffn_per_expert = 4·d_model/N also has
8·d_model² (plus a tiny d_model·N router head). For d_model=128, N=4:
both at 131K, matched within ~0.5% (verified by
`test_moe_ffn_param_count_matches_gelu_baseline`).

### Walkthrough — `MoEFFN.forward`

```python
B, T, C = x.shape
x_flat = x.reshape(B * T, C)
router_logits = self.router(x_flat.float())
router_probs = F.softmax(router_logits, dim=-1)
top_probs, top_idx = router_probs.topk(self.top_k, dim=-1)
top_probs = top_probs / top_probs.sum(dim=-1, keepdim=True)
```

Flatten batch + time into one token axis (so we can do per-token routing
without tedious per-position bookkeeping). Compute router logits in fp32
(softmax stability + the load-balancing math being small-number-sensitive).
Pick top-k experts per token; renormalize so the top-k weights sum to 1
(otherwise the per-expert weighted sum would have norm < 1 in expectation,
biasing the output).

```python
N = self.num_experts
token_count = float(B * T)
one_hot = F.one_hot(top_idx, num_classes=N).float()
assigned = one_hot.sum(dim=1).clamp(max=1.0)
f_i = assigned.sum(dim=0) / token_count
P_i = router_probs.mean(dim=0)
self.last_aux_loss = N * (f_i * P_i).sum()
```

Compute aux loss. `assigned[t, e] = 1` if expert e is in token t's top-k,
else 0. `f_i` is the fraction of tokens routed to expert i.
`P_i` is the average router probability for expert i. At perfect balance:
`f_i = top_k / N`, `P_i = 1/N`, so `aux = N · N · (top_k/N · 1/N) = top_k`.
At collapse (one expert always picked): `f_0 = 1, P_0 ≈ 1`, so `aux ≈ N`.

```python
out_flat = torch.zeros_like(x_flat)
for e in range(N):
    slot_mask = (top_idx == e)
    token_mask = slot_mask.any(dim=-1)
    if not token_mask.any(): continue
    weight = (top_probs * slot_mask.to(top_probs.dtype)).sum(dim=-1)
    expert_in = x_flat[token_mask]
    expert_out = self.experts[e](expert_in)
    out_flat[token_mask] = out_flat[token_mask] + weight[token_mask].unsqueeze(-1) * expert_out
return out_flat.reshape(B, T, C)
```

Loop over experts (small N: 4). For each, gather the tokens routed to
it, compute the FFN forward on just those tokens, scatter the
weighted output back. This is the "dispatch + combine" pattern. For
large-N production MoE (e.g., 64+ experts) this is replaced by tensor
ops or specialized libraries (Megatron, DeepSpeed-MoE); for N=4 a
Python loop is fine.

### Why the aux loss is added in `model.forward` and not in MoEFFN

MoEFFN stores `self.last_aux_loss` after each forward. The model is the
right place to *use* it because:
- The aux loss is per-block, but the regularization is on the model as
  a whole — sum across blocks then multiply by one coefficient.
- The aux loss only matters when computing training loss (targets provided);
  during pure inference (`targets is None`), the aux can be ignored.
- The coefficient is a model-level hyperparameter, not a per-layer one.

Implementation in `model.forward`:
```python
aux_total = 0.0
for block in self.blocks:
    aux = getattr(block.ffn, "last_aux_loss", None)
    if aux is not None:
        aux_total = aux_total + aux
if isinstance(aux_total, torch.Tensor):
    coef = getattr(self.config, "moe_aux_loss_coef", 0.01)
    loss = loss + coef * aux_total
```

`getattr` with default returns None for non-MoE FFNs, so dense models
incur no cost. The `isinstance(aux_total, torch.Tensor)` guard means the
add only happens when at least one block contributed — otherwise
`aux_total` stays a Python float 0.0 and is skipped.

### Interview Qs you should be able to answer

- **Why MoE? What's the win?**
  Conditional computation: at training time, total params can be much
  larger than dense (because each token only activates a fraction). At
  inference, only top-k experts compute per token. Lets you scale model
  capacity without scaling per-token compute. Used in GPT-4, Mixtral 8x7B,
  Grok, DeepSeek V3.

- **Why doesn't it pay at 10M scale?**
  Two reasons. First, the FLOP savings only matter once the dense
  baseline is too expensive to train — at 10M, dense is trivially
  cheap, so MoE's "save compute" pitch is moot. Second, expert
  differentiation requires that each expert gets enough tokens to
  learn distinct functions; at small batch × small expert count, the
  router doesn't get the signal to differentiate.

- **What's the load-balancing loss for?**
  Without it, the router learns the easiest gradient path: route
  everything to one expert. That collapses MoE into "one expert + N-1
  dead experts." The aux loss penalizes imbalance: at perfect balance
  aux = top_k; at collapse aux = N. With coefficient 0.01, even a
  small imbalance creates a counter-gradient that nudges the router
  back toward uniform.

- **Why use the renormalized top-k probabilities for combining?**
  Top-k slicing throws away the bottom-(N-k) probabilities. If we used
  the raw top-k probs to weight expert outputs, the result would have
  norm proportional to `sum(top_probs)` which is generally < 1, biasing
  the model toward smaller activations. Renormalizing the top-k probs
  to sum to 1 keeps the activation magnitude consistent across tokens
  regardless of how peaked vs flat the router was.

- **Why top-k=2 (not 1)?**
  Top-1 (Switch Transformer) has the smallest compute but the largest
  gradient variance — only one expert gets a gradient signal per token.
  Top-2 doubles the compute but stabilizes training: gradients flow
  through both selected experts. Mixtral uses top-2; DeepSeek V3 uses
  top-8. There's no single right answer; it's a compute-vs-stability
  knob.

- **What does the router actually compute, geometrically?**
  A linear projection from `d_model` to `num_experts`, then softmax.
  Each expert's column in the router weight is essentially a "preferred
  direction" in embedding space — tokens whose embedding aligns with
  that direction get routed to that expert. Over training, experts
  specialize on different input distributions; the router learns to
  send each token to the experts that handle its kind of input best.
  At 10M scale this specialization is weak; at 100M+ it becomes
  measurable (e.g., one expert handles math, another handles code).

### Test coverage

`tests/test_layers.py` (+8):
- `test_block_moe_picker` — `num_experts>=2` -> MoEFFN; else GELU/SwiGLU.
- `test_moe_ffn_output_shape`
- `test_moe_ffn_rejects_top_k_above_num_experts`
- `test_moe_ffn_aux_loss_is_set_after_forward` — None before, tensor after.
- `test_moe_ffn_aux_loss_at_perfect_balance` — uniform router, aux ≈ top_k.
- `test_moe_ffn_aux_loss_at_collapse` — collapsed router, aux > balanced.
- `test_moe_ffn_gradient_flows` — router gradient required; individual
  experts may not get grad if no tokens routed to them in this batch.
- `test_moe_ffn_param_count_matches_gelu_baseline` — within 1% of
  baseline at d_model=128, N=4.

`tests/test_model.py` (+1):
- `test_model_with_moe_forward_and_loss_includes_aux` — end-to-end
  through TransformerLM. Both dense and MoE models forward + backward
  successfully; MoE loss includes the aux term.

`tests/test_ablation.py`:
- `test_variants_dict_contents` updated to expect 6 variants including
  `moe`, with `num_experts=4, top_k_experts=2`.

### Open question for the ablation run

The hypothesis is that at toy scale, `moe` will roughly match `baseline`
on val loss but cost slightly more wall-clock (router overhead). The
real signal would come at 100M+ dense baseline, which is out of scope.
The `summary.csv` should show whether this hypothesis holds.

**Ablation result (2026-05-20):** `moe` ended up *worse* than `baseline`
on val loss (3.455 vs 3.417, +0.04 nats) and ~2.6x slower per step
(251 s/run vs 95 s). See `ABLATION.md` for the full discussion. The
"slightly more wall-clock" prediction held; the "roughly match" prediction
was too optimistic — top-2 of 4 experts has ~half the active params per
token, and at this scale that capacity cut isn't compensated by routing.

---

## 11. SFT (Part 6) — `sft_data.py` + `sft.py`

### Goal

Demonstrate the supervised fine-tuning mechanic on the pretrained model
(`runs/ablation/modern/seed_1/best.pt`): teach it to follow a fixed
input format that the base model has never seen. Specifically, after
SFT the prompt `"Here is a story:\n\n"` should reliably elicit a
TinyStories-style story.

This is **not** instruction tuning in the modern sense (no chat-format
data, no held-out instructions). It's the minimum viable demo of:
(a) loss masking, (b) lower-LR fine-tuning, (c) format learning.

### Design decisions

**Template choice.** Fixed prefix `"Here is a story:\n\n"` on every
example. Alternatives considered: per-example title extraction
(`"Story title: <noun phrase>\n\nStory: <body>"`) and question-answer
phrasings. Rejected for time-budget reasons — the fixed prefix is the
cleanest demo of the SFT mechanic without per-example NLP preprocessing
that would add scope without changing the conclusions.

**Loss masking via `ignore_index=-100`.** `TransformerLM.forward` already
calls `F.cross_entropy(..., ignore_index=-100)`. So masking prompt
tokens means: produce targets with `-100` wherever loss should be
skipped, and the existing forward path Just Works. No model surgery.

Concrete pipeline:
1. `sft_data.py` writes parallel `tokens.bin` (uint16) and `mask.bin`
   (uint8) where `mask=0` on prompt tokens, `mask=1` on completion + EOT.
2. `get_sft_batch` samples random windows (sample-with-replacement,
   matching pretrain), then sets `targets[mask==0] = -100` before
   returning the tensors.
3. Training is unchanged from pretrain except for: smaller LR, no
   weight decay, fewer steps.

**Why no weight decay?** Standard SFT practice. The pretrained weights
were already regularized during pretraining; further WD during fine-tuning
just degrades them away from where the pretraining process landed.
GPT-3/InstructGPT-style fine-tuning recipes follow this. (If we were
SFT-ing from scratch on a much larger dataset, decay might come back.)

**LR choice.** 3e-5 (one order of magnitude below pretrain's 3e-4).
Standard rule of thumb: SFT LR should be 1/10 to 1/100 of pretrain LR
to avoid overwriting the base distribution. At LR=3e-4 the base model
collapses to producing only the SFT-distribution outputs and forgets
the rest of the pretraining distribution (catastrophic forgetting).

**Random sampling over example boundaries.** Windows can straddle an
EOT (end-of-story) and start mid-prefix of the next example. This is
intentional and matches pretraining's flat-stream sampling. The mask
remains correct because it was written in lockstep with tokens. The
alternative (per-example padding) would waste compute on padding tokens
and complicate the batch interface.

### Why fixed-prefix SFT demonstrates the mechanic honestly

A skeptic asks: "If the base model already produces TinyStories
continuations from any TinyStories-like prompt, what did SFT actually
teach?" The answer is in the before/after sample at the same prompt
`"Here is a story:\n\n"` (seed 42, temp 0.8, top-k 40):

- **Before SFT** (`runs/ablation/modern/seed_1/best.pt`): The base model
  continues from "Here is a story:" by treating it as mid-document text
  — it dives into dialogue mid-scene ("She said, 'Wow, Lily. It is a
  beautiful place.'"). It also emits `<|endoftext|>` mid-output because
  the token was in pretrain but never used as a clean stop.
- **After SFT** (`runs/sft/modern/best.pt`): The model recognizes the
  format and opens with a canonical TinyStories story-start ("One day,
  a little boy named Lily found a big house in a big room."). No stray
  `<|endoftext|>`.

So SFT taught a specific behavior: "when you see *this exact prefix*,
start a fresh story." That's the format-following result that scales
up to "when you see `<|user|>...<|assistant|>`, respond as the assistant
would" in real chat-tuned models.

### Test coverage (`tests/test_sft.py`, 5 new tests, 114 total)

- `test_iter_stories_splits_on_eot` — parser correctly splits a synthetic
  TinyStoriesV2-style file on `<|endoftext|>` separators.
- `test_mask_zero_on_prefix_one_elsewhere` — after `build_split`, the
  mask has exactly `n_stories * n_prefix_tokens` zeros and the rest ones.
- `test_get_sft_batch_sets_targets_to_minus_100_where_mask_zero` —
  positional alignment: target is `-100` iff `mask=False`, else target
  is a valid token id (>=0).
- `test_get_sft_batch_has_some_unmasked_positions` — sanity guardrail.
  If a window were entirely masked the SFT step would learn nothing.
  At default settings >50% of positions should be unmasked.
- `test_sft_one_step_finite_loss_and_grad` — end-to-end: build a tiny
  TransformerLM, sample an SFT batch, forward + backward, verify loss
  is finite and at least one parameter received a non-zero gradient.

### Interview-style questions to defend

1. **"Why use `ignore_index=-100` instead of multiplying the per-token
   loss by a 0/1 mask?"**
   Both work; `-100` is cleaner because it skips the softmax + log-prob
   computation for those positions entirely (cross_entropy short-circuits),
   whereas the multiplicative mask still does the full math and then
   zeros it out. Memory and FLOPs win, no behavioral difference.

2. **"Why does the mask align with TARGETS (`y`), not INPUTS (`x`)?"**
   Because the loss at position `t` is on the predicted distribution over
   *next* token, which is `y[t] = tokens[t+1]`. The mask should describe
   "do I care about the prediction at this step?" — which is exactly
   "is the *target* a completion token?" So `mask` is sliced from
   `tokens[s+1:s+1+T]`, not `tokens[s:s+T]`.

3. **"What goes wrong if we SFT at the pretrain LR (3e-4)?"**
   Catastrophic forgetting. The pretrained distribution shifts hard
   toward the narrow SFT distribution. Visible failure mode: the model
   produces *only* the prefix-following format and forgets how to handle
   normal continuations. At LR=3e-5 the shift is gentle and the base
   distribution is mostly preserved outside the prefix.

4. **"Why does the model still confuse pronouns and character names
   after SFT?"**
   SFT shifts behavior, not capability. We're still at 566k non-embedding
   params trained on 4M pretrain tokens + ~150k SFT tokens-of-loss
   (500 steps × batch 16 × ~18 unmasked positions on average). Pronoun
   coherence requires long-range bookkeeping the model doesn't have at
   this scale. SFT can't add capability the base doesn't have.

### Smoke run results (`runs/sft/modern/`)

- 500 steps, LR 3e-5 (cosine to 3e-6), warmup 50, batch 8 × accum 2.
- Wall clock: ~34 s on RTX 4060.
- Val loss trajectory: 3.41 (step 100) → 3.37 (step 300, best) → 3.39
  (step 400). Modest absolute change because the base model is already
  good at TinyStories continuation; SFT is teaching format, not content.
- Best checkpoint at step 300 (`runs/sft/modern/best.pt`).

### Open follow-ups for Parts 7-9

- The current SFT bin has no per-example-boundary metadata, so we can't
  build purely per-example batches. For RLHF rollouts (Part 8 PPO) we
  need per-prompt generation, which means a different data interface
  — likely a separate `prompts.jsonl` emitted alongside the SFT bin,
  or rebuilding the bin with offset metadata.
- For Part 7 (RM/RLAIF): the SFT'd model is the rollout policy. We'll
  need to generate K completions per prompt and ask Claude/GPT-4 to
  pick a winner. API cost estimate: ~$5-20 depending on prompt count.

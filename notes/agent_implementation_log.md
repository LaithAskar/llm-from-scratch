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

<!-- Next entries (MultiHeadAttention, GELU FFN, SwiGLU FFN, RoPE,
     TransformerBlock) appended as components land. -->

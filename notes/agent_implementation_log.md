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

<!-- Next entries (RMSNorm, MultiHeadAttention, GELU FFN, SwiGLU FFN, RoPE,
     TransformerBlock) appended as components land. -->

# Ablation Study: Modern vs Classic Transformer at Toy Scale

## What this measures

Five architectural switches that distinguish a modern LLaMA-style transformer
from the classic GPT-2 stack, plus a Mixture-of-Experts variant, evaluated at
**toy scale** (~590k non-embedding params, 7M total with the GPT-2 vocab
embedding) on TinyStories. Every variant is trained with identical seed
schedule, identical compute budget, identical parameter count (FFN width
adjusted to match), and identical data sampling. Any quality delta is
attributable to the architecture knob, not confounded by compute or size.

## Setup

| | |
|-|-|
| Dataset | TinyStories V2 GPT-4-cleaned (564M train tokens, 5.7M val tokens) |
| Tokenizer | tiktoken GPT-2 BPE (vocab 50257) |
| Model | 3 layers, 4 heads, d_model=128, context 128 |
| Params (non-emb) | 590,720 |
| Training | AdamW, lr=3e-4 → 3e-5 cosine, 100 warmup steps, **2000 steps**, effective batch 16 (micro 8 × accum 2), bf16 AMP |
| Hardware | RTX 4060 Laptop 8 GB, Windows 11, PyTorch 2.6.0+cu124 |
| Seeds | 0, 1, 2 |

## Variants

| Variant     | Norm        | FFN                          | Position encoding |
|-------------|-------------|------------------------------|-------------------|
| `baseline`  | LayerNorm   | GELU (d_ffn = 4·d_model)     | Learned-absolute  |
| `rmsnorm`   | **RMSNorm** | GELU                         | Learned-absolute  |
| `rope`      | LayerNorm   | GELU                         | **RoPE**          |
| `swiglu`    | LayerNorm   | **SwiGLU** (d_ffn = 8/3·d_model, matched params) | Learned-absolute |
| `modern`    | **RMSNorm** | **SwiGLU**                   | **RoPE**          |
| `moe`       | LayerNorm   | **MoE** (4 experts, top-2, d_ffn = d_model per expert) | Learned-absolute |

Parameter count is matched across non-MoE variants by sizing SwiGLU's three
matrices at `(8/3)·d_model` to equal two GELU matrices at `4·d_model`
(PaLM, Chowdhery et al. 2022). The `moe` variant has the same *total*
FFN capacity but lower *active* params per token (top-2 of 4 experts).

## Results

### Best validation loss (3 seeds each)

| Variant     | Best val (mean ± range) | Val PPL (mean) | Δ vs baseline | Wall-clock (mean) |
|-------------|-------------------------|----------------|---------------|-------------------|
| `baseline`  | 3.417 ± 0.030           | 31.6           | —             | 95 s              |
| `rmsnorm`   | 3.416 ± 0.029           | 31.7           | **−0.001** (noise) | 108 s        |
| `rope`      | 3.191 ± 0.025           | 25.2           | **−0.226**    | 105 s             |
| `swiglu`    | 3.369 ± 0.032           | 30.1           | **−0.048**    | 88 s              |
| **`modern`** | **3.146 ± 0.027**      | **24.1**       | **−0.272**    | 114 s             |
| `moe`       | 3.455 ± 0.020           | 32.8           | **+0.037**    | 251 s             |

The seed range across 3 seeds is ~0.02–0.03 for every variant. Any
|Δ| > ~0.05 is comfortably outside seed noise.

Numbers come from `runs/ablation/summary.csv`. `best val` is the lowest
validation loss observed across the 9 val evaluations during each 2000-step
run (eval cadence: every 200 steps). Perplexity reported here is the mean
of `ppl_val` (separate evaluation of the best checkpoint on 100 val batches).

### Loss curves

![training loss by variant](runs/ablation/figures/loss_curves_train.png)
![validation loss by variant](runs/ablation/figures/loss_curves_val.png)
![best val per variant](runs/ablation/figures/best_val_bar.png)

## Discussion

**RoPE does almost all the work.** The `modern` stack beats `baseline` by
0.27 nats. RoPE alone contributes 0.23 of that — roughly 85% of the total
gain. The remaining ~0.05 nats are SwiGLU; RMSNorm contributes
indistinguishable signal.

**RMSNorm is a wash at this scale.** Mean val loss is 3.416 vs `baseline`'s
3.417 — a difference of one thousandth of a nat, well inside seed noise.
This is consistent with RMSNorm's real advantages being elsewhere:
fewer operations, no mean subtraction, slightly better numerical
behavior at scale. None of those show up as a loss improvement at
~590k params and 2000 steps.

**SwiGLU helps a little.** A 0.05-nat improvement over baseline — outside
the seed range of any single variant, but small. It's possible the gain
grows with depth or width; at 3 layers and d_model=128 we are at the
small end of where SwiGLU's gating mechanism has room to express anything
the GELU FFN can't already represent. Param count is matched (8/3 × d_model
for SwiGLU vs 4 × d_model for GELU) so this isn't a capacity confound.

**MoE underperforms the dense baseline.** `moe` (4 experts, top-2 routing)
ends up 0.04 nats *worse* than `baseline` despite having the same total
FFN parameter count. Two things are happening:

1. With top-2 of 4 experts, each token sees only ~half the FFN capacity
   per forward pass that the dense baseline sees. At this scale, the
   sparse routing doesn't pay for itself — there isn't enough total
   capacity for specialization to matter.
2. Routing adds wall-clock cost: `moe` takes 251 s/run vs ~95 s for the
   dense baseline — **~2.6× slower for worse loss**. The routing overhead
   (top-k selection, scatter/gather, auxiliary load-balance loss) is
   pure cost at this size.

This is the canonical MoE-at-small-scale result. MoE wins when total
parameter count vastly exceeds active parameter count per token (Switch
Transformer: 1.6T total, ~7B active). At 590k total, there is no
"specialization budget" to spread across experts.

**Practical takeaway for this scale.** If you're shipping a small
transformer and can pick exactly one modernization, **pick RoPE**.
Adding RMSNorm gets you nothing at this scale; adding SwiGLU gets you
a small bonus. MoE is a net loss without dramatically more parameters.

## Sanity check: text generation

Sample from `runs/ablation/modern/seed_1/best.pt` (val 3.118, the best run
overall), prompted with `"Once upon a time"`, temperature 0.8, top-k 40:

> *Once upon a time, there was a little boy named Lily. She had a friend,
> big ball named Max. Tim was very happy. He wanted to play with his
> friend, Tim, named Max. Tim had a great day. Tim loved to play with
> his friends. One day, Tim saw a big tree. Tom wanted to play with
> his mom. Mia was a big, but he could not have a new friends. He was
> sad and had an idea. He had lots of many toys. Tim was very happy.
> He looked at the sun, but she could not like the toy box. He thought
> it was a big*

Grammatical, recognizes the story-opening register, uses TinyStories
vocabulary. Character coherence and pronoun agreement are broken
(Lily/Max/Tim/Tom/Mia in one paragraph, "he"/"she" swapped) — expected
at 590k params and ~4M training tokens (<1% of one epoch over TinyStories).
This confirms the training/sampling pipeline works end-to-end; the model
is in the "babbling fluently" regime, not the "tells coherent stories"
regime.

## Limitations

- **Toy scale.** 590k non-embedding params is ~5 orders of magnitude smaller
  than frontier LLMs. Architectural choices that matter at scale may show
  no signal here, and vice versa.
- **Short training.** 2000 steps × effective batch 16 × context 128 ≈ 4M
  tokens trained. TinyStories has 564M train tokens, so we see <1% of an
  epoch. Loss curves are still declining at the end of training.
- **Single dataset.** TinyStories has constrained vocabulary and syntax.
  Architectural choices interact with data distribution; these results
  do not necessarily transfer to general web text.
- **3 seeds.** Enough to flag obvious wins from noise but not for tight
  confidence intervals. Treat small inter-variant differences as inconclusive.
- **No regularization sweep.** Single dropout (0.1), single weight decay
  (0.1), single LR. A variant that loses here at fixed hparams might win
  after tuning.

## Reproducing

```
python data.py prepare --out data/tinystories
python ablation.py --out runs/ablation --data data/tinystories --seeds 0 1 2
python plot_ablation.py --root runs/ablation
```

## Author and authorship

Per-component implementation by Claude (Opus 4.7) under Laith Askar's
project-management and code-review supervision. See [README.md](README.md)
for the full authorship split and
[`notes/agent_implementation_log.md`](notes/agent_implementation_log.md)
for the implementation log used in review.

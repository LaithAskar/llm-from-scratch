# llm-from-scratch-laith — Transformer + Ablation Study (AI-assisted)

A from-scratch GPT-style transformer language model (~10M params, trained
on TinyStories) with a **controlled architectural ablation** comparing the
classic Transformer stack against a modern LLaMA-style stack at matched
parameter count.

## Authorship and how this project was built

This is an **AI-assisted, human-supervised** project. Honest split of work:

- **Laith Askar (project manager + reviewer):** project scope and goals,
  ablation matrix design, hardware/dataset scoping decisions for Parts 4-9
  (skipped — see *Scope* below), code-review of the agent's output, test
  review.
- **Claude (Anthropic, Opus 4.7 1M-context):** all code in
  `layers.py` (MultiHeadAttention, RMSNorm, RotaryEmbedding, GeluFFN,
  SwiGLUFFN, TransformerBlock, causal_mask), surrounding infrastructure
  (`config.py`, `data.py`, `model.py`, `train.py`, `eval.py`,
  `ablation.py`), the 76-case test suite, the CI workflow, and the
  implementation log.

A per-component **implementation log** with design rationale,
interview-style defense questions, and pitfall notes lives at
[`notes/agent_implementation_log.md`](notes/agent_implementation_log.md) —
this is the load-bearing artifact for the supervision/review claim.

To see code Laith wrote *without* AI implementation, see the two pre-AI
commits in `git log` (`ce75355`, `34eed37`): project scope notes and
`verify_setup.py`. Everything after is AI-implemented under PM review.

## Why this project exists

Most "GPT from scratch" repos stop at "I trained a model." This one asks
the next question: **how much does each modern architectural choice
actually buy you at small model scale?** The answer is roughly known at
frontier scale (the modern stack wins) but rarely measured cleanly in the
tiny-model regime that fits on a laptop GPU.

## The ablation

Identical seed, identical dataset, identical compute budget, identical
parameter count across all variants. Single architecture knob per row:

| Variant     | Norm        | FFN activation | Position encoding |
|-------------|-------------|----------------|-------------------|
| `baseline`  | LayerNorm   | GELU           | Learned-absolute  |
| `rmsnorm`   | RMSNorm     | GELU           | Learned-absolute  |
| `rope`      | LayerNorm   | GELU           | RoPE              |
| `swiglu`    | LayerNorm   | SwiGLU         | Learned-absolute  |
| `modern`    | RMSNorm     | SwiGLU         | RoPE              |
| `moe`       | LayerNorm   | GELU (4 experts, top-2) | Learned-absolute |

Parameter count is held constant via PaLM's `d_ffn = (8/3) * d_model` rule
for SwiGLU variants — three FFN matrices sized to match a two-matrix
GELU-FFN at `4 * d_model`. The `moe` variant uses per-expert
`d_ffn = d_model` × 4 experts to match the baseline's `4 * d_model` total
FFN capacity, though *active* params per token are lower (top-2 routing).

**Deliverable:** `runs/ablation/summary.csv` (18 runs) + `ABLATION.md`
writeup with the per-variant best-val, perplexity, wall-clock, and a
discussion of which switches actually move the needle at toy scale.
See [ABLATION.md](ABLATION.md) for the results.

## Status

- Environment verified (Python 3.13, PyTorch 2.6.0+cu124, RTX 4060 8 GB).
- **Parts 1-5 + Part 3\* (KV cache) + Part 6 (SFT): complete.**
- `layers.py`: MHA, causal_mask, RMSNorm, GeluFFN, SwiGLUFFN,
  RotaryEmbedding, TransformerBlock, MoEFFN.
- Training infrastructure: `config.py`, `data.py`, `model.py`, `train.py`,
  `eval.py`, `ablation.py`.
- Part 3\*: KV cache plumbed through MHA + Block + `TransformerLM.generate`.
- Part 4: standalone BPE trainer (`bpe.py`) — used by tests; main pipeline
  uses `tiktoken` GPT-2.
- Part 5: MoE as a 6th ablation variant (honest negative result; see
  `ABLATION.md`).
- Part 6: SFT pipeline (`sft_data.py`, `sft.py`) with fixed-prefix template,
  loss masking via `ignore_index=-100`, before/after sample showing format
  learning. Smoke run at `runs/sft/modern/`.
- Ablation matrix complete: 6 variants × 3 seeds × 2000 steps, 18 runs.
- **114 pytest cases passing**, CI green on Python 3.11 + 3.12.
- **Next session:** Parts 7 (RM/RLAIF) → 8 (PPO) → 9 (GRPO).

## Quickstart

Requires Python ≥ 3.11. A CUDA GPU is recommended; CPU works for tests
and small smoke runs.

```bash
# Install (CPU torch shown; for CUDA see requirements.txt)
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -e ".[dev]"

# Verify environment
python verify_setup.py

# Run tests (~8s, 114 cases)
pytest tests/ -q

# Download + tokenize TinyStories (~2 GB)
python data.py prepare

# Smoke training: 1 variant, ~200 steps (must exceed warmup_steps)
python ablation.py --max-steps 200 --variants baseline --seeds 0

# Full ablation: 6 variants × 3 seeds × 2000 steps (~30 min on RTX 4060)
python ablation.py --seeds 0 1 2

# Plots from runs/ablation/summary.csv
python plot_ablation.py --root runs/ablation

# Sample from any trained checkpoint
python generate_sample.py runs/ablation/modern/seed_1/best.pt \
    --prompt "Once upon a time" --tokens 150

# Part 6: SFT (fixed-prefix template, ~1 min on RTX 4060)
python sft_data.py prepare --raw data/tinystories --out data/sft
python sft.py --base runs/ablation/modern/seed_1/best.pt \
    --data data/sft --out runs/sft/modern --max-steps 500
python generate_sample.py runs/sft/modern/best.pt \
    --prompt "Here is a story:

"
```

## Scope decisions (Parts 4-9 and KV cache)

The source video curriculum has nine parts. The original scope plan
(Parts 1-3 + ablation only) was revised after the authorship reframing
to a PM/reviewer model (see *Authorship* above). Each remaining part was
re-scoped to find an honest, in-budget implementation. Status today:

| Part | What                  | Status      | Honest framing                                                                                                                            |
|------|-----------------------|-------------|-------------------------------------------------------------------------------------------------------------------------------------------|
| 3\*  | KV cache              | **Done**    | Inference-only optimization. Default `use_cache=True` in `generate`; hard-stops at `context_len`. Verified against the recompute path.   |
| 4    | BPE training          | **Done**    | Standalone implementation in `bpe.py` with CLI; the main training pipeline still uses `tiktoken.get_encoding("gpt2")` (50257-vocab GPT-2). |
| 5    | Mixture of Experts    | **Done**    | Added as the 6th ablation variant. **Honest negative result** at this scale — see `ABLATION.md` discussion. Not a win; documented as such. |
| 6    | Supervised fine-tuning | **Done**   | Fixed-prefix template (`"Here is a story:\n\n"`) over TinyStories. Demonstrates loss-masking + format learning mechanics, not real instruction tuning. Documented as such in `notes/agent_implementation_log.md` §11. |
| 7    | Reward modeling       | Next session | RLAIF: use Claude/GPT-4 as preference oracle over K policy completions per prompt. ~$5-20 API spend.                                  |
| 8    | PPO                   | Next session | Standard impl on top of the SFT'd policy + Part 7 RM. ~16 MB extra VRAM for the reference policy at toy scale.                          |
| 9    | GRPO                  | Next session | DeepSeek group-relative variant on top of the Part 8 PPO infrastructure.                                                                |

The previous scoping (only Parts 1-3 implemented) was the right call
under the earlier "no AI on layers.py" rule. Under the current PM/review
model, each later part has a narrow, honest implementation that's
documented with its limitations.

## Hardware

- Windows 11, NVIDIA RTX 4060 Laptop GPU (8 GB VRAM), driver 581.83
- Python 3.13.13 (local), 3.11 & 3.12 (CI)
- PyTorch 2.6.0+cu124
- AMP (bf16) + gradient accumulation are mandatory to fit at the chosen
  config

## Project structure

```
.
├── config.py                 # ModelConfig / TrainConfig / Config dataclasses
├── data.py                   # TinyStories downloader, tiktoken BPE -> uint16 memmap
├── layers.py                 # MHA, RMSNorm, RoPE, FFNs, MoEFFN, TransformerBlock, causal_mask
├── model.py                  # TransformerLM: embeddings + blocks + tied LM head + generate(use_cache=True)
├── train.py                  # AdamW + cosine LR + AMP + grad accum + checkpoints
├── eval.py                   # Perplexity + sampling from a checkpoint
├── ablation.py               # Run all variants × seeds, write summary.csv
├── bpe.py                    # Part 4: standalone BPE trainer + encode/decode + CLI
├── sft_data.py               # Part 6: SFT pairs (tokens + loss-mask bin files)
├── sft.py                    # Part 6: fine-tuning loop with masked CE
├── generate_sample.py        # Sample text from any pretrained or SFT'd checkpoint
├── plot_ablation.py          # Loss curves + best-val bar from runs/ablation/
├── ABLATION.md               # Full ablation writeup with results + discussion
├── pyproject.toml            # Project metadata, pytest pythonpath
├── verify_setup.py           # CUDA / deps sanity check
├── .github/workflows/        # CI: pytest on push/PR, Python 3.11 + 3.12
├── notes/
│   ├── layers_cheatsheet.md            # Concept reference for layers.py
│   └── agent_implementation_log.md     # Per-component design + walkthrough (review artifact)
└── tests/                    # 114 pytest cases
    ├── test_layers.py        # MHA, RMSNorm, RoPE, FFNs, TransformerBlock, MoEFFN
    ├── test_model.py         # LM wrapper (incl. MoE end-to-end)
    ├── test_train.py         # Training loop
    ├── test_eval.py          # Checkpoint loading + perplexity + sample
    ├── test_ablation.py      # Ablation runner (6 variants)
    ├── test_data.py          # Pretrain data pipeline
    ├── test_bpe.py           # Standalone BPE trainer
    └── test_sft.py           # SFT mask construction + masked-CE smoke
```

## CI

GitHub Actions runs the full test suite on every push to `main` and on
every PR, against Python 3.11 and 3.12. See
[`.github/workflows/test.yml`](.github/workflows/test.yml).

## License

MIT (see [`LICENSE`](LICENSE)).

## Acknowledgments

- Vivek Kalyanarangan's freeCodeCamp curriculum
  [LLMs from Scratch](https://youtu.be/p3sij8QzONQ) and the matching
  reference repo
  [`vivekkalyanarangan30/llm_from_scratch`](https://github.com/vivekkalyanarangan30/llm_from_scratch).
  The project scope is derived from the first three parts of his video.
  Code in this repo was *not* copied from the reference; it was
  re-implemented by Claude based on the architecture descriptions.

Primary references for the modern components ablated here:
- **RMSNorm**: Zhang & Sennrich, *Root Mean Square Layer Normalization* (2019)
- **RoPE**: Su et al., *RoFormer: Enhanced Transformer with Rotary Position
  Embedding* (2021)
- **SwiGLU**: Shazeer, *GLU Variants Improve Transformer* (2020); PaLM (2022)
  for the matched-FFN sizing rule
- **Pre-norm vs post-norm**: Xiong et al., *On Layer Normalization in the
  Transformer Architecture* (2020)
- **Tied LM head, AdamW param-group split**: Karpathy's nanoGPT

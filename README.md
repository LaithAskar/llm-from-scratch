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

Parameter count is held constant via PaLM's `d_ffn = (8/3) * d_model` rule
for SwiGLU variants — three FFN matrices sized to match a two-matrix
GELU-FFN at `4 * d_model`. Any quality delta is attributable to the
architecture knob, not param count.

**Deliverable:** `runs/ablation/summary.csv` with per-variant best-val,
final-step val perplexity, and wall-clock; plotting/analysis to follow.

## Status

- Environment verified (Python 3.13, PyTorch 2.6.0+cu124, RTX 4060 8 GB).
- Infrastructure: `config.py`, `data.py`, `model.py`, `train.py`,
  `eval.py`, `ablation.py` — **complete**, 28 pytest cases.
- `layers.py`: MHA, causal_mask, RMSNorm, GeluFFN, SwiGLUFFN,
  RotaryEmbedding, TransformerBlock — **complete**, 48 pytest cases.
- **76 pytest cases passing**, CI green on Python 3.11 + 3.12.
- TinyStories download + full ablation run: pending.
- `ABLATION.md` writeup: pending (after the run).

## Quickstart

Requires Python ≥ 3.11. A CUDA GPU is recommended; CPU works for tests
and small smoke runs.

```bash
# Install (CPU torch shown; for CUDA see requirements.txt)
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -e ".[dev]"

# Verify environment
python verify_setup.py

# Run tests (~5s, 76 cases)
pytest tests/ -q

# Download + tokenize TinyStories (~2 GB)
python data.py prepare

# Smoke training: 1 variant, 500 steps
python ablation.py --max-steps 500 --variants baseline --seeds 0

# Full ablation: 5 variants × 3 seeds × 5000 steps
python ablation.py

# Use a trained checkpoint
python eval.py perplexity --ckpt runs/ablation/modern/seed_0/best.pt --data data/tinystories
python eval.py sample --ckpt runs/ablation/modern/seed_0/best.pt \
    --prompt "Once upon a time" --max-tokens 100 --top-k 50
```

## Scope decisions (Parts 4-9 and KV cache)

The source video curriculum has nine parts; only Parts 1-3 are implemented.
Each of the rest was evaluated and scoped out for hardware/dataset reasons
that are independent of the project's AI-assisted authorship:

| Part | What                | Why out of scope                                                                                                                       |
|------|---------------------|----------------------------------------------------------------------------------------------------------------------------------------|
| 3*   | KV cache            | Inference-only optimization; the ablation measures training quality, not inference latency. Cleanly implementing it without affecting training-path correctness would add 1-2 days for no ablation signal. |
| 4    | BPE training        | A day of work for ~zero learning over using `tiktoken.get_encoding("gpt2")`. The BPE algorithm is interesting; engineering one is not. |
| 5    | Mixture of Experts  | 8 GB VRAM cannot hold one expert at the model size where MoE gains become measurable (gating needs ≥~100M dense baseline). MoE on 10M is a toy of a toy. |
| 6    | Supervised fine-tuning | No relevant instruction dataset exists for a 10M-param TinyStories-trained base. SFT'ing a TinyStories model on Alpaca data is incoherent. |
| 7    | Reward modeling     | Needs a preference dataset (human ranking of outputs from this specific base). None exists; building one is its own project. |
| 8    | PPO                 | Requires (6) and (7); compounding the upstream gaps would produce a fake artifact. |
| 9    | GRPO                | Same constraint as (8). |

A faked Part 4-9 implementation is a worse resume signal than clean Parts
1-3 + ablation with explicit *why-not* on the rest. The scoping decision
is itself a project artifact.

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
├── layers.py                 # MHA, RMSNorm, RoPE, FFNs, TransformerBlock, causal_mask
├── model.py                  # TransformerLM: embeddings + blocks + tied LM head + generate
├── train.py                  # AdamW + cosine LR + AMP + grad accum + checkpoints
├── eval.py                   # Perplexity + sampling from a checkpoint
├── ablation.py               # Run all variants × seeds, write summary.csv
├── pyproject.toml            # Project metadata, pytest pythonpath
├── verify_setup.py           # CUDA / deps sanity check
├── .github/workflows/        # CI: pytest on push/PR, Python 3.11 + 3.12
├── notes/
│   ├── layers_cheatsheet.md            # Concept reference for layers.py
│   └── agent_implementation_log.md     # Per-component design + walkthrough (review artifact)
└── tests/                    # 76 pytest cases
    ├── test_layers.py        # 48 cases over MHA, RMSNorm, RoPE, FFNs, TransformerBlock
    ├── test_model.py         # 10 cases over the LM wrapper
    ├── test_train.py         # 4 cases over the training loop
    ├── test_eval.py          # 4 cases over checkpoint loading + perplexity + sample
    ├── test_ablation.py      # 3 cases over the ablation runner
    ├── test_data.py          # 5 cases over the data pipeline
    └── conftest.py + _dummies.py   # (now mostly vestigial; layers complete)
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

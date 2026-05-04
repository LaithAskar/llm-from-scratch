# LLM From Scratch — with an Ablation Study

Building a small Transformer language model in PyTorch from the ground up, then running a controlled ablation study to measure the contribution of each "modern" component (RMSNorm, RoPE, SwiGLU) versus the classical baseline (LayerNorm, learned absolute position embeddings, GELU).

This repo is my **own implementation**, written while learning from Vivek Kalyanarangan's freeCodeCamp course [LLMs from Scratch](https://youtu.be/p3sij8QzONQ) and the matching reference repo [vivekkalyanarangan30/llm_from_scratch](https://github.com/vivekkalyanarangan30/llm_from_scratch). Code here is re-derived rather than copied; the ablation study is original work.

## Why this exists

Most "GPT from scratch" repos stop at "I trained a model." This one asks the next question: **how much does each modern architectural choice actually buy you on a small model?** The answer is mostly known at scale, but rarely measured cleanly at the tiny-model regime that's accessible on a laptop GPU.

## Hardware

Trained on an NVIDIA RTX 4060 Laptop GPU (8 GB VRAM), Windows 11. Everything is sized so the full ablation suite fits in this budget using AMP + gradient accumulation.

## Plan

| Phase | Goal | Status |
|---|---|---|
| Part 1 | Core Transformer (attention, MHA, FFN, residuals, LayerNorm) | not started |
| Part 2 | Train a tiny LM on character-level data | not started |
| Part 3 | Modernize: RMSNorm, RoPE, SwiGLU, KV cache | not started |
| Ablation | Measure delta of each modern component vs baseline | not started |
| Writeup | `ABLATION.md` with curves, hypotheses, surprises | not started |

Parts 4–9 of the source course (BPE training, MoE, SFT, reward modeling, PPO, GRPO) are intentionally **not implemented here** — they require compute beyond a single 8 GB GPU and would dilute the focus of this project. They are studied conceptually but not reproduced.

## Ablation design

Identical seed, identical dataset, identical compute budget across runs. Single-variable change per row.

| Variant | Position encoding | Norm | FFN activation |
|---|---|---|---|
| Baseline (modern) | RoPE | RMSNorm | SwiGLU |
| -RoPE | learned absolute | RMSNorm | SwiGLU |
| -RMSNorm | RoPE | LayerNorm | SwiGLU |
| -SwiGLU | RoPE | RMSNorm | GELU |

Reported metrics: train loss curve, val loss curve, final perplexity, wall-clock per step, VRAM peak. Findings written up in `ABLATION.md`.

## Setup

```
python -m venv .venv
.venv\Scripts\activate            # Windows
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
python verify_setup.py
```

`verify_setup.py` must print "All checks passed" before starting Part 1. If CUDA is unavailable, the install above fixes it.

## Credits

- Vivek Kalyanarangan — [LLMs from Scratch (freeCodeCamp, 6h 6m)](https://youtu.be/p3sij8QzONQ) and the reference repo.
- Karpathy's nanoGPT, Su et al. (RoPE, 2021), Zhang & Sennrich (RMSNorm, 2019), Shazeer (SwiGLU, 2020) — primary sources for the components being ablated.

## License

MIT. See `LICENSE`.

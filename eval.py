"""
Evaluate a trained checkpoint: validation perplexity and sample generation.

CLI:
    python eval.py perplexity --ckpt runs/exp1/best.pt --data data/tinystories
    python eval.py sample     --ckpt runs/exp1/best.pt --prompt "Once upon a time" \
                              --max-tokens 100 --top-k 50
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import tiktoken
import torch

from data import TokenDataset
from train import _AMP_DTYPES, evaluate


def load_checkpoint(ckpt_path: str | Path, device: str | None = None):
    # Local import to avoid any circular shenanigans during test monkey-patching.
    from model import TransformerLM

    ckpt = torch.load(ckpt_path, weights_only=False, map_location="cpu")
    config = ckpt["config"]
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerLM(config.model).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, config, device


def compute_perplexity(
    ckpt_path: str | Path,
    data_dir: str | Path,
    n_iters: int = 200,
    batch_size: int = 16,
    device: str | None = None,
) -> float:
    model, config, device = load_checkpoint(ckpt_path, device)
    val_ds = TokenDataset(Path(data_dir) / "val.bin")
    amp_dtype = _AMP_DTYPES[config.train.dtype]
    rng = np.random.default_rng(42)
    mean_loss = evaluate(
        model, val_ds,
        n_iters=n_iters,
        batch_size=batch_size,
        context_len=config.model.context_len,
        device=device,
        amp_dtype=amp_dtype,
        rng=rng,
    )
    return math.exp(mean_loss)


def sample(
    ckpt_path: str | Path,
    prompt: str = "",
    max_tokens: int = 100,
    temperature: float = 1.0,
    top_k: int | None = None,
    device: str | None = None,
) -> str:
    enc = tiktoken.get_encoding("gpt2")
    model, _, device = load_checkpoint(ckpt_path, device)

    token_ids = enc.encode_ordinary(prompt) if prompt else [enc.encode_ordinary("\n")[0]]
    seed = torch.tensor([token_ids], dtype=torch.long, device=device)
    out = model.generate(
        seed,
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_k=top_k,
    )
    return enc.decode(out[0].tolist())


def _main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    ppl = sub.add_parser("perplexity", help="Validation perplexity from a checkpoint")
    ppl.add_argument("--ckpt", required=True)
    ppl.add_argument("--data", default="data/tinystories")
    ppl.add_argument("--n-iters", type=int, default=200)
    ppl.add_argument("--batch-size", type=int, default=16)

    smp = sub.add_parser("sample", help="Generate text from a checkpoint")
    smp.add_argument("--ckpt", required=True)
    smp.add_argument("--prompt", default="")
    smp.add_argument("--max-tokens", type=int, default=100)
    smp.add_argument("--temperature", type=float, default=1.0)
    smp.add_argument("--top-k", type=int, default=None)

    args = p.parse_args()
    if args.cmd == "perplexity":
        ppl_val = compute_perplexity(args.ckpt, args.data, args.n_iters, args.batch_size)
        print(f"val perplexity: {ppl_val:.3f}")
    elif args.cmd == "sample":
        text = sample(args.ckpt, args.prompt, args.max_tokens, args.temperature, args.top_k)
        print(text)


if __name__ == "__main__":
    _main()

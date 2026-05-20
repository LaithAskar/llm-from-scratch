"""
Generate text from a trained checkpoint. Sanity-check that "the model works."

CLI:
    python generate_sample.py runs/ablation/baseline/seed_0/best.pt
    python generate_sample.py path/to.pt --prompt "Once upon a time" --tokens 200
"""

from __future__ import annotations

import argparse

import tiktoken
import torch

from config import Config, ModelConfig
from model import TransformerLM


def load_model(ckpt_path: str, device: str) -> tuple[TransformerLM, Config]:
    blob = torch.load(ckpt_path, map_location=device, weights_only=False)
    config: Config = blob["config"]
    if not isinstance(config.model, ModelConfig):
        config.model = ModelConfig(**config.model)  # type: ignore[arg-type]
    model = TransformerLM(config.model).to(device)
    model.load_state_dict(blob["model"])
    model.eval()
    return model, config


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt", help="Path to a .pt checkpoint")
    p.add_argument("--prompt", default="Once upon a time")
    p.add_argument("--tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    enc = tiktoken.get_encoding("gpt2")
    model, config = load_model(args.ckpt, device)

    print(f"loaded {args.ckpt}")
    print(f"model: n_layer={config.model.n_layer} d_model={config.model.d_model} "
          f"variant={config.name}")
    print(f"prompt: {args.prompt!r}")
    print("---")

    ids = enc.encode_ordinary(args.prompt)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(
        x,
        max_new_tokens=args.tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    text = enc.decode(out[0].tolist())
    print(text)


if __name__ == "__main__":
    main()

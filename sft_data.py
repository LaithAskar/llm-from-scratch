"""
SFT data pipeline: wrap each TinyStories example with a fixed-prefix template,
tokenize, and write parallel (tokens, mask) bin files.

Template:
    "Here is a story:\\n\\n<story_text><|endoftext|>"

Loss mask: 0 on the prompt prefix tokens, 1 on the story body + <|endoftext|>.
At training time, targets are derived from tokens (shifted by 1) and positions
with mask=0 are set to -100 (ignored by cross_entropy).

Why a fixed prefix:
    The pretrained base has never seen this exact prefix. After SFT,
    prompting with the prefix should reliably elicit a story. This is the
    minimal demonstration of (a) loss masking and (b) format following
    that does not require an instruction-tuned source dataset.

Input format:
    Stories in the raw TinyStories text file are separated by a special
    line "<|endoftext|>" (in the V2 release) or a blank line. We detect
    both. Each block of non-empty lines between separators is one story.

CLI:
    python sft_data.py prepare \
        --raw data/tinystories/train.txt \
        --out data/sft \
        --max-stories 50000   # subset for fast SFT

Outputs (under --out):
    train_tokens.bin   uint16 flat array
    train_mask.bin     uint8  flat array (same length as train_tokens.bin)
    val_tokens.bin     uint16
    val_mask.bin       uint8
    meta.json          {"prefix": "...", "prefix_n_tokens": ..., "n_train": ..., "n_val": ...}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tiktoken

PREFIX = "Here is a story:\n\n"
EOT_TOKEN = "<|endoftext|>"  # tiktoken gpt2 id 50256


def iter_stories(raw_path: Path):
    """Yield one story (str) at a time from a TinyStories text file.

    Stories in TinyStoriesV2 are separated by lines containing literally
    '<|endoftext|>'. Older releases used blank lines. We support both.
    """
    buf: list[str] = []
    with open(raw_path, encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped == "<|endoftext|>":
                if buf:
                    yield "".join(buf).strip()
                    buf = []
            else:
                buf.append(line)
    if buf:
        tail = "".join(buf).strip()
        if tail:
            yield tail


def build_split(
    raw_path: Path,
    out_tokens: Path,
    out_mask: Path,
    enc: tiktoken.Encoding,
    prefix_ids: list[int],
    eot_id: int,
    max_stories: int | None,
) -> int:
    """Stream stories -> tokens/mask -> bin files. Returns n stories written."""
    out_tokens.parent.mkdir(parents=True, exist_ok=True)
    tok_chunks: list[np.ndarray] = []
    mask_chunks: list[np.ndarray] = []
    n = 0
    for story in iter_stories(raw_path):
        body_ids = enc.encode_ordinary(story)
        full_ids = prefix_ids + body_ids + [eot_id]
        # mask: 0 on prefix, 1 on body + eot
        mask = np.concatenate([
            np.zeros(len(prefix_ids), dtype=np.uint8),
            np.ones(len(body_ids) + 1, dtype=np.uint8),
        ])
        tok_chunks.append(np.array(full_ids, dtype=np.uint16))
        mask_chunks.append(mask)
        n += 1
        if max_stories is not None and n >= max_stories:
            break
    tokens = np.concatenate(tok_chunks)
    mask = np.concatenate(mask_chunks)
    assert tokens.shape == mask.shape
    tokens.tofile(out_tokens)
    mask.tofile(out_mask)
    print(f"  {raw_path.name}: {n:,} stories -> {tokens.size:,} tokens "
          f"-> {out_tokens.name}, {out_mask.name}")
    return n


def prepare(
    raw_dir: str,
    out_dir: str,
    max_stories_train: int | None,
    max_stories_val: int | None,
) -> None:
    raw = Path(raw_dir)
    out = Path(out_dir)
    enc = tiktoken.get_encoding("gpt2")
    prefix_ids = enc.encode_ordinary(PREFIX)
    # tiktoken treats <|endoftext|> as a special token; allow_special is needed.
    eot_id = enc.encode(EOT_TOKEN, allowed_special={EOT_TOKEN})[0]

    n_train = build_split(
        raw / "train.txt", out / "train_tokens.bin", out / "train_mask.bin",
        enc, prefix_ids, eot_id, max_stories_train,
    )
    n_val = build_split(
        raw / "val.txt", out / "val_tokens.bin", out / "val_mask.bin",
        enc, prefix_ids, eot_id, max_stories_val,
    )

    meta = {
        "prefix": PREFIX,
        "prefix_n_tokens": len(prefix_ids),
        "prefix_ids": prefix_ids,
        "eot_id": eot_id,
        "n_train_stories": n_train,
        "n_val_stories": n_val,
        "encoding": "gpt2",
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"  meta -> {out / 'meta.json'}")


class SFTDataset:
    """Memory-mapped view of (tokens.bin, mask.bin) for one split."""

    def __init__(self, tokens_path: str | Path, mask_path: str | Path):
        self.tokens_path = Path(tokens_path)
        self.mask_path = Path(mask_path)
        if not self.tokens_path.exists() or not self.mask_path.exists():
            raise FileNotFoundError(
                f"missing {self.tokens_path} or {self.mask_path} — "
                f"run: python sft_data.py prepare"
            )
        self.tokens = np.memmap(self.tokens_path, dtype=np.uint16, mode="r")
        self.mask = np.memmap(self.mask_path, dtype=np.uint8, mode="r")
        if self.tokens.size != self.mask.size:
            raise ValueError(
                f"tokens ({self.tokens.size}) and mask ({self.mask.size}) length mismatch"
            )

    def __len__(self) -> int:
        return self.tokens.size


def get_sft_batch(
    dataset: SFTDataset,
    batch_size: int,
    context_len: int,
    device: str,
    rng: np.random.Generator | None = None,
):
    """
    Sample (x, y, loss_mask) where:
        x         : (B, T) int64, input tokens
        y         : (B, T) int64, shifted-by-1 targets with -100 on prompt positions
        loss_mask : (B, T) bool, mirrors which positions contribute to loss
                    (kept around for diagnostics/tests; loss is enforced via -100 in y)

    Random starting positions over the concatenated stream (sample-with-replacement,
    matching the pretrain pipeline).
    """
    import torch
    if rng is None:
        rng = np.random.default_rng()
    n = dataset.tokens.size
    if n < context_len + 1:
        raise ValueError(f"dataset has {n} tokens, need at least {context_len + 1}")

    starts = rng.integers(0, n - context_len - 1, size=batch_size)
    x_np = np.stack([np.asarray(dataset.tokens[s : s + context_len], dtype=np.int64) for s in starts])
    y_np = np.stack([np.asarray(dataset.tokens[s + 1 : s + 1 + context_len], dtype=np.int64) for s in starts])
    # mask aligns with TARGETS (the predicted token at position t is y[t]).
    m_np = np.stack([np.asarray(dataset.mask[s + 1 : s + 1 + context_len], dtype=np.bool_) for s in starts])

    # Set ignored target positions to -100 so F.cross_entropy(ignore_index=-100) skips them.
    y_np[~m_np] = -100

    x = torch.from_numpy(x_np)
    y = torch.from_numpy(y_np)
    m = torch.from_numpy(m_np)
    if device.startswith("cuda"):
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
        m = m.pin_memory().to(device, non_blocking=True)
    else:
        x = x.to(device)
        y = y.to(device)
        m = m.to(device)
    return x, y, m


def _main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    prep = sub.add_parser("prepare", help="Tokenize TinyStories into SFT format")
    prep.add_argument("--raw", default="data/tinystories", help="dir containing train.txt and val.txt")
    prep.add_argument("--out", default="data/sft", help="output dir")
    prep.add_argument("--max-stories-train", type=int, default=50000,
                      help="cap on number of training stories (default 50k)")
    prep.add_argument("--max-stories-val", type=int, default=2000)

    args = p.parse_args()
    if args.cmd == "prepare":
        prepare(args.raw, args.out, args.max_stories_train, args.max_stories_val)


if __name__ == "__main__":
    _main()

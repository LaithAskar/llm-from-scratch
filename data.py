"""
Data pipeline: download text -> tokenize with tiktoken GPT-2 BPE -> uint16 bin.

CLI:
    python data.py prepare                                  # default: TinyStories
    python data.py prepare --out data/custom --source x.txt # use a local file

The bin format is a flat uint16 array (GPT-2 vocab = 50257 < 65536) memory-mapped
at training time so the dataset doesn't need to fit in RAM.

Batches are sampled with random starting positions (sample-with-replacement).
No epoch bookkeeping — for sub-1-epoch training on a small dataset this is
standard practice (see nanoGPT).
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

import numpy as np
import tiktoken
import torch
from tqdm import tqdm

TINYSTORIES_URLS = {
    "train": "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt",
    "val": "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt",
}


def _download_with_progress(url: str, dest: Path) -> None:
    """urllib + tqdm progress bar (urlretrieve has no progress hook with bytes)."""
    with urllib.request.urlopen(url) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        with open(dest, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=dest.name
        ) as bar:
            while chunk := resp.read(1024 * 1024):
                f.write(chunk)
                bar.update(len(chunk))


def download_tinystories(out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for split, url in TINYSTORIES_URLS.items():
        dest = out_dir / f"{split}.txt"
        paths[split] = dest
        if dest.exists():
            print(f"  {dest} already exists, skipping download")
            continue
        print(f"  downloading -> {dest}")
        _download_with_progress(url, dest)
    return paths


def tokenize_file(text_path: Path, bin_path: Path, encoder: tiktoken.Encoding) -> int:
    """Tokenize text_path with encoder, write uint16 array to bin_path. Returns token count."""
    bin_path.parent.mkdir(parents=True, exist_ok=True)
    text = text_path.read_text(encoding="utf-8")
    # encode_ordinary: skip the special-token check (no BOS/EOS in raw text).
    tokens = encoder.encode_ordinary(text)
    if max(tokens) >= 2**16:
        raise ValueError(
            f"Token id {max(tokens)} exceeds uint16 range; use a smaller-vocab encoding."
        )
    arr = np.array(tokens, dtype=np.uint16)
    arr.tofile(bin_path)
    return len(tokens)


def prepare(out_dir: str, source: str | None = None) -> None:
    out = Path(out_dir)
    enc = tiktoken.get_encoding("gpt2")  # vocab_size = 50257

    if source is None:
        paths = download_tinystories(out)
        for split, src in paths.items():
            bin_path = out / f"{split}.bin"
            n = tokenize_file(src, bin_path, enc)
            print(f"  {split}: {n:,} tokens -> {bin_path}")
    else:
        src = Path(source)
        if not src.exists():
            raise FileNotFoundError(src)
        text = src.read_text(encoding="utf-8")
        cut = int(len(text) * 0.9)
        out.mkdir(parents=True, exist_ok=True)
        (out / "train.txt").write_text(text[:cut], encoding="utf-8")
        (out / "val.txt").write_text(text[cut:], encoding="utf-8")
        n_train = tokenize_file(out / "train.txt", out / "train.bin", enc)
        n_val = tokenize_file(out / "val.txt", out / "val.bin", enc)
        print(f"  train: {n_train:,} tokens -> {out / 'train.bin'}")
        print(f"  val:   {n_val:,} tokens -> {out / 'val.bin'}")


class TokenDataset:
    """Memory-mapped view of a uint16 bin file. Read-only."""

    def __init__(self, bin_path: str | Path):
        self.bin_path = Path(bin_path)
        if not self.bin_path.exists():
            raise FileNotFoundError(
                f"{self.bin_path} not found. Run: "
                f"python data.py prepare --out {self.bin_path.parent}"
            )
        self.data = np.memmap(self.bin_path, dtype=np.uint16, mode="r")

    def __len__(self) -> int:
        return len(self.data)


def get_batch(
    dataset: TokenDataset,
    batch_size: int,
    context_len: int,
    device: str,
    rng: np.random.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample (x, y) where y is x shifted by 1 (next-token prediction targets).
    Random starting positions from the token stream.

    Shapes: x, y are both (batch_size, context_len), dtype int64.
    """
    if rng is None:
        rng = np.random.default_rng()
    n = len(dataset)
    if n < context_len + 1:
        raise ValueError(f"dataset has {n} tokens, need at least context_len+1 = {context_len + 1}")

    starts = rng.integers(0, n - context_len - 1, size=batch_size)
    # Cast to int64 here so torch doesn't have to.
    x = np.stack([np.asarray(dataset.data[s : s + context_len], dtype=np.int64) for s in starts])
    y = np.stack([np.asarray(dataset.data[s + 1 : s + 1 + context_len], dtype=np.int64) for s in starts])

    x_t = torch.from_numpy(x)
    y_t = torch.from_numpy(y)
    if device.startswith("cuda"):
        x_t = x_t.pin_memory().to(device, non_blocking=True)
        y_t = y_t.pin_memory().to(device, non_blocking=True)
    else:
        x_t = x_t.to(device)
        y_t = y_t.to(device)
    return x_t, y_t


def _main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    prep = sub.add_parser("prepare", help="Download and tokenize dataset")
    prep.add_argument("--out", default="data/tinystories")
    prep.add_argument(
        "--source",
        default=None,
        help="Local text file (90/10 train/val split). If omitted, downloads TinyStories.",
    )

    args = p.parse_args()
    if args.cmd == "prepare":
        prepare(args.out, args.source)


if __name__ == "__main__":
    _main()

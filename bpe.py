"""
Byte-Pair Encoding tokenizer, trained from scratch.

Sennrich, Haddush, Birch (2015). The algorithm:
  1. Pre-tokenize text into "words" using a regex (preserves whitespace,
     prevents merges across word boundaries).
  2. Initialize vocab with all 256 bytes.
  3. Repeat until vocab_size reached:
     a. Count adjacent (token_a, token_b) pairs across all words.
     b. Find the most-frequent pair.
     c. Add a new token = token_a + token_b. Record the merge.
     d. Replace all (token_a, token_b) sequences in all words with the
        new token.

Encoding new text: apply the same merges in their original learned order
(lowest-index merges first — "greedy lowest-priority" decode).

The ablation keeps using tiktoken's GPT-2 BPE so vocab is held constant
across variants. This module is a standalone artifact demonstrating the
training algorithm; the trained tokenizer here is NOT vocab-compatible
with GPT-2 BPE.

CLI:
    python bpe.py train --corpus path/to/text.txt --vocab-size 4096 --out bpe.json
    python bpe.py encode --tokenizer bpe.json --text "hello world"
    python bpe.py decode --tokenizer bpe.json --ids 0,1,2,3
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import regex as re

# GPT-2's pre-tokenization regex. Splits text into "words" before BPE so
# that merges don't cross natural boundaries. Unicode property classes
# (\p{L}, \p{N}) require the `regex` module — stdlib `re` doesn't support
# them as of Python 3.13.
_GPT2_PRETOKEN = re.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


def pretokenize(text: str) -> list[bytes]:
    """Split text into pre-tokens, each encoded as UTF-8 bytes."""
    return [m.group(0).encode("utf-8") for m in _GPT2_PRETOKEN.finditer(text)]


def _count_pairs(word_freqs: dict[tuple[bytes, ...], int]) -> dict[tuple[bytes, bytes], int]:
    """Count adjacent (token_a, token_b) pairs across all words, weighted by freq."""
    counts: dict[tuple[bytes, bytes], int] = collections.defaultdict(int)
    for word, freq in word_freqs.items():
        for i in range(len(word) - 1):
            counts[(word[i], word[i + 1])] += freq
    return counts


def _apply_merge(word: tuple[bytes, ...], pair: tuple[bytes, bytes]) -> tuple[bytes, ...]:
    """Replace every occurrence of `pair` in `word` with the merged token."""
    if len(word) < 2:
        return word
    out: list[bytes] = []
    i = 0
    while i < len(word):
        if i < len(word) - 1 and (word[i], word[i + 1]) == pair:
            out.append(word[i] + word[i + 1])
            i += 2
        else:
            out.append(word[i])
            i += 1
    return tuple(out)


def train_bpe(
    corpus: str,
    vocab_size: int,
    verbose: bool = False,
) -> tuple[list[tuple[bytes, bytes]], dict[bytes, int]]:
    """
    Train a BPE tokenizer on the given corpus.

    Returns:
        merges: list of (token_a, token_b) tuples in the order they were learned.
        vocab:  dict mapping each bytes-token to its integer ID.

    `vocab_size` must be >= 256 (the initial byte-level vocabulary).
    """
    if vocab_size < 256:
        raise ValueError(f"vocab_size must be >= 256 (initial bytes); got {vocab_size}")

    # Pre-tokenize and count word frequencies.
    words = pretokenize(corpus)
    word_freqs = collections.Counter(words)

    # Each word starts as a tuple of single-byte tokens.
    word_freqs_split: dict[tuple[bytes, ...], int] = {
        tuple(bytes([b]) for b in word): freq
        for word, freq in word_freqs.items()
    }

    # Initial vocab: all 256 bytes.
    vocab: dict[bytes, int] = {bytes([i]): i for i in range(256)}
    merges: list[tuple[bytes, bytes]] = []

    n_merges = vocab_size - len(vocab)
    for step in range(n_merges):
        pair_counts = _count_pairs(word_freqs_split)
        if not pair_counts:
            break  # No more pairs — corpus fully merged.

        best_pair, best_count = max(pair_counts.items(), key=lambda kv: kv[1])
        merged = best_pair[0] + best_pair[1]
        vocab[merged] = len(vocab)
        merges.append(best_pair)

        if verbose and step % 100 == 0:
            print(f"  step {step:5d}: merged {best_pair!r} ({best_count} occurrences)")

        # Re-merge every affected word. Efficiency could be improved with
        # incremental pair-count updates, but for toy-scale corpora the
        # naive full re-merge is fine.
        word_freqs_split = {
            _apply_merge(word, best_pair): freq
            for word, freq in word_freqs_split.items()
        }

    return merges, vocab


def encode(
    text: str,
    merges: list[tuple[bytes, bytes]],
    vocab: dict[bytes, int],
) -> list[int]:
    """
    Encode text into token IDs using a trained BPE.

    Greedy lowest-merge-index strategy: at each step, find the pair in the
    current token sequence with the smallest merge index (= earliest-learned
    merge), apply it, repeat. This matches tiktoken and HuggingFace BPE
    decoders.
    """
    merge_rank = {pair: i for i, pair in enumerate(merges)}
    out: list[int] = []

    for word in pretokenize(text):
        tokens: list[bytes] = [bytes([b]) for b in word]

        while len(tokens) > 1:
            # Find the pair with the lowest merge rank.
            best_rank = len(merges)  # sentinel = "not a learned merge"
            best_pos = -1
            for i in range(len(tokens) - 1):
                rank = merge_rank.get((tokens[i], tokens[i + 1]), len(merges))
                if rank < best_rank:
                    best_rank = rank
                    best_pos = i
            if best_pos < 0:
                break
            tokens = (
                tokens[:best_pos]
                + [tokens[best_pos] + tokens[best_pos + 1]]
                + tokens[best_pos + 2:]
            )

        out.extend(vocab[t] for t in tokens)

    return out


def decode(ids: list[int], vocab: dict[bytes, int]) -> str:
    """Decode token IDs back to text. Uses errors='replace' for partial UTF-8."""
    id_to_token = {v: k for k, v in vocab.items()}
    return b"".join(id_to_token[i] for i in ids).decode("utf-8", errors="replace")


# --- Serialization (JSON with hex-encoded bytes) ---------------------------


def save(path: str | Path, merges: list[tuple[bytes, bytes]], vocab: dict[bytes, int]) -> None:
    """
    Save tokenizer to a JSON file.

    Bytes-keyed dicts and bytes-tuple lists don't serialize natively to
    JSON, so we encode every bytes value as a hex string.
    """
    data = {
        "merges": [[a.hex(), b.hex()] for a, b in merges],
        "vocab": {tok.hex(): idx for tok, idx in vocab.items()},
    }
    Path(path).write_text(json.dumps(data, indent=2))


def load(path: str | Path) -> tuple[list[tuple[bytes, bytes]], dict[bytes, int]]:
    """Load tokenizer from a JSON file written by save()."""
    data = json.loads(Path(path).read_text())
    merges = [(bytes.fromhex(a), bytes.fromhex(b)) for a, b in data["merges"]]
    vocab = {bytes.fromhex(tok): idx for tok, idx in data["vocab"].items()}
    return merges, vocab


# --- CLI -------------------------------------------------------------------


def _cmd_train(args):
    corpus_text = Path(args.corpus).read_text(encoding="utf-8")
    if args.max_chars:
        corpus_text = corpus_text[: args.max_chars]
    print(f"Training BPE: vocab_size={args.vocab_size}, "
          f"corpus={len(corpus_text):,} chars")
    merges, vocab = train_bpe(corpus_text, args.vocab_size, verbose=True)
    save(args.out, merges, vocab)
    print(f"\nWrote {args.out}: {len(merges)} merges, {len(vocab)} vocab tokens")

    # Compression diagnostic on the training text.
    encoded = encode(corpus_text[:10_000], merges, vocab)
    bytes_in = len(corpus_text[:10_000].encode("utf-8"))
    print(f"Compression on first 10K chars: {bytes_in} bytes -> {len(encoded)} tokens "
          f"({bytes_in / len(encoded):.2f} bytes/token)")


def _cmd_encode(args):
    merges, vocab = load(args.tokenizer)
    print(encode(args.text, merges, vocab))


def _cmd_decode(args):
    _, vocab = load(args.tokenizer)
    ids = [int(s) for s in args.ids.split(",")]
    print(decode(ids, vocab))


def _main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    tr = sub.add_parser("train", help="Train a BPE tokenizer on a text corpus")
    tr.add_argument("--corpus", required=True, help="Path to a UTF-8 text file")
    tr.add_argument("--vocab-size", type=int, default=4096)
    tr.add_argument("--out", default="bpe.json")
    tr.add_argument("--max-chars", type=int, default=None,
                    help="Truncate corpus to first N chars (for fast iteration)")
    tr.set_defaults(func=_cmd_train)

    en = sub.add_parser("encode")
    en.add_argument("--tokenizer", required=True)
    en.add_argument("--text", required=True)
    en.set_defaults(func=_cmd_encode)

    de = sub.add_parser("decode")
    de.add_argument("--tokenizer", required=True)
    de.add_argument("--ids", required=True, help="Comma-separated token IDs")
    de.set_defaults(func=_cmd_decode)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    _main()

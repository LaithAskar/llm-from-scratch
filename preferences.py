"""
Preference-data helpers for Part 7 (reward modeling / RLAIF-style data).

This module is intentionally small and file-format-first. A real RLAIF run can
fill the JSONL file with model-generated completions and AI-judge preferences;
the training code only needs token IDs for prompt/chosen/rejected.

JSONL schema per row:
    {"prompt_ids": [...], "chosen_ids": [...], "rejected_ids": [...]}

`chosen_ids` and `rejected_ids` are completions only. Collation concatenates
prompt + completion so a reward model can score the whole sequence.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch


@dataclass(frozen=True)
class PreferenceExample:
    prompt_ids: list[int]
    chosen_ids: list[int]
    rejected_ids: list[int]

    def chosen_sequence(self) -> list[int]:
        return self.prompt_ids + self.chosen_ids

    def rejected_sequence(self) -> list[int]:
        return self.prompt_ids + self.rejected_ids


class PreferenceDataset:
    """Load preference triples from JSONL."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        self.examples = [self._parse_line(line, i + 1) for i, line in enumerate(self.path.read_text().splitlines()) if line.strip()]
        if not self.examples:
            raise ValueError(f"no preference rows found in {self.path}")

    @staticmethod
    def _parse_line(line: str, line_no: int) -> PreferenceExample:
        row = json.loads(line)
        required = ["prompt_ids", "chosen_ids", "rejected_ids"]
        missing = [k for k in required if k not in row]
        if missing:
            raise ValueError(f"line {line_no}: missing keys {missing}")
        ex = PreferenceExample(
            prompt_ids=[int(x) for x in row["prompt_ids"]],
            chosen_ids=[int(x) for x in row["chosen_ids"]],
            rejected_ids=[int(x) for x in row["rejected_ids"]],
        )
        if not ex.prompt_ids or not ex.chosen_ids or not ex.rejected_ids:
            raise ValueError(f"line {line_no}: prompt/chosen/rejected must all be non-empty")
        return ex

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> PreferenceExample:
        return self.examples[idx]


def _pad(seqs: list[list[int]], pad_id: int, max_len: int | None) -> tuple[torch.Tensor, torch.Tensor]:
    lengths = torch.tensor([min(len(s), max_len or len(s)) for s in seqs], dtype=torch.long)
    width = int(lengths.max().item()) if max_len is None else max_len
    out = torch.full((len(seqs), width), pad_id, dtype=torch.long)
    for i, seq in enumerate(seqs):
        clipped = seq[:width]
        out[i, : len(clipped)] = torch.tensor(clipped, dtype=torch.long)
    return out, lengths


def collate_preferences(
    examples: Iterable[PreferenceExample],
    pad_id: int = 0,
    max_len: int | None = None,
) -> dict[str, torch.Tensor]:
    """Pad a batch into chosen/rejected tensors plus true sequence lengths."""
    batch = list(examples)
    if not batch:
        raise ValueError("cannot collate an empty preference batch")
    chosen_seqs = [ex.chosen_sequence() for ex in batch]
    rejected_seqs = [ex.rejected_sequence() for ex in batch]
    if max_len is None:
        max_len = max(max(map(len, chosen_seqs)), max(map(len, rejected_seqs)))
    chosen, chosen_lengths = _pad(chosen_seqs, pad_id, max_len)
    rejected, rejected_lengths = _pad(rejected_seqs, pad_id, max_len)
    return {
        "chosen": chosen,
        "rejected": rejected,
        "chosen_lengths": chosen_lengths,
        "rejected_lengths": rejected_lengths,
    }


def pick_preference_from_scores(
    prompt_ids: list[int],
    completions: list[list[int]],
    scores: list[float],
) -> PreferenceExample:
    """Build one preference triple by taking max-score vs min-score completion.

    This is useful for RLAIF data generation after an external judge scores K
    candidate completions. Ties are rejected because they carry no preference
    signal.
    """
    if len(completions) != len(scores):
        raise ValueError("completions and scores must have the same length")
    if len(completions) < 2:
        raise ValueError("need at least two completions to form a preference")
    best = max(range(len(scores)), key=lambda i: scores[i])
    worst = min(range(len(scores)), key=lambda i: scores[i])
    if scores[best] == scores[worst]:
        raise ValueError("all scores are tied; no preference signal")
    return PreferenceExample(prompt_ids, completions[best], completions[worst])

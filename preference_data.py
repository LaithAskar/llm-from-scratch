"""
Preference data pipeline for the reward model (Part 7).

Flow:
    1. Roll out K completions per prompt from the SFT policy.
    2. For each pair of completions, a PreferenceOracle picks the better one.
    3. Emit (prompt, chosen, rejected) triples as JSONL.

The oracle is an interface with two implementations:

    HeuristicOracle  — a transparent, deterministic scoring function. No API,
                       no spend. Used to test and demonstrate the whole RM
                       pipeline end-to-end *today*. The RM trained against it
                       genuinely learns to rank, but it is ranking against a
                       hand-written heuristic, NOT real LLM judgement.

    ApiOracle        — asks Claude (or OpenAI) which completion is better.
                       This is the real RLAIF signal. Inert until an API key
                       exists; enforces a hard spend cap.

Honest framing: until ApiOracle is wired with a key, "Part 7" demonstrates
a complete, tested RM pipeline whose preference signal is a heuristic
placeholder. Swapping in real LLM feedback is a one-line oracle change.

CLI:
    python preference_data.py rollout \
        --policy runs/sft/modern/best.pt \
        --out data/prefs/pairs.jsonl \
        --n-prompts 64 --k 4 --oracle heuristic
"""

from __future__ import annotations

import argparse
import json
import os
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import tiktoken
import torch

from config import ModelConfig
from model import TransformerLM

# Seed prompts. All start with the SFT prefix so the SFT'd policy is on
# distribution; the trailing fragment nudges different story openings so
# rollouts have content variety for the RM to discriminate on.
STORY_SEED_PROMPTS = [
    "Here is a story:\n\n",
    "Here is a story:\n\nOnce upon a time",
    "Here is a story:\n\nOne day",
    "Here is a story:\n\nThere was a",
    "Here is a story:\n\nA little",
    "Here is a story:\n\nThe sun was",
    "Here is a story:\n\nTom and",
    "Here is a story:\n\nIn a big",
]

EOT_ID = 50256  # tiktoken gpt2 <|endoftext|>


# --------------------------------------------------------------------------
# Heuristic quality score
# --------------------------------------------------------------------------

def quality_score(completion_ids: list[int], eot_id: int = EOT_ID) -> float:
    """
    Transparent quality heuristic for a TinyStories completion. Two axes,
    both things the toy model genuinely gets wrong:

      distinct-2 : fraction of distinct bigrams. Repetitive babble ("He was
                   happy. He was happy.") scores low; varied text scores high.
      stray-EOT  : <|endoftext|> appearing before the final position means
                   the model failed to produce one clean story. Penalised.

    Returns a float; higher is better. Used only to *compare* two completions
    of the same prompt, so the absolute scale is irrelevant.
    """
    n = len(completion_ids)
    if n < 2:
        return -10.0
    bigrams = list(zip(completion_ids, completion_ids[1:]))
    distinct2 = len(set(bigrams)) / len(bigrams)
    stray_eot = sum(1 for t in completion_ids[:-1] if t == eot_id)
    return distinct2 - 0.5 * stray_eot


# --------------------------------------------------------------------------
# Oracle interface
# --------------------------------------------------------------------------

class PreferenceOracle(ABC):
    """Given a prompt and two completions, decide which completion is better."""

    @abstractmethod
    def prefer(
        self,
        prompt_ids: list[int],
        comp_a_ids: list[int],
        comp_b_ids: list[int],
    ) -> int:
        """Return 0 if A is preferred, 1 if B is preferred, -1 to skip (tie /
        unusable). Skipped pairs are not added to the dataset."""
        ...


class HeuristicOracle(PreferenceOracle):
    """Deterministic: prefers the completion with the higher quality_score.
    Ties (equal score) are skipped to keep the training signal clean."""

    def __init__(self, eot_id: int = EOT_ID, tie_eps: float = 1e-9):
        self.eot_id = eot_id
        self.tie_eps = tie_eps

    def prefer(self, prompt_ids, comp_a_ids, comp_b_ids) -> int:
        sa = quality_score(comp_a_ids, self.eot_id)
        sb = quality_score(comp_b_ids, self.eot_id)
        if abs(sa - sb) < self.tie_eps:
            return -1
        return 0 if sa > sb else 1


class ApiOracle(PreferenceOracle):
    """
    Asks an LLM which completion is the better children's story. This is the
    real RLAIF signal. Inert until an API key is present.

    Enforces a hard spend cap: every call increments an estimated-cost
    counter; once it would exceed `max_spend_usd` the oracle raises instead
    of making the call.
    """

    # rough per-call cost estimate (one short judgement prompt). Deliberately
    # conservative so the cap triggers early rather than late.
    _EST_COST_PER_CALL_USD = 0.002

    def __init__(
        self,
        provider: str = "anthropic",
        model: str = "claude-haiku-4-5",
        max_spend_usd: float = 5.0,
        decoder: tiktoken.Encoding | None = None,
    ):
        self.provider = provider
        self.model = model
        self.max_spend_usd = max_spend_usd
        self.spent_usd = 0.0
        self.n_calls = 0
        self._decoder = decoder or tiktoken.get_encoding("gpt2")
        self._client = None  # lazily constructed on first call

    def _ensure_client(self):
        if self._client is not None:
            return
        if self.provider == "anthropic":
            if not os.environ.get("ANTHROPIC_API_KEY"):
                raise RuntimeError(
                    "ApiOracle: ANTHROPIC_API_KEY not set. Set a key, or use "
                    "HeuristicOracle for a no-API run."
                )
            try:
                import anthropic  # lazy: not a hard dependency of the repo
            except ImportError as e:
                raise RuntimeError(
                    "ApiOracle: `anthropic` package not installed. "
                    "pip install anthropic"
                ) from e
            self._client = anthropic.Anthropic()
        elif self.provider == "openai":
            if not os.environ.get("OPENAI_API_KEY"):
                raise RuntimeError("ApiOracle: OPENAI_API_KEY not set.")
            try:
                import openai
            except ImportError as e:
                raise RuntimeError("ApiOracle: `openai` package not installed.") from e
            self._client = openai.OpenAI()
        else:
            raise ValueError(f"unknown provider {self.provider!r}")

    def _judge_prompt(self, prompt_text: str, a_text: str, b_text: str) -> str:
        return (
            "You are judging two children's-story completions of the same "
            "prompt. Pick the one that is more coherent, less repetitive, and "
            "reads more like a real short story.\n\n"
            f"PROMPT:\n{prompt_text}\n\n"
            f"COMPLETION A:\n{a_text}\n\n"
            f"COMPLETION B:\n{b_text}\n\n"
            "Answer with exactly one character: A or B."
        )

    def prefer(self, prompt_ids, comp_a_ids, comp_b_ids) -> int:
        self._ensure_client()
        if self.spent_usd + self._EST_COST_PER_CALL_USD > self.max_spend_usd:
            raise RuntimeError(
                f"ApiOracle: spend cap ${self.max_spend_usd:.2f} reached "
                f"after {self.n_calls} calls (~${self.spent_usd:.4f}). "
                f"Raise --max-spend or stop here."
            )

        prompt_text = self._decoder.decode(prompt_ids)
        a_text = self._decoder.decode(comp_a_ids)
        b_text = self._decoder.decode(comp_b_ids)
        question = self._judge_prompt(prompt_text, a_text, b_text)

        if self.provider == "anthropic":
            msg = self._client.messages.create(
                model=self.model,
                max_tokens=4,
                messages=[{"role": "user", "content": question}],
            )
            answer = msg.content[0].text.strip().upper()
        else:  # openai
            resp = self._client.chat.completions.create(
                model=self.model,
                max_tokens=4,
                messages=[{"role": "user", "content": question}],
            )
            answer = resp.choices[0].message.content.strip().upper()

        self.n_calls += 1
        self.spent_usd += self._EST_COST_PER_CALL_USD

        if answer.startswith("A"):
            return 0
        if answer.startswith("B"):
            return 1
        return -1  # unparseable -> skip


def make_oracle(name: str, max_spend_usd: float = 5.0) -> PreferenceOracle:
    if name == "heuristic":
        return HeuristicOracle()
    if name in ("anthropic", "openai"):
        return ApiOracle(provider=name, max_spend_usd=max_spend_usd)
    raise ValueError(f"unknown oracle {name!r} (have: heuristic, anthropic, openai)")


# --------------------------------------------------------------------------
# Rollout
# --------------------------------------------------------------------------

def load_policy(ckpt_path: str, device: str) -> TransformerLM:
    blob = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = blob["config"]
    model_cfg = cfg.model if isinstance(cfg.model, ModelConfig) else ModelConfig(**cfg.model)
    model = TransformerLM(model_cfg).to(device)
    model.load_state_dict(blob["model"])
    model.eval()
    return model


@torch.no_grad()
def rollout_completions(
    policy: TransformerLM,
    enc: tiktoken.Encoding,
    prompt_text: str,
    k: int,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    device: str,
) -> tuple[list[int], list[list[int]]]:
    """Generate k completions for one prompt. Returns (prompt_ids, [completion_ids]*k).

    `completion_ids` are the *newly generated* tokens only (prompt stripped)."""
    prompt_ids = enc.encode_ordinary(prompt_text)
    x = torch.tensor([prompt_ids], dtype=torch.long, device=device).repeat(k, 1)  # (k, P)
    out = policy.generate(x, max_new_tokens=max_new_tokens,
                          temperature=temperature, top_k=top_k)               # (k, P+N)
    completions = [out[i, len(prompt_ids):].tolist() for i in range(k)]
    return prompt_ids, completions


def build_preference_dataset(
    policy_ckpt: str,
    out_path: str,
    oracle: PreferenceOracle,
    n_prompts: int,
    k: int,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    seed: int,
) -> dict:
    """Roll out, label with the oracle, write (prompt, chosen, rejected) JSONL.

    For each prompt we generate k completions and form all k*(k-1)/2 unordered
    pairs, label each with the oracle, and keep the non-skipped ones."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    np.random.seed(seed)
    enc = tiktoken.get_encoding("gpt2")
    policy = load_policy(policy_ckpt, device)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    n_pairs, n_skipped = 0, 0
    with open(out, "w", encoding="utf-8") as f:
        for i in range(n_prompts):
            prompt_text = STORY_SEED_PROMPTS[i % len(STORY_SEED_PROMPTS)]
            prompt_ids, comps = rollout_completions(
                policy, enc, prompt_text, k, max_new_tokens,
                temperature, top_k, device,
            )
            for a in range(k):
                for b in range(a + 1, k):
                    pref = oracle.prefer(prompt_ids, comps[a], comps[b])
                    if pref == -1:
                        n_skipped += 1
                        continue
                    chosen, rejected = (comps[a], comps[b]) if pref == 0 else (comps[b], comps[a])
                    f.write(json.dumps({
                        "prompt": prompt_ids,
                        "chosen": chosen,
                        "rejected": rejected,
                    }) + "\n")
                    n_pairs += 1
            if (i + 1) % 16 == 0:
                print(f"  {i + 1}/{n_prompts} prompts -> {n_pairs} pairs ({n_skipped} skipped)")

    stats = {"n_prompts": n_prompts, "k": k, "n_pairs": n_pairs, "n_skipped": n_skipped}
    if isinstance(oracle, ApiOracle):
        stats["oracle_calls"] = oracle.n_calls
        stats["oracle_spent_usd"] = round(oracle.spent_usd, 4)
    print(f"  wrote {n_pairs} preference pairs -> {out}")
    return stats


# --------------------------------------------------------------------------
# Dataset for RM training
# --------------------------------------------------------------------------

class PreferenceDataset:
    """Loads (prompt, chosen, rejected) JSONL. Each example is scored by the
    RM as prompt+completion concatenated."""

    def __init__(self, jsonl_path: str | Path, context_len: int):
        self.path = Path(jsonl_path)
        if not self.path.exists():
            raise FileNotFoundError(f"{self.path} — run: python preference_data.py rollout")
        self.context_len = context_len
        self.examples: list[dict] = []
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.examples.append(json.loads(line))
        if not self.examples:
            raise ValueError(f"{self.path} is empty")

    def __len__(self) -> int:
        return len(self.examples)

    def _seq(self, ex: dict, key: str) -> list[int]:
        # prompt + completion, truncated to context_len from the LEFT-anchored
        # start (keep the prompt, drop overflowing completion tail).
        return (ex["prompt"] + ex[key])[: self.context_len]


def collate_preference_batch(
    dataset: PreferenceDataset,
    indices: list[int],
    pad_id: int,
    device: str,
) -> dict:
    """Build a padded batch for one set of example indices.

    Returns dict with chosen/rejected token tensors (B,T) and length tensors (B,)."""
    chosen_seqs = [dataset._seq(dataset.examples[i], "chosen") for i in indices]
    rejected_seqs = [dataset._seq(dataset.examples[i], "rejected") for i in indices]
    all_seqs = chosen_seqs + rejected_seqs
    max_len = max(len(s) for s in all_seqs)

    def pad(seqs):
        ids = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
        lengths = torch.empty(len(seqs), dtype=torch.long)
        for r, s in enumerate(seqs):
            ids[r, : len(s)] = torch.tensor(s, dtype=torch.long)
            lengths[r] = len(s)
        return ids, lengths

    c_ids, c_len = pad(chosen_seqs)
    r_ids, r_len = pad(rejected_seqs)
    return {
        "chosen_ids": c_ids.to(device), "chosen_len": c_len.to(device),
        "rejected_ids": r_ids.to(device), "rejected_len": r_len.to(device),
    }


def _main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("rollout", help="Roll out completions + label preferences")
    r.add_argument("--policy", required=True, help="SFT checkpoint path")
    r.add_argument("--out", default="data/prefs/pairs.jsonl")
    r.add_argument("--n-prompts", type=int, default=64)
    r.add_argument("--k", type=int, default=4, help="completions per prompt")
    r.add_argument("--max-new-tokens", type=int, default=96)
    r.add_argument("--temperature", type=float, default=0.9)
    r.add_argument("--top-k", type=int, default=40)
    r.add_argument("--oracle", default="heuristic", choices=["heuristic", "anthropic", "openai"])
    r.add_argument("--max-spend", type=float, default=5.0, help="ApiOracle hard cap, USD")
    r.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if args.cmd == "rollout":
        oracle = make_oracle(args.oracle, max_spend_usd=args.max_spend)
        stats = build_preference_dataset(
            policy_ckpt=args.policy, out_path=args.out, oracle=oracle,
            n_prompts=args.n_prompts, k=args.k, max_new_tokens=args.max_new_tokens,
            temperature=args.temperature, top_k=args.top_k, seed=args.seed,
        )
        print(f"  stats: {stats}")


if __name__ == "__main__":
    _main()

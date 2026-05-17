"""
Configuration dataclasses for model architecture and training.

A `Config` fully specifies one ablation variant. Flipping `model.norm_type`,
`model.activation`, or `model.pos_encoding` defines a new variant; every
other field stays identical for a controlled comparison.

Configs serialize to JSON next to each checkpoint so any run can be replayed.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal, get_args

NormType = Literal["layernorm", "rmsnorm"]
Activation = Literal["gelu", "swiglu"]
PosEncoding = Literal["learned", "rope"]
Dtype = Literal["fp32", "bf16", "fp16"]


@dataclass
class ModelConfig:
    vocab_size: int = 50257  # tiktoken gpt2 BPE
    n_layer: int = 4
    n_head: int = 6
    d_model: int = 192  # head_dim = 32
    d_ffn: int | None = None  # see __post_init__ for default sizing
    context_len: int = 256
    dropout: float = 0.1
    bias: bool = False  # for QKV / out_proj / FFN linears

    # ablation switches
    norm_type: NormType = "layernorm"
    activation: Activation = "gelu"
    pos_encoding: PosEncoding = "learned"

    # rope-specific (ignored if pos_encoding != "rope")
    rope_base: float = 10000.0

    # norm-specific
    norm_eps: float = 1e-5

    def __post_init__(self):
        if self.d_model % self.n_head != 0:
            raise ValueError(
                f"d_model ({self.d_model}) not divisible by n_head ({self.n_head})"
            )
        if self.norm_type not in get_args(NormType):
            raise ValueError(f"norm_type must be one of {get_args(NormType)}, got {self.norm_type!r}")
        if self.activation not in get_args(Activation):
            raise ValueError(f"activation must be one of {get_args(Activation)}, got {self.activation!r}")
        if self.pos_encoding not in get_args(PosEncoding):
            raise ValueError(f"pos_encoding must be one of {get_args(PosEncoding)}, got {self.pos_encoding!r}")

        if self.d_ffn is None:
            # Match SwiGLU FFN params to GELU FFN with d_ffn = 4*d_model.
            # SwiGLU uses 3 matrices (gate, up, down) vs GELU's 2 (up, down),
            # so each SwiGLU matrix is sized to (8/3)*d_model to keep total
            # FFN param count comparable. Rounded to a multiple of 64.
            # Reference: Chowdhery et al., "PaLM" (2022), §3.
            if self.activation == "swiglu":
                self.d_ffn = int(round(8 / 3 * self.d_model / 64)) * 64
            else:
                self.d_ffn = 4 * self.d_model

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_head


@dataclass
class TrainConfig:
    # optimization
    lr: float = 3e-4
    min_lr: float = 3e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    # batching
    micro_batch_size: int = 16
    grad_accum_steps: int = 2  # effective batch = micro * grad_accum
    max_steps: int = 5000
    warmup_steps: int = 200

    # eval / checkpoint cadence (in steps)
    eval_every: int = 250
    eval_iters: int = 50
    ckpt_every: int = 1000
    log_every: int = 10

    # io
    out_dir: str = "runs/default"
    data_dir: str = "data/tinystories"

    # reproducibility
    seed: int = 0

    # mixed precision
    dtype: Dtype = "bf16"
    compile: bool = False  # torch.compile — opt-in (flaky on Windows)

    def __post_init__(self):
        if self.dtype not in get_args(Dtype):
            raise ValueError(f"dtype must be one of {get_args(Dtype)}, got {self.dtype!r}")
        if self.grad_accum_steps < 1:
            raise ValueError("grad_accum_steps must be >= 1")
        if self.warmup_steps >= self.max_steps:
            raise ValueError("warmup_steps must be < max_steps")

    @property
    def effective_batch_size(self) -> int:
        return self.micro_batch_size * self.grad_accum_steps


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    name: str = "default"
    notes: str = ""

    def to_json(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def from_json(cls, path: str | Path) -> "Config":
        data = json.loads(Path(path).read_text())
        return cls(
            model=ModelConfig(**data["model"]),
            train=TrainConfig(**data["train"]),
            name=data.get("name", "default"),
            notes=data.get("notes", ""),
        )


if __name__ == "__main__":
    # Smoke test: instantiate defaults, round-trip through JSON.
    import tempfile

    cfg = Config(name="smoke")
    assert cfg.model.head_dim == 32
    assert cfg.model.d_ffn == 4 * 192  # gelu default

    cfg_swiglu = Config(model=ModelConfig(activation="swiglu"))
    # (8/3) * 192 = 512, rounded to mult of 64 = 512
    assert cfg_swiglu.model.d_ffn == 512, f"expected 512, got {cfg_swiglu.model.d_ffn}"

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        path = f.name
    cfg.to_json(path)
    cfg2 = Config.from_json(path)
    assert cfg2.model.d_model == cfg.model.d_model
    assert cfg2.train.effective_batch_size == 32

    try:
        ModelConfig(norm_type="layernorrm")  # typo on purpose
    except ValueError as e:
        print(f"ok — caught typo: {e}")

    print("config.py smoke test passed.")

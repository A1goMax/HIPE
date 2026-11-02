"""Serializable execution settings. This module has no ML-runtime dependency."""
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
import json
import math


@dataclass
class ExperimentConfig:
    experiment_id: str = "manual"
    paper_source: str = "MS04"
    dataset: str = "wikitext103"
    model_size: str = "60M"
    method: str = "hipe"
    sigma: float = 500.0
    schedule: str = "bipartite"
    layer_sigmas: list | None = None
    layer_threshold: int | None = None
    learnable_sigma: bool = False
    attention_pattern: str = "global"
    local_window_size: int = 256
    local_layers: int = 4
    legacy_absolute_positions: bool = False
    seq_len: int = 1024
    seed: int = 6198
    global_batch_size: int = 64
    micro_batch_size: int = 8
    max_tokens: int = 100_000_000
    max_train_steps: int | None = None
    lr: float = 6e-4
    betas: list = field(default_factory=lambda: [0.9, 0.95])
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    eval_interval: int = 100
    checkpoint_interval: int = 1000
    precision: str = "bf16"
    attention_backend: str = "flash"
    device: str = "cuda"
    data_pipeline: str = "legacy_wiki"
    num_proc: int = 4
    validation_num_proc: int = 4
    max_train_samples: int | None = None
    max_val_samples: int | None = None
    max_val_batches: int = 0
    data_path: str = ""
    validation_path: str | None = None
    tokenizer_path: str = ""
    output_dir: str = ""
    batch_profile: str = "explicit"
    source_notes: list = field(default_factory=list)

    @classmethod
    def load(cls, path):
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        unknown = set(raw) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown experiment fields: {sorted(unknown)}")
        config = cls(**raw)
        config.validate()
        return config

    def validate(self):
        if self.method not in {"rope", "hipe", "nope", "alibi", "xpos"}:
            raise ValueError("Unknown method")
        if self.dataset not in {"wikitext103", "c4"}:
            raise ValueError("This trainer supports WikiText-103/C4")
        if self.model_size not in {"20M", "60M", "300M"}:
            raise ValueError("Unknown model size")
        for name in ["seq_len", "global_batch_size", "micro_batch_size", "eval_interval",
                     "checkpoint_interval", "num_proc", "validation_num_proc", "max_tokens"]:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.global_batch_size % self.micro_batch_size:
            raise ValueError("Global batch must be divisible by micro batch")
        if self.max_train_steps is not None and self.max_train_steps <= 0:
            raise ValueError("max_train_steps must be positive")
        if self.max_train_steps is None and self.max_tokens < self.global_batch_size * self.seq_len:
            raise ValueError("Token budget is smaller than one original global batch")
        if self.precision not in {"fp32", "bf16"} or self.attention_backend not in {"flash", "sdpa", "math"}:
            raise ValueError("Unknown precision or attention backend")
        if self.data_pipeline not in {"legacy_wiki", "legacy_c4", "release"}:
            raise ValueError("Unknown data preprocessing protocol")
        if len(self.betas) != 2 or not all(0 <= b < 1 for b in self.betas):
            raise ValueError("Invalid Adam betas")
        if not math.isfinite(self.lr) or self.lr <= 0:
            raise ValueError("lr must be positive and finite")
        if self.max_val_batches < 0:
            raise ValueError("max_val_batches must be nonnegative")
        for name in ["max_train_samples", "max_val_samples"]:
            if getattr(self, name) is not None and getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        return self

    @property
    def accumulation_steps(self):
        return self.global_batch_size // self.micro_batch_size

    @property
    def total_steps(self):
        return self.max_train_steps or self.max_tokens // (self.global_batch_size * self.seq_len)

    def to_dict(self):
        return asdict(self)


def legacy_micro_batch(dataset, method, model_size, seq_len, profile="march2026"):
    """Explicit launcher profiles; never scale the global batch with GPU count."""
    if profile not in {"march2026", "january2026", "attention2026"}:
        raise ValueError("Unknown batch profile")
    if model_size == "300M":
        value = 8 if method == "hipe_learnable" else 16
        return value // 2 if seq_len >= 2048 else value
    value = 32 if model_size == "20M" else 16
    # January HIPE sweep used 16/8, unlike the later WikiText multi-seed sweep.
    if method == "hipe" and profile != "attention2026" and (profile == "january2026" or dataset == "c4"):
        value //= 2
    if seq_len >= 2048:
        value //= 2
    if method == "alibi" and dataset == "wikitext103" and seq_len >= 2048:
        value = 8 if model_size == "20M" else 4
    return value

"""Shared model configuration for main experiments and ablations."""
from dataclasses import dataclass
import math
from typing import Optional, Sequence

from .olmo.config import ModelConfig


@dataclass(frozen=True)
class ModelSize:
    d_model: int
    n_heads: int
    n_layers: int
    mlp_ratio: int


MODEL_SIZES = {
    "20M": ModelSize(256, 8, 8, 8),
    "60M": ModelSize(512, 8, 8, 8),
    "300M": ModelSize(1024, 16, 16, 8),
}


def default_layer_threshold(n_layers: int) -> int:
    if n_layers < 2 or n_layers % 2:
        raise ValueError("The bipartite schedule requires an even layer count >= 2")
    return n_layers // 2 - 1


def resolve_layer_sigmas(n_layers: int, sigma: float, schedule: str = "bipartite",
                         layer_threshold: Optional[int] = None,
                         layer_sigmas: Optional[Sequence[Optional[float]]] = None):
    """None denotes actual unscaled RoPE; zero is not used as a RoPE alias."""
    if not math.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be positive")
    if schedule == "explicit":
        if layer_sigmas is None or len(layer_sigmas) != n_layers:
            raise ValueError("Explicit schedules require one sigma/None per layer")
        result = list(layer_sigmas)
    elif layer_sigmas is not None:
        raise ValueError("layer_sigmas requires schedule='explicit'")
    elif schedule == "uniform":
        result = [float(sigma)] * n_layers
    elif schedule in {"bipartite", "reverse"}:
        threshold = default_layer_threshold(n_layers) if layer_threshold is None else layer_threshold
        if not 0 <= threshold < n_layers - 1:
            raise ValueError("The split must leave at least one layer on each side")
        result = [None] * (threshold + 1) + [float(sigma)] * (n_layers - threshold - 1)
        if schedule == "reverse":
            result = [float(sigma) if value is None else None for value in result]
    else:
        raise ValueError(f"Unknown sigma schedule: {schedule}")
    if any(value is not None and (not math.isfinite(value) or value <= 0) for value in result):
        raise ValueError("Explicit non-RoPE scales must be positive")
    return result


def resolve_local_layers(n_layers: int, attention_pattern: str, local_layers: int = 4):
    if attention_pattern == "global":
        return []
    if not 0 < local_layers <= n_layers:
        raise ValueError("local_layers must be in [1, n_layers]")
    if attention_pattern == "local-global":
        return list(range(local_layers))
    if attention_pattern == "global-local":
        return list(range(n_layers - local_layers, n_layers))
    raise ValueError(f"Unknown attention pattern: {attention_pattern}")


def build_model_config(model_size: str, vocab_size: int, sequence_length: int,
                       sigma: float = 500.0, layer_threshold: Optional[int] = None,
                       flash_attention: bool = False, *, method: str = "hipe",
                       schedule: str = "bipartite", layer_sigmas=None,
                       learnable_sigma: bool = False, attention_pattern: str = "global",
                       local_window_size: int = 256, local_layers: int = 4,
                       legacy_absolute_positions: bool = False,
                       xpos_variant: str = "torchscale", xpos_scale_base: float = 512.0) -> ModelConfig:
    if model_size not in MODEL_SIZES:
        raise ValueError(f"Unknown model size: {model_size}")
    if method not in {"hipe", "rope", "nope", "alibi", "xpos"}:
        raise ValueError(f"Unknown positional method: {method}")
    if learnable_sigma and method != "hipe":
        raise ValueError("Learnable sigma requires HIPE")
    if legacy_absolute_positions and method not in {"nope", "xpos"}:
        raise ValueError("The legacy absolute-position switch only applies to NoPE/XPos")
    if xpos_variant not in {"torchscale", "legacy"}:
        raise ValueError("Unknown XPos implementation")
    if not math.isfinite(xpos_scale_base) or xpos_scale_base <= 0:
        raise ValueError("xpos_scale_base must be positive and finite")
    size = MODEL_SIZES[model_size]
    sigmas = resolve_layer_sigmas(size.n_layers, sigma, schedule, layer_threshold, layer_sigmas) if method == "hipe" else None
    local_ids = resolve_local_layers(size.n_layers, attention_pattern, local_layers)
    if local_ids and local_window_size <= 0:
        raise ValueError("Local attention requires a positive window")
    threshold = (default_layer_threshold(size.n_layers) if layer_threshold is None else layer_threshold) if schedule == "bipartite" else -1
    return ModelConfig(
        d_model=size.d_model, n_heads=size.n_heads, n_layers=size.n_layers,
        mlp_ratio=size.mlp_ratio, max_sequence_length=sequence_length,
        vocab_size=vocab_size, embedding_size=vocab_size, init_std=0.02,
        rope=method in {"rope", "hipe"}, alibi=method == "alibi",
        nope=method == "nope", xpos=method == "xpos",
        xpos_variant=xpos_variant, xpos_scale_base=float(xpos_scale_base),
        use_scaled_rope1=method == "hipe", scaled_rope_sigma=float(sigma),
        scaled_rope_sigmas=None if schedule == "bipartite" else sigmas, rope_scaling_threshold=threshold,
        learnable_sigma=learnable_sigma, decay_func="gaussian",
        local_attention_layers=local_ids, local_window_size=local_window_size,
        legacy_absolute_positions=legacy_absolute_positions,
        flash_attention=flash_attention and method != "alibi" and not local_ids,
    )

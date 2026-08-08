"""HIPE model configurations."""

from dataclasses import dataclass
from typing import Optional

from .olmo.config import ModelConfig


@dataclass(frozen=True)
class ModelSize:
    d_model: int
    n_heads: int
    n_layers: int
    mlp_ratio: int


MODEL_SIZES = {
    "20M": ModelSize(d_model=256, n_heads=8, n_layers=8, mlp_ratio=8),
    "60M": ModelSize(d_model=512, n_heads=8, n_layers=8, mlp_ratio=8),
    "300M": ModelSize(d_model=1024, n_heads=16, n_layers=16, mlp_ratio=8),
}


def default_layer_threshold(n_layers: int) -> int:
    """Return the bipartite split used by the paper's hierarchical schedule."""
    if n_layers % 2 != 0:
        raise ValueError("HIPE's bipartite schedule requires an even number of layers")
    return n_layers // 2 - 1


def build_model_config(
    model_size: str,
    vocab_size: int,
    sequence_length: int,
    sigma: float,
    layer_threshold: Optional[int] = None,
    flash_attention: bool = False,
) -> ModelConfig:
    try:
        size = MODEL_SIZES[model_size]
    except KeyError as exc:
        raise ValueError(f"Unknown model size: {model_size}") from exc

    threshold = (
        default_layer_threshold(size.n_layers)
        if layer_threshold is None
        else layer_threshold
    )
    if not 0 <= threshold < size.n_layers:
        raise ValueError("layer_threshold must select at least one standard and one HIPE layer")
    if sigma <= 0:
        raise ValueError("sigma must be positive")

    return ModelConfig(
        d_model=size.d_model,
        n_heads=size.n_heads,
        n_layers=size.n_layers,
        mlp_ratio=size.mlp_ratio,
        max_sequence_length=sequence_length,
        vocab_size=vocab_size,
        embedding_size=vocab_size,
        init_std=0.02,
        rope=True,
        use_scaled_rope1=True,
        scaled_rope_sigma=sigma,
        rope_scaling_threshold=threshold,
        decay_func="gaussian",
        flash_attention=flash_attention,
    )

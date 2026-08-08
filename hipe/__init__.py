"""HIPE: the public implementation of the main HIPE method."""

from .olmo.config import ModelConfig
from .olmo.model import OLMo

__all__ = ["ModelConfig", "OLMo"]

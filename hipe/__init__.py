"""HIPE research package; planning tools can run without the ML runtime."""
__all__ = ["ModelConfig", "OLMo"]


def __getattr__(name):
    if name == "ModelConfig":
        from .olmo.config import ModelConfig
        return ModelConfig
    if name == "OLMo":
        from .olmo.model import OLMo
        return OLMo
    raise AttributeError(name)

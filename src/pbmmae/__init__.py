"""PB-MMAE reproducibility package."""

from .config import CHANNEL_NAMES, PHYSICS_FAMILIES, Paper1Config, load_config

__all__ = ["CHANNEL_NAMES", "PHYSICS_FAMILIES", "Paper1Config", "PBMMAELinear64", "load_config"]


def __getattr__(name: str):
    """Delay the PyTorch import until the model is actually requested."""
    if name == "PBMMAELinear64":
        from .model import PBMMAELinear64
        return PBMMAELinear64
    raise AttributeError(name)

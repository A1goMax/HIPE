"""Small configuration helpers used by the vendored model implementation."""

from enum import Enum
from pathlib import Path
from typing import Optional


class StrEnum(str, Enum):
    """String-valued enum compatible with the Python versions used by HIPE."""

    def __str__(self) -> str:
        return self.value


def is_url(value: str) -> bool:
    return value.startswith(("http://", "https://", "s3://", "gs://"))


def find_latest_checkpoint(path: str) -> Optional[Path]:
    root = Path(path)
    if not root.exists():
        return None
    candidates = sorted(
        (item for item in root.iterdir() if item.is_dir() and item.name.startswith("step")),
        key=lambda item: item.name,
    )
    return candidates[-1] if candidates else None

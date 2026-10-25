"""Определение устройства для ML-моделей (torch-совместимое)."""

import platform
import sys
from typing import Literal


def detect_device() -> Literal["cpu", "cuda", "mps"]:
    """Возвращает имя устройства, которое принимает torch.device и SentenceTransformer.

    Возвращаемые значения:
    - ``"mps"`` — Apple Silicon (Metal)
    - ``"cuda"`` — NVIDIA GPU
    - ``"cpu"`` — fallback
    """
    # Apple Silicon
    if sys.platform == "darwin" and platform.machine() == "arm64":
        try:
            import torch

            if torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
    # NVIDIA
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"

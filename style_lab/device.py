"""Which device models run on. CPU by default; set STE_DEVICE (cpu / mps / cuda) to override.

Nothing is picked automatically: Apple's MPS backend crashes in recent torch builds on older macOS
versions, a 0.5B model is fast enough on CPU, and CUDA is used only when asked for.
"""
import os

DEVICE_ENV_VAR = "STE_DEVICE"


def pick_device() -> str:
    return os.environ.get(DEVICE_ENV_VAR) or "cpu"


def is_cuda(device: str) -> bool:
    return device.startswith("cuda")

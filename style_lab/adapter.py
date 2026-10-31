"""Where the trained adapter and its loss history live, and how to read them.

`scripts/train_adapter.py` writes these files; tab ③ reads them.
"""
import json
from pathlib import Path

ADAPTER_BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"  # an adapter only works with the base model it was trained on
ADAPTER_DIR = Path(__file__).parent.parent / "adapters" / "qwen2.5-0.5b-email"
LOSS_HISTORY_FILE = "loss_history.json"
TRAIN_COMMAND = "uv run python scripts/train_adapter.py"


def adapter_available(adapter_dir: Path = ADAPTER_DIR) -> bool:
    return (adapter_dir / "adapter_config.json").exists() and (adapter_dir / "adapter_model.safetensors").exists()


def load_loss_history(adapter_dir: Path = ADAPTER_DIR) -> dict | None:
    """`{"config": {...}, "steps": [{"step", "epoch", "loss"}, ...]}`, or None if not trained yet."""
    path = adapter_dir / LOSS_HISTORY_FILE
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

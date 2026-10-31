"""The email example pairs from the book's dataset, and their deterministic train / test split.

This split is the only source of email example pairs in the project: few-shot prompts (tab ②),
the adapter training script, and the test-pair dropdown (tab ③) all read it from here.

The dataset repeats some notes (140 unique notes in 200 records), so a plain random split would
leak test notes into train. Test pairs are therefore drawn only from records whose notes appear
exactly once, by a fixed-seed shuffle; every other record is a train pair, in dataset order.
"""
import json
import random
from collections import Counter
from dataclasses import dataclass
from functools import cache
from pathlib import Path

from style_lab.prompts import ExamplePair

DATA_FILE = Path(__file__).parent.parent / "data" / "fine_tuning_dataset.jsonl"
SEED = 42
N_TEST = 20


@dataclass(frozen=True)
class Split:
    train: tuple[ExamplePair, ...]  # dataset order; few-shot takes them from the front
    test: tuple[ExamplePair, ...]   # test pairs, dataset order


def load_example_pairs(path: Path = DATA_FILE) -> list[ExamplePair]:
    """All records as (notes -> email) example pairs, in file order."""
    pairs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            messages = {m["role"]: m["content"] for m in json.loads(line)["messages"]}
            pairs.append(ExamplePair(messages["user"], messages["assistant"]))
    return pairs


@cache
def train_test_split() -> Split:
    """The fixed 180 train / 20 test split. Same result on every run; no test notes appear in train."""
    pairs = load_example_pairs()
    counts = Counter(p.input for p in pairs)
    candidates = [i for i, p in enumerate(pairs) if counts[p.input] == 1]
    random.Random(SEED).shuffle(candidates)
    test_ids = set(candidates[:N_TEST])
    return Split(
        train=tuple(p for i, p in enumerate(pairs) if i not in test_ids),
        test=tuple(p for i, p in enumerate(pairs) if i in test_ids),
    )

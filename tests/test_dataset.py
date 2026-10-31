"""The email dataset and its deterministic train / test split (read from the real data file)."""
import json
from collections import Counter
from pathlib import Path

from style_lab.dataset import load_example_pairs, train_test_split

DATA_FILE = Path(__file__).parent.parent / "data" / "fine_tuning_dataset.jsonl"


def test_loads_all_200_email_example_pairs_from_the_book_dataset():
    pairs = load_example_pairs()

    assert len(pairs) == 200
    assert pairs[0].input.startswith("feedback 4 mobile app\n")
    assert pairs[0].output.startswith("Subject: Feedback on mobile app\n\nBody: Hello {name},")


def test_split_is_180_train_and_20_test_pairs():
    split = train_test_split()

    assert len(split.train) == 180
    assert len(split.test) == 20


def test_split_is_the_same_on_every_run():
    first, second = train_test_split(), train_test_split()

    assert first.train == second.train
    assert first.test == second.test
    # Pinned so a change to the seed or algorithm is noticed (the adapter is trained on this split).
    assert first.test[0].input.splitlines()[0] == EXPECTED_FIRST_TEST_NOTES_LINE


def test_no_test_pair_notes_appear_in_train():
    split = train_test_split()
    train_notes = {p.input for p in split.train}

    assert not [p for p in split.test if p.input in train_notes]


def test_test_pairs_have_notes_that_appear_once_in_the_whole_dataset():
    counts = Counter(json.loads(line)["messages"][1]["content"] for line in DATA_FILE.read_text().splitlines())

    assert all(counts[p.input] == 1 for p in train_test_split().test)


def test_split_covers_every_record_exactly_once():
    split = train_test_split()

    assert sorted(map(repr, split.train + split.test)) == sorted(map(repr, load_example_pairs()))


EXPECTED_FIRST_TEST_NOTES_LINE = "mtg w/ Design team @ {location} - {date} {time}"

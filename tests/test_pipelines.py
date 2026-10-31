"""Seam A: technique pipelines, driven by a fake text model (no downloads)."""
import pytest

from style_lab.dataset import train_test_split
from style_lab.pipelines import (fine_tuned_inputs, run_few_shot, run_fine_tuned_comparison,
                                 run_three_way_comparison, run_zero_shot)
from style_lab.presets import PRESETS
from style_lab.prompts import ExamplePair


class FakeTextModel:
    """Renders `role: content` lines, counts words as tokens, and replies with `output`.

    `output` is a string, or a function of (messages, max_new_tokens, temperature) for tests that need
    the reply to show what the model was given.
    """

    def __init__(self, output, name: str = "fake"):
        self.output = output
        self.name = name

    def render_prompt(self, messages):
        return "\n".join(f"{m['role']}: {m['content']}" for m in messages) + "\nassistant:"

    def count_tokens(self, text):
        return len(text.split())

    def generate(self, messages, max_new_tokens, temperature):
        return self.output(messages, max_new_tokens, temperature) if callable(self.output) else self.output


def settings_echo(messages, max_new_tokens, temperature):
    """A reply that reports the sampling settings it was called with."""
    return f"max_new_tokens={max_new_tokens} temperature={temperature}"


EMAIL_OUTPUT = """Subject: Minutes

Body: Hi {name},

- This is needed for our upcoming presi to stakeholders.
- will need ur help asap
- send 2 me when rdy

Best,
{name}"""


def test_zero_shot_returns_rendered_prompt_token_count_output_and_checklist():
    preset = PRESETS["Professional email"]
    model = FakeTextModel(EMAIL_OUTPUT)

    result = run_zero_shot(model, preset, preset.default_input)

    assert result.prompt == (
        "system: You are a helpful assistant converting notes to professional emails.\n"
        f"user: {preset.default_input}\nassistant:"
    )
    assert result.prompt_tokens == len(result.prompt.split())
    assert result.output == EMAIL_OUTPUT
    assert result.model_name == "fake"
    assert [c.name for c in result.checklist] == [
        "Has Subject: line", "Has Body: section", "Greeting", "Sign-off with {name}", "Content kept",
    ]
    assert all(c.passed for c in result.checklist)


def test_zero_shot_uses_edited_instructions_and_sampling_settings():
    preset = PRESETS["Professional email"]
    result = run_zero_shot(FakeTextModel(settings_echo), preset, "some notes", instructions="Write like a pirate.",
                           temperature=0.7, max_new_tokens=50)

    assert result.prompt == "system: Write like a pirate.\nuser: some notes\nassistant:"
    assert result.messages == [{"role": "system", "content": "Write like a pirate."},
                               {"role": "user", "content": "some notes"}]
    assert result.output == "max_new_tokens=50 temperature=0.7"


def test_zero_shot_social_media_gets_social_media_checklist():
    preset = PRESETS["Social media"]
    result = run_zero_shot(FakeTextModel("That sunset tho 😍🌅 #sunsetvibes"), preset, preset.default_input)

    assert result.prompt.startswith("system: Convert the following text into social media styled text:")
    assert {c.name: c.passed for c in result.checklist} == {"Emoji": True, "Hashtags": True}


# ---- ② few-shot ------------------------------------------------------------------------

def test_few_shot_with_k_0_is_exactly_the_zero_shot_prompt():
    for preset in PRESETS.values():
        zero = run_zero_shot(FakeTextModel("x"), preset, preset.default_input)
        few = run_few_shot(FakeTextModel("x"), preset, preset.default_input, k=0)

        assert few.prompt == zero.prompt
        assert few.prompt_tokens == zero.prompt_tokens
        assert few.technique == "Few-shot"


def example_turns(result):
    """The (input, output) example pairs a result's prompt contained, in order."""
    turns = result.messages[1:-1]
    return [(u["content"], a["content"]) for u, a in zip(turns[::2], turns[1::2])]


def test_few_shot_email_includes_exactly_k_train_pairs_in_fixed_order():
    preset = PRESETS["Professional email"]
    train = train_test_split().train

    for k in range(9):
        result = run_few_shot(FakeTextModel("x"), preset, "my notes", k=k)

        assert example_turns(result) == [(p.input, p.output) for p in train[:k]]
        assert result.messages[0] == {"role": "system", "content": preset.instructions}
        assert result.messages[-1] == {"role": "user", "content": "my notes"}
    first = example_turns(run_few_shot(FakeTextModel("x"), preset, "n", k=1))[0]
    assert first[0].startswith("feedback 4 mobile app")  # record 0 of the dataset


def test_few_shot_prompt_never_contains_a_test_pair():
    preset = PRESETS["Professional email"]
    result = run_few_shot(FakeTextModel("x"), preset, preset.default_input, k=8)

    for pair in train_test_split().test:
        assert pair.input not in result.prompt
        assert pair.output not in result.prompt


def test_few_shot_token_count_grows_with_k():
    preset = PRESETS["Professional email"]
    counts = [run_few_shot(FakeTextModel("x"), preset, preset.default_input, k=k).prompt_tokens
              for k in range(9)]

    assert all(a < b for a, b in zip(counts, counts[1:]))


def test_few_shot_social_media_uses_notebook_pairs_capped_at_four():
    preset = PRESETS["Social media"]
    result = run_few_shot(FakeTextModel("x"), preset, "The sunset was beautiful this evening.", k=8)

    assert [i for i, _ in example_turns(result)] == [
        "The movie was fantastic!",
        "I enjoyed the concert last night.",
        "The new restaurant has great food.",
        "The hiking trail was beautiful and peaceful.",
    ]


def test_few_shot_uses_edited_example_pairs_and_attaches_checklist():
    preset = PRESETS["Professional email"]
    edited = [ExamplePair("notes A", "Subject: wrong"), ExamplePair("notes B", "email B")]
    result = run_few_shot(FakeTextModel(EMAIL_OUTPUT), preset, preset.default_input, k=1, example_pairs=edited)

    assert example_turns(result) == [("notes A", "Subject: wrong")]
    assert all(c.passed for c in result.checklist)


# ---- ③ fine-tuned -----------------------------------------------------------------------

def test_fine_tuned_dropdown_offers_only_the_20_test_pairs():
    split = train_test_split()
    choices = fine_tuned_inputs()

    assert choices == list(split.test)
    assert len(choices) == 20
    train_inputs = {p.input for p in split.train}
    assert not any(c.input in train_inputs for c in choices)


def short_email_prompt(notes):
    """Tab ③'s short prompt: the email system prompt + the notes, no examples."""
    return [{"role": "system", "content": PRESETS["Professional email"].instructions},
            {"role": "user", "content": notes}]


def test_fine_tuned_comparison_uses_the_base_and_adapter_models_it_was_given():
    pair = fine_tuned_inputs()[0]
    base = FakeTextModel("Dear team, here is an email.", name="base")
    tuned = FakeTextModel(pair.output, name="base + adapter")

    comparison = run_fine_tuned_comparison(base, tuned, pair.input, expected_output=pair.output)

    assert (comparison.base.model_name, comparison.base.output) == ("base", "Dear team, here is an email.")
    assert (comparison.fine_tuned.model_name, comparison.fine_tuned.output) == ("base + adapter", pair.output)
    assert comparison.base.technique == "Zero-shot"
    assert comparison.fine_tuned.technique == "Fine-tuned"
    assert comparison.expected_output == pair.output
    assert comparison.base.messages == comparison.fine_tuned.messages == short_email_prompt(pair.input)
    assert comparison.base.prompt == comparison.fine_tuned.prompt
    assert comparison.base.prompt_tokens == comparison.fine_tuned.prompt_tokens
    assert {c.name: c.passed for c in comparison.base.checklist}["Has Subject: line"] is False
    assert all(c.passed for c in comparison.fine_tuned.checklist)


def test_fine_tuned_comparison_passes_the_sampling_settings_to_both_models():
    comparison = run_fine_tuned_comparison(FakeTextModel(settings_echo), FakeTextModel(settings_echo), "notes",
                                           temperature=0.3, max_new_tokens=40)

    assert comparison.base.output == comparison.fine_tuned.output == "max_new_tokens=40 temperature=0.3"


def test_fine_tuned_comparison_on_free_text_has_no_expected_output():
    comparison = run_fine_tuned_comparison(FakeTextModel("a"), FakeTextModel("b"),
                                           "schedule a meeting with John about the Q4 planning")
    assert comparison.expected_output is None


def test_three_way_comparison_runs_zero_shot_few_shot_and_fine_tuned_on_a_test_pair():
    pair = fine_tuned_inputs()[0]
    base = FakeTextModel("Dear team, here is an email.", name="base")
    tuned = FakeTextModel(pair.output, name="base + adapter")

    comparison = run_three_way_comparison(base, tuned, pair)
    results = comparison.results

    assert comparison.expected_output == pair.output
    assert [r.technique for r in results] == ["Zero-shot", "Few-shot", "Fine-tuned"]
    assert [r.model_name for r in results] == ["base", "base", "base + adapter"]
    assert [r.output for r in results] == ["Dear team, here is an email.", "Dear team, here is an email.",
                                           pair.output]
    assert all(r.messages[-1] == {"role": "user", "content": pair.input} for r in results)
    assert all(r.checklist for r in results)
    zero, few, fine = results
    assert example_turns(few) == [(p.input, p.output) for p in train_test_split().train[:3]]  # k = 3
    assert fine.messages == zero.messages == short_email_prompt(pair.input)  # fine-tuned is cheap again
    assert fine.prompt_tokens == zero.prompt_tokens < few.prompt_tokens
    assert all(c.passed for c in fine.checklist)
    assert {c.name: c.passed for c in zero.checklist}["Has Subject: line"] is False


def test_three_way_comparison_refuses_a_pair_that_is_not_a_test_pair():
    train_pair = train_test_split().train[0]

    with pytest.raises(ValueError, match="test pair"):
        run_three_way_comparison(FakeTextModel("a"), FakeTextModel("b"), train_pair)
    with pytest.raises(ValueError, match="test pair"):
        run_three_way_comparison(FakeTextModel("a"), FakeTextModel("b"), ExamplePair("free text", "made up"))

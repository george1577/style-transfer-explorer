"""Seam A: technique pipelines, driven by a fake text model (no downloads)."""
from style_lab.dataset import train_test_split
from style_lab.pipelines import (fine_tuned_inputs, run_few_shot, run_fine_tuned_comparison,
                                 run_three_way_comparison, run_zero_shot)
from style_lab.presets import PRESETS
from style_lab.prompts import ExamplePair


class FakeTextModel:
    """Renders `role: content` lines, counts words as tokens, returns a canned output."""

    def __init__(self, output: str, name: str = "fake"):
        self.output = output
        self.name = name
        self.calls = []

    def render_prompt(self, messages):
        return "\n".join(f"{m['role']}: {m['content']}" for m in messages) + "\nassistant:"

    def count_tokens(self, text):
        return len(text.split())

    def generate(self, messages, max_new_tokens, temperature):
        self.calls.append({"messages": messages, "max_new_tokens": max_new_tokens, "temperature": temperature})
        return self.output


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
    model = FakeTextModel("hello")

    result = run_zero_shot(model, preset, "some notes", instructions="Write like a pirate.",
                           temperature=0.7, max_new_tokens=50)

    assert result.prompt.startswith("system: Write like a pirate.\nuser: some notes")
    assert model.calls == [{
        "messages": [{"role": "system", "content": "Write like a pirate."},
                     {"role": "user", "content": "some notes"}],
        "max_new_tokens": 50,
        "temperature": 0.7,
    }]


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


def test_fine_tuned_comparison_uses_the_base_and_adapter_models_it_was_given():
    pair = fine_tuned_inputs()[0]
    base = FakeTextModel("Dear team, here is an email.", name="base")
    tuned = FakeTextModel(pair.output, name="base + adapter")

    comparison = run_fine_tuned_comparison(base, tuned, pair.input, reference=pair.output, temperature=0.3)

    assert comparison.base.output == "Dear team, here is an email."
    assert comparison.base.model_name == "base"
    assert comparison.fine_tuned.output == pair.output
    assert comparison.fine_tuned.model_name == "base + adapter"
    assert comparison.reference == pair.output
    short_prompt = [{"role": "system", "content": PRESETS["Professional email"].instructions},
                    {"role": "user", "content": pair.input}]
    assert base.calls[0]["messages"] == short_prompt  # system prompt + notes, no examples
    assert tuned.calls[0]["messages"] == short_prompt
    assert base.calls[0]["temperature"] == tuned.calls[0]["temperature"] == 0.3
    assert comparison.base.prompt_tokens == comparison.fine_tuned.prompt_tokens
    assert {c.name: c.passed for c in comparison.base.checklist}["Has Subject: line"] is False
    assert all(c.passed for c in comparison.fine_tuned.checklist)


def test_fine_tuned_comparison_on_free_text_has_no_reference():
    comparison = run_fine_tuned_comparison(FakeTextModel("a"), FakeTextModel("b"),
                                           "schedule a meeting with John about the Q4 planning")
    assert comparison.reference is None


def test_three_way_comparison_returns_zero_shot_few_shot_and_fine_tuned_on_the_same_input():
    pair = fine_tuned_inputs()[0]
    base = FakeTextModel("Dear team, here is an email.", name="base")
    tuned = FakeTextModel(pair.output, name="base + adapter")

    results = run_three_way_comparison(base, tuned, pair.input)

    assert [r.technique for r in results] == ["Zero-shot", "Few-shot", "Fine-tuned"]
    assert [r.model_name for r in results] == ["base", "base", "base + adapter"]
    assert all(r.messages[-1] == {"role": "user", "content": pair.input} for r in results)
    assert all(r.checklist for r in results)
    zero, few, fine = results
    assert example_turns(few) == [(p.input, p.output) for p in train_test_split().train[:3]]  # k = 3
    assert fine.messages == zero.messages  # fine-tuned is cheap again: the zero-shot prompt
    assert fine.prompt_tokens == zero.prompt_tokens < few.prompt_tokens
    assert all(c.passed for c in fine.checklist)
    assert {c.name: c.passed for c in zero.checklist}["Has Subject: line"] is False

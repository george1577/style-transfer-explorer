"""Seam A: technique pipelines, driven by a fake text model (no downloads)."""
from style_lab.pipelines import run_zero_shot
from style_lab.presets import PRESETS


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

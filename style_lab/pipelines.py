"""Wire a text model to each technique. The Streamlit app calls only these functions.

Every pipeline returns `TextResult`s, so tabs and comparisons can show any technique the same way.
"""
from dataclasses import dataclass

from style_lab.checklist import Check
from style_lab.presets import StylePreset
from style_lab.prompts import ExamplePair, Message, few_shot_messages, zero_shot_messages
from style_lab.text_model import TextModel

MAX_NEW_TOKENS = 300  # dataset answers are well under 300 tokens


@dataclass(frozen=True)
class TextResult:
    technique: str
    model_name: str
    messages: list[Message]
    prompt: str  # fully rendered, chat template applied
    prompt_tokens: int
    output: str
    checklist: list[Check]


def _run(technique: str, model: TextModel, preset: StylePreset, text: str, messages: list[Message],
         max_new_tokens: int, temperature: float) -> TextResult:
    prompt = model.render_prompt(messages)
    output = model.generate(messages, max_new_tokens=max_new_tokens, temperature=temperature)
    return TextResult(
        technique=technique,
        model_name=model.name,
        messages=messages,
        prompt=prompt,
        prompt_tokens=model.count_tokens(prompt),
        output=output,
        checklist=preset.checklist(text, output),
    )


# ---- ① zero-shot ----------------------------------------------------------------------

def run_zero_shot(model: TextModel, preset: StylePreset, text: str, instructions: str | None = None,
                  max_new_tokens: int = MAX_NEW_TOKENS, temperature: float = 0.0) -> TextResult:
    messages = zero_shot_messages(preset.instructions if instructions is None else instructions, text)
    return _run("Zero-shot", model, preset, text, messages, max_new_tokens, temperature)


# ---- ② few-shot -----------------------------------------------------------------------

def run_few_shot(model: TextModel, preset: StylePreset, text: str, k: int, instructions: str | None = None,
                 example_pairs: list[ExamplePair] | None = None,
                 max_new_tokens: int = MAX_NEW_TOKENS, temperature: float = 0.0) -> TextResult:
    """Zero-shot plus the first k example pairs. `example_pairs` overrides the preset's (e.g. edited in the UI)."""
    pairs = (preset.example_pairs() if example_pairs is None else example_pairs)[:k]
    messages = few_shot_messages(preset.instructions if instructions is None else instructions, pairs, text)
    return _run("Few-shot", model, preset, text, messages, max_new_tokens, temperature)

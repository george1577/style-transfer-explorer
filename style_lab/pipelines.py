"""Wire a text model to each technique. The Streamlit app calls only these functions.

Every pipeline returns `TextResult`s, so tabs and comparisons can show any technique the same way.
"""
from dataclasses import dataclass

from style_lab.checklist import Check
from style_lab.dataset import train_test_split
from style_lab.presets import PRESETS, StylePreset
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


# ---- ③ fine-tuned ---------------------------------------------------------------------

def fine_tuned_inputs() -> list[ExamplePair]:
    """The test pairs tab ③ offers as inputs: held out from training, so the comparison is fair."""
    return list(train_test_split().test)


@dataclass(frozen=True)
class FineTunedComparison:
    base: TextResult
    fine_tuned: TextResult
    reference: str | None  # the test pair's reference answer; None for free text


def run_fine_tuned_comparison(base: TextModel, fine_tuned: TextModel, text: str, reference: str | None = None,
                              max_new_tokens: int = MAX_NEW_TOKENS, temperature: float = 0.0) -> FineTunedComparison:
    """Base model vs base + adapter on the same short prompt (email system prompt + notes, no examples)."""
    preset = PRESETS["Professional email"]  # the adapter is email-only
    messages = zero_shot_messages(preset.instructions, text)
    return FineTunedComparison(
        base=_run("Base model", base, preset, text, messages, max_new_tokens, temperature),
        fine_tuned=_run("Fine-tuned", fine_tuned, preset, text, messages, max_new_tokens, temperature),
        reference=reference,
    )


THREE_WAY_K = 3


def run_three_way_comparison(base: TextModel, fine_tuned: TextModel, text: str,
                             max_new_tokens: int = MAX_NEW_TOKENS, temperature: float = 0.0) -> list[TextResult]:
    """Zero-shot, few-shot (k = 3) and fine-tuned on the same email input: the cost/quality lesson in one view."""
    preset = PRESETS["Professional email"]
    return [
        run_zero_shot(base, preset, text, max_new_tokens=max_new_tokens, temperature=temperature),
        run_few_shot(base, preset, text, THREE_WAY_K, max_new_tokens=max_new_tokens, temperature=temperature),
        _run("Fine-tuned", fine_tuned, preset, text, zero_shot_messages(preset.instructions, text),
             max_new_tokens, temperature),
    ]

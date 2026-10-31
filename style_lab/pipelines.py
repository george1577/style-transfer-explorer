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

EMAIL = PRESETS["Professional email"]  # the adapter is email-only


def fine_tuned_inputs() -> list[ExamplePair]:
    """The test pairs tab ③ offers as inputs: held out from fine-tuning, so the comparison is fair."""
    return list(train_test_split().test)


def _run_fine_tuned(fine_tuned: TextModel, text: str, max_new_tokens: int, temperature: float) -> TextResult:
    """The fine-tuned model on the short zero-shot prompt (email system prompt + notes, no examples)."""
    return _run("Fine-tuned", fine_tuned, EMAIL, text, zero_shot_messages(EMAIL.instructions, text),
                max_new_tokens, temperature)


@dataclass(frozen=True)
class FineTunedComparison:
    base: TextResult  # zero-shot on the base model: the same prompt, without the adapter
    fine_tuned: TextResult
    expected_output: str | None  # the test pair's expected email; None for free text


def run_fine_tuned_comparison(base: TextModel, fine_tuned: TextModel, text: str, expected_output: str | None = None,
                              max_new_tokens: int = MAX_NEW_TOKENS, temperature: float = 0.0) -> FineTunedComparison:
    """Base model vs base + adapter on the same short prompt, so any difference is what fine-tuning added."""
    return FineTunedComparison(
        base=run_zero_shot(base, EMAIL, text, max_new_tokens=max_new_tokens, temperature=temperature),
        fine_tuned=_run_fine_tuned(fine_tuned, text, max_new_tokens, temperature),
        expected_output=expected_output,
    )


THREE_WAY_K = 3


@dataclass(frozen=True)
class ThreeWayComparison:
    results: list[TextResult]  # zero-shot, few-shot, fine-tuned
    expected_output: str  # the test pair's expected email


def run_three_way_comparison(base: TextModel, fine_tuned: TextModel, pair: ExamplePair,
                             max_new_tokens: int = MAX_NEW_TOKENS, temperature: float = 0.0) -> ThreeWayComparison:
    """Zero-shot, few-shot (k = 3) and fine-tuned on one test pair: the cost/quality lesson in one view.

    Only test pairs are accepted: they are the inputs none of the three techniques has seen.
    """
    if pair not in train_test_split().test:
        raise ValueError("The three-way comparison runs only on a test pair.")
    text = pair.input
    return ThreeWayComparison(
        results=[
            run_zero_shot(base, EMAIL, text, max_new_tokens=max_new_tokens, temperature=temperature),
            run_few_shot(base, EMAIL, text, THREE_WAY_K, max_new_tokens=max_new_tokens, temperature=temperature),
            _run_fine_tuned(fine_tuned, text, max_new_tokens, temperature),
        ],
        expected_output=pair.output,
    )

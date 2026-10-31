"""Build the chat message lists each text technique sends to the model."""
from dataclasses import dataclass

Message = dict[str, str]


@dataclass(frozen=True)
class ExamplePair:
    """One input together with its correct output in the target style."""
    input: str
    output: str


def zero_shot_messages(instructions: str, text: str) -> list[Message]:
    """Instructions as the system prompt, the input as the user turn. No examples."""
    return [
        {"role": "system", "content": instructions},
        {"role": "user", "content": text},
    ]


def few_shot_messages(instructions: str, pairs: list[ExamplePair], text: str) -> list[Message]:
    """Zero-shot plus each example pair as a user / assistant turn before the input.

    With no pairs this is exactly `zero_shot_messages`.
    """
    examples = [turn for pair in pairs
                for turn in ({"role": "user", "content": pair.input},
                             {"role": "assistant", "content": pair.output})]
    system, user = zero_shot_messages(instructions, text)
    return [system, *examples, user]

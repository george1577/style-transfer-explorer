"""Build the chat message lists each text technique sends to the model."""

Message = dict[str, str]


def zero_shot_messages(instructions: str, text: str) -> list[Message]:
    """Instructions as the system prompt, the input as the user turn. No examples."""
    return [
        {"role": "system", "content": instructions},
        {"role": "user", "content": text},
    ]

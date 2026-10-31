"""The text model interface, and the only module that touches a real model.

Pipelines depend on the `TextModel` protocol, so tests pass a fake and the app passes a `QwenTextModel`.
"""
import os
from typing import Protocol

from style_lab.prompts import Message

MODELS = ["Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct"]
DEVICE_ENV_VAR = "STE_DEVICE"


class TextModel(Protocol):
    name: str

    def render_prompt(self, messages: list[Message]) -> str:
        """The exact text the model receives: chat template applied, ready for the reply."""

    def count_tokens(self, text: str) -> int:
        """How many tokens `text` costs."""

    def generate(self, messages: list[Message], max_new_tokens: int, temperature: float) -> str:
        """The model's reply to `messages`. Temperature 0 means greedy decoding."""


def pick_device() -> str:
    # Apple's MPS backend is opt-in: recent torch builds crash on older macOS versions,
    # and a 0.5B model is fast enough on CPU. Set STE_DEVICE=mps (or cuda) to override.
    import torch

    if os.environ.get(DEVICE_ENV_VAR):
        return os.environ[DEVICE_ENV_VAR]
    return "cuda" if torch.cuda.is_available() else "cpu"


class QwenTextModel:
    """A Qwen2.5 instruct model behind the `TextModel` interface."""

    def __init__(self, model_id: str = MODELS[0]):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.name = model_id
        self.device = pick_device()
        dtype = torch.float32 if self.device == "cpu" else torch.float16
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(self.device)
        self.model.eval()

    def render_prompt(self, messages: list[Message]) -> str:
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    def generate(self, messages: list[Message], max_new_tokens: int, temperature: float) -> str:
        import torch

        prompt = self.render_prompt(messages)
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(self.device)
        # Override Qwen's generation_config defaults (top_p/top_k/temperature) so 0 is plain greedy
        # and >0 is plain temperature sampling, as in the notebooks.
        if temperature > 0:
            sampling = {"do_sample": True, "temperature": temperature, "top_p": 1.0, "top_k": 0}
        else:
            sampling = {"do_sample": False, "temperature": None, "top_p": None, "top_k": None}
        with torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, pad_token_id=self.tokenizer.eos_token_id, **sampling
            )
        new_ids = out[0, inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()

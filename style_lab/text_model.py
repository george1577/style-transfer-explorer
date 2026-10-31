"""The text model interface, and the only module that touches a real model.

Pipelines depend on the `TextModel` protocol, so tests pass a fake and the app passes a `QwenTextModel`.
"""
import contextlib
import copy
from pathlib import Path
from typing import Protocol

from style_lab.device import pick_device
from style_lab.prompts import Message

MODELS = ["Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct"]


class TextModel(Protocol):
    name: str

    def render_prompt(self, messages: list[Message]) -> str:
        """The exact text the model receives: chat template applied, ready for the reply."""

    def count_tokens(self, text: str) -> int:
        """How many tokens `text` costs."""

    def generate(self, messages: list[Message], max_new_tokens: int, temperature: float) -> str:
        """The model's reply to `messages`. Temperature 0 means greedy decoding."""


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
        with self._weights():
            return self._generate(messages, max_new_tokens, temperature)

    def _weights(self):
        """Context that selects which weights generate with. Plain models have only one set."""
        return contextlib.nullcontext()

    def _generate(self, messages: list[Message], max_new_tokens: int, temperature: float) -> str:
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


class AdaptedQwenTextModel(QwenTextModel):
    """A Qwen2.5 model with a LoRA adapter loaded on top. `base_view()` is the same weights, adapter off.

    Base and fine-tuned share one copy of the base weights (peft's `disable_adapter()`), so tab ③
    costs one model's memory, not two.
    """

    def __init__(self, model_id: str, adapter_dir: Path):
        from peft import PeftModel

        super().__init__(model_id)
        self.model = PeftModel.from_pretrained(self.model, str(adapter_dir))
        self.model.eval()
        self.name = f"{model_id} + adapter"
        self.adapter_on = True

    def base_view(self) -> "AdaptedQwenTextModel":
        view = copy.copy(self)
        view.adapter_on = False
        view.name = self.name.removesuffix(" + adapter")
        return view

    def _weights(self):
        return contextlib.nullcontext() if self.adapter_on else self.model.disable_adapter()

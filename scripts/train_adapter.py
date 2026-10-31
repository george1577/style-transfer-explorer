"""LoRA-fine-tune Qwen2.5-0.5B-Instruct on the 180 train pairs, on CPU, and save the adapter.

    uv run python scripts/train_adapter.py            # ~80 minutes on an M2 CPU

Each train pair becomes the same short prompt tab ③ sends (system prompt + notes, no examples)
followed by the expected email. The chat template is applied, and the loss is computed on the
assistant tokens only, so the model learns to write the email, not to repeat the notes.

Writes `adapters/qwen2.5-0.5b-email/` (adapter weights + config + loss_history.json).
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from style_lab.adapter import ADAPTER_BASE_MODEL, ADAPTER_DIR, LOSS_HISTORY_FILE  # noqa: E402
from style_lab.dataset import train_test_split  # noqa: E402
from style_lab.presets import EMAIL_INSTRUCTIONS  # noqa: E402
from style_lab.prompts import zero_shot_messages  # noqa: E402

CONFIG = {
    "base_model": ADAPTER_BASE_MODEL,
    "epochs": 3,
    "batch_size": 1,
    "grad_accum": 4,  # effective batch 4, without the memory of batch 4 (8 GB machine)
    "learning_rate": 2e-4,
    "lora_r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "max_length": 512,
    "seed": 0,
}


def encode(tokenizer, pair) -> dict[str, torch.Tensor]:
    """Token ids for prompt + answer, with labels masked (-100) everywhere except the answer."""
    messages = zero_shot_messages(EMAIL_INSTRUCTIONS, pair.input)
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    full = tokenizer.apply_chat_template(
        [*messages, {"role": "assistant", "content": pair.output}], tokenize=False)
    assert full.startswith(prompt)
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full, add_special_tokens=False)["input_ids"][:CONFIG["max_length"]]
    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
    return {"input_ids": torch.tensor([full_ids]), "labels": torch.tensor([labels])}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ADAPTER_DIR)
    parser.add_argument("--max-steps", type=int, default=None, help="stop early (smoke test only)")
    args = parser.parse_args()

    random.seed(CONFIG["seed"])
    torch.manual_seed(CONFIG["seed"])
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["base_model"])
    model = AutoModelForCausalLM.from_pretrained(CONFIG["base_model"], dtype=torch.float32)
    model = get_peft_model(model, LoraConfig(
        r=CONFIG["lora_r"], lora_alpha=CONFIG["lora_alpha"], lora_dropout=CONFIG["lora_dropout"],
        target_modules=CONFIG["target_modules"], task_type="CAUSAL_LM"))
    model.print_trainable_parameters()
    model.train()

    examples = [encode(tokenizer, p) for p in train_test_split().train]
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=CONFIG["learning_rate"])
    steps_per_epoch = len(examples) // CONFIG["grad_accum"]
    total_steps = steps_per_epoch * CONFIG["epochs"]
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: max(0.0, 1 - s / total_steps))

    history, step, start = [], 0, time.time()
    for epoch in range(CONFIG["epochs"]):
        order = list(range(len(examples)))
        random.shuffle(order)
        for i in range(steps_per_epoch):
            batch_loss = 0.0
            for j in order[i * CONFIG["grad_accum"]:(i + 1) * CONFIG["grad_accum"]]:
                loss = model(**examples[j]).loss / CONFIG["grad_accum"]
                loss.backward()
                batch_loss += loss.item()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            step += 1
            history.append({"step": step, "epoch": round(epoch + (i + 1) / steps_per_epoch, 3),
                            "loss": round(batch_loss, 4)})
            print(f"step {step}/{total_steps} epoch {history[-1]['epoch']:.2f} loss {batch_loss:.4f} "
                  f"({time.time() - start:.0f}s)", flush=True)
            if args.max_steps and step >= args.max_steps:
                break
        if args.max_steps and step >= args.max_steps:
            break

    args.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out)
    (args.out / LOSS_HISTORY_FILE).write_text(json.dumps({
        "config": {**CONFIG, "train_pairs": len(examples), "steps": step,
                   "train_seconds": round(time.time() - start)},
        "steps": history,
    }, indent=1) + "\n", encoding="utf-8")
    print(f"Saved adapter and loss history to {args.out}")


if __name__ == "__main__":
    main()

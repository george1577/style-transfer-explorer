# Style Transfer Explorer

An interactive Streamlit app for seeing how a generative model can be made to rewrite text in a target style, and what each technique costs. It's a companion to the *Style Transfer* examples in [Generative AI Design Patterns](https://github.com/lakshmanok/generative-ai-design-patterns) (`examples/03_style_transfer`).

> Terms are defined in [GLOSSARY.md](GLOSSARY.md).

---

## Quick start

Requirements: [uv](https://docs.astral.sh/uv/), about 2 GB of free RAM, and internet on the first run (the 0.5B model is ~1 GB).

```bash
uv sync                          # installs Python 3.12 + dependencies into .venv
uv run streamlit run app.py      # opens http://localhost:8501
```

The first click is slow while the model downloads and loads; after that it is cached across reruns. Run the tests with `uv run pytest`.

### Configuration

| Setting | How | Default |
|---|---|---|
| Model | Sidebar dropdown | Qwen2.5 0.5B Instruct (1.5B optional) |
| Temperature | Sidebar slider | 0 (greedy, reproducible) |
| Device | `STE_DEVICE=mps uv run streamlit run app.py` (or `cuda`, `cpu`) | CUDA if available, else CPU |

> **Why not Apple's MPS by default?** Recent PyTorch builds crash on MPS on older macOS versions (seen on macOS 14), and a 0.5B model is fast enough on CPU.

---

## How the app works

### ① Zero-shot

**What it shows:** how far instructions alone get you towards a target style.

**You provide:** a style preset (Professional email or Social media), the instructions (system prompt), and the input. Both are pre-filled: the email default is notebook 0's meeting-minutes notes.

**What happens when you click *Generate*:**
1. The instructions and input become a two-message chat (system + user).
2. The model's chat template turns them into the exact prompt text, whose tokens are counted.
3. The model writes a reply (greedy at temperature 0).
4. The style checklist for the preset checks the reply.

**What you see:**
- The prompt token count and the fully rendered prompt.
- The output.
- The style checklist: each check passes or fails, with its evidence. Email checks `Subject:`, `Body:`, a greeting, a sign-off with `{name}`, and whether every point from the notes was kept; social media counts emoji and hashtags.

## Project structure

```
app.py                  Streamlit UI. The only file that knows about Streamlit.
style_lab/
  text_model.py         Text model interface + Qwen implementation; the only file touching torch.
  pipelines.py          One function per tab; each returns a TextResult. The app calls only these.
  prompts.py            Builds the chat messages for each technique.
  presets.py            Style presets: instructions, default input, example pairs, checklist.
  checklist.py          The style checklist, as pure functions.
tests/                  Fake text model, no downloads.
docs/adr/               Architecture decisions.
```

## Differences from the reference notebooks

- Text runs on a local Qwen2.5 model instead of OpenAI `gpt-3.5-turbo` (see [ADR 0001](docs/adr/0001-local-qwen-instead-of-openai.md)).
- Social media instructions go in a system message rather than the first line of a single user message.

## Credits

The source notebooks are from [Generative AI Design Patterns](https://github.com/lakshmanok/generative-ai-design-patterns) by Valliappa Lakshmanan and Hannes Hapke (Apache License 2.0).

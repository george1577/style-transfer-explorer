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

### ② Few-shot

**What it shows:** how example pairs in the prompt buy style, and what they cost in tokens.

**You provide:** a style preset, k (0–8, how many example pairs go in the prompt), the instructions, and the input. Each included example pair is shown in an expander you can edit, to see what inconsistent or wrong examples do.

**Where the example pairs come from:**
- **Professional email:** the dataset's training pairs, always in the same fixed order (see [The dataset and its split](#the-dataset-and-its-split)). Test pairs never appear in a prompt.
- **Social media:** notebook 1's four example pairs, so k stops at 4.

**What happens when you click *Generate*:**
1. The instructions become the system message; each example pair becomes a user turn (its input) followed by an assistant turn (its output); the input is the last user turn. With k = 0 this is exactly the zero-shot prompt.
2. The prompt is rendered with the chat template and its tokens counted, then the model replies and the style checklist checks the reply.

**What you see:** the prompt token count (it grows with every example pair), the rendered prompt, the output and its style checklist.

## The dataset and its split

`data/fine_tuning_dataset.jsonl` holds 200 email example pairs (notes → professional email), copied from the book's repo (see [Credits](#credits)). `style_lab/dataset.py` splits them, the same way on every run, into **180 training pairs** and **20 test pairs**. This split is the only source of email example pairs: few-shot prompts, adapter training and the test-pair dropdown all use it.

The dataset repeats some notes (only 140 of the 200 are unique), so a plain random split would put the same notes in both halves. Test pairs are therefore chosen, by a fixed-seed shuffle, only from records whose notes appear exactly once; all other records are training pairs, kept in file order. (Notebook 2 tests its fine-tuned model on a training example, and says so; the held-out test pairs avoid that.)

## Project structure

```
app.py                  Streamlit UI. The only file that knows about Streamlit.
style_lab/
  text_model.py         Text model interface + Qwen implementation; the only file touching torch.
  pipelines.py          One function per tab; each returns a TextResult. The app calls only these.
  prompts.py            Builds the chat messages for each technique.
  presets.py            Style presets: instructions, default input, example pairs, checklist.
  dataset.py            Loads the email example pairs and makes the fixed 180/20 train/test split.
  checklist.py          The style checklist, as pure functions.
data/                   The book's email dataset (fine_tuning_dataset.jsonl).
tests/                  Fake text model, no downloads.
docs/adr/               Architecture decisions.
```

## Differences from the reference notebooks

- Text runs on a local Qwen2.5 model instead of OpenAI `gpt-3.5-turbo` (see [ADR 0001](docs/adr/0001-local-qwen-instead-of-openai.md)).
- Social media instructions go in a system message rather than the first line of a single user message.
- Few-shot example pairs are user / assistant chat turns, rather than notebook 1's `Input:` / `Output:` lines in one message.

## Credits

The source notebooks are from [Generative AI Design Patterns](https://github.com/lakshmanok/generative-ai-design-patterns) by Valliappa Lakshmanan and Hannes Hapke (Apache License 2.0). The email dataset in `data/fine_tuning_dataset.jsonl` is copied unchanged from that repo (`examples/03_style_transfer/fine_tuning_dataset.jsonl`).

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

### ④ Image

**What it shows:** how a depth map carries a reference image's composition into image generation, and how strength and guidance shape the result.

**You provide:** one of two public-domain reference images (or your own upload), one of two prompts per reference image, and a strength (0.3 / 0.5 / 0.7 / 0.9) and guidance (3 / 7.5 / 12) value.

**What happens:**
1. Depth Anything V2 Small computes the depth map live on CPU (first run downloads ~100 MB).
2. The sliders snap to the values precomputed in the image grid, and the app looks the image up by its parameters in `data/grid/manifest.json`.
3. On a GPU device only (`STE_DEVICE=cuda`), *Generate live* runs the same SD1.5 + depth ControlNet pipeline on your prompt.

**What you see:**
- The reference image next to its depth map.
- The grid image for the chosen settings, with its seed, steps and models; optionally the full 4 × 3 grid.
- The same prompt with and without depth conditioning (strength 0.7, guidance 7.5).
- Placeholders with instructions until the grid is rendered: run `scripts/render_grid.py` on a Colab T4 (how-to at the top of the script) and commit `data/grid/`. See [ADR 0002](docs/adr/0002-precomputed-image-grid-instead-of-live-generation.md) for why the grid is precomputed.

## Project structure

```
app.py                  Streamlit UI. The only file that knows about Streamlit.
style_lab/
  text_model.py         Text model interface + Qwen implementation; the only file touching torch.
  pipelines.py          One function per tab; each returns a TextResult. The app calls only these.
  prompts.py            Builds the chat messages for each technique.
  presets.py            Style presets: instructions, default input, example pairs, checklist.
  checklist.py          The style checklist, as pure functions.
  image_grid.py         Reference images, prompts, grid axes, and the grid manifest (written by the script, read by the app).
  depth.py              Depth maps with Depth Anything V2 Small (app and grid script).
  live_generation.py    SD1.5 + depth ControlNet img2img; GPU only.
scripts/render_grid.py  Colab T4 script that renders the image grid and its manifest.
data/images/            The two reference images (512 px).
tests/                  Fake text model and fake grid manifests, no downloads.
docs/adr/               Architecture decisions.
```

## Differences from the reference notebooks

- Text runs on a local Qwen2.5 model instead of OpenAI `gpt-3.5-turbo` (see [ADR 0001](docs/adr/0001-local-qwen-instead-of-openai.md)).
- Social media instructions go in a system message rather than the first line of a single user message.
- Depth maps use Depth Anything V2 Small instead of the notebook's default DPT-Large, and the image results come from a precomputed grid instead of live generation (see [ADR 0002](docs/adr/0002-precomputed-image-grid-instead-of-live-generation.md)). The grid uses a fixed seed and 50 steps (the notebook uses 100 steps and no seed); images without depth conditioning use the same pipeline with the ControlNet's conditioning scale set to 0.

## Credits

The source notebooks are from [Generative AI Design Patterns](https://github.com/lakshmanok/generative-ai-design-patterns) by Valliappa Lakshmanan and Hannes Hapke (Apache License 2.0).

Reference images (public domain, via Wikimedia Commons; resized to 512 px in `data/images/`):
- *Wanderer above the Sea of Fog* (c. 1817), Caspar David Friedrich: [Commons page](https://commons.wikimedia.org/wiki/File:Caspar_David_Friedrich_-_Wanderer_above_the_sea_of_fog.jpg)
- *Paris Street; Rainy Day* (1877), Gustave Caillebotte: [Commons page](https://commons.wikimedia.org/wiki/File:Gustave_Caillebotte_-_Paris_Street;_Rainy_Day_-_Google_Art_Project.jpg)

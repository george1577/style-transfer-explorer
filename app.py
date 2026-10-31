"""Style Transfer Explorer: the Streamlit UI. The only file that knows about Streamlit."""
from pathlib import Path

import streamlit as st

from style_lab.pipelines import TextResult, run_few_shot, run_zero_shot
from style_lab.presets import PRESETS
from style_lab.prompts import ExamplePair
from style_lab.text_model import DEVICE_ENV_VAR, MODELS, QwenTextModel, pick_device

DOC_LINK = "docs/how-style-transfer-works.md"
# A relative path is not a working link inside Streamlit, so link to the repo's copy on GitHub.
DOC_URL = f"https://github.com/george1577/style-transfer-explorer/blob/main/{DOC_LINK}"
RESULT_KEYS = ("t1", "t2", "t3", "t3_three")

st.set_page_config(page_title="Style Transfer Explorer", page_icon="🎨", layout="wide")


def code_version() -> str:
    files = sorted((Path(__file__).parent / "style_lab").glob("*.py"))
    return "|".join(f"{f.name}:{f.stat().st_mtime_ns}" for f in files)


# Drop results computed by older package code, so a stale result never masquerades as a new one.
if st.session_state.get("_code_version") != (version := code_version()):
    for key in RESULT_KEYS:
        st.session_state.pop(key, None)
    st.session_state["_code_version"] = version


@st.cache_resource(show_spinner="Loading model (first run downloads it)…")
def get_model(model_id: str) -> QwenTextModel:
    return QwenTextModel(model_id)


def show_result(result: TextResult) -> None:
    st.markdown(f"**Prompt tokens:** {result.prompt_tokens}")
    with st.expander("Rendered prompt (exactly what the model receives)"):
        st.code(result.prompt, language=None, wrap_lines=True)
    st.markdown("**Output**")
    st.code(result.output, language=None, wrap_lines=True)
    st.markdown("**Style checklist**")
    st.dataframe(
        [{"Check": c.name, "Result": "✅ pass" if c.passed else "❌ fail", "Evidence": c.evidence}
         for c in result.checklist],
        hide_index=True, width="stretch",
    )


with st.sidebar:
    st.title("🎨 Style Transfer Explorer")
    model_id = st.selectbox("Model", MODELS,
                            help="0.5B is fast on a laptop CPU. 1.5B writes better but is ~3x slower.")
    temperature = st.slider("Temperature", 0.0, 1.5, 0.0, 0.1,
                            help="0 = greedy (same output every time). The notebooks use 0.7.")
    lm = get_model(model_id)
    st.caption(f"Running on **{lm.device}** (set `{DEVICE_ENV_VAR}` to change)")
    st.markdown("**Learning path**\n1. Zero-shot\n2. Few-shot\n3. Fine-tuned\n4. Image\n\n"
                f"Read the companion guide: [`{DOC_LINK}`]({DOC_URL}) (also in this repo)")

tab1, tab2, tab3, tab4 = st.tabs(["① Zero-shot", "② Few-shot", "③ Fine-tuned", "④ Image"])

with tab1:
    st.header("① Zero-shot")
    st.markdown(
        "Zero-shot style transfer describes the target style in instructions alone, with no examples. "
        "Pick a style preset, edit the instructions if you like, and see what the model makes of them. "
        "The style checklist shows which style features the output has and whether the content was kept."
    )
    preset = PRESETS[st.selectbox("Style preset", list(PRESETS), key="t1_preset")]
    instructions = st.text_area("Instructions (system prompt)", preset.instructions,
                                key=f"t1_instructions_{preset.name}")
    text = st.text_area("Input", preset.default_input, height=180, key=f"t1_input_{preset.name}")
    if st.button("Generate", type="primary", key="t1_go"):
        with st.spinner("Generating…"):
            st.session_state["t1"] = run_zero_shot(lm, preset, text, instructions, temperature=temperature)
    if result := st.session_state.get("t1"):
        show_result(result)

with tab2:
    st.header("② Few-shot")
    st.markdown(
        "Few-shot style transfer is zero-shot plus example pairs: input→output demonstrations placed in the "
        "prompt before the new input. Slide k to add examples and watch the prompt, its token count and the "
        "style checklist change together. At k = 0 the prompt is exactly the zero-shot prompt."
    )
    preset = PRESETS[st.selectbox("Style preset", list(PRESETS), key="t2_preset")]
    available = preset.example_pairs()
    max_k = min(8, len(available))
    k = st.slider("k (example pairs in the prompt)", 0, max_k, min(3, max_k), key=f"t2_k_{preset.name}")
    if max_k < 8:
        st.caption(f"This preset has {len(available)} example pairs (notebook 1's), so k stops at {max_k}.")
    elif preset.name == "Professional email":
        st.caption("Email example pairs come, in fixed order, from the dataset's 180 training pairs; "
                   "the 20 test pairs never appear in a prompt.")
    instructions = st.text_area("Instructions (system prompt)", preset.instructions,
                                key=f"t2_instructions_{preset.name}")
    pairs = []
    for i, pair in enumerate(available[:k], start=1):
        with st.expander(f"Example pair {i} (editable)"):
            pair_in = st.text_area("Input", pair.input, key=f"t2_pair_in_{preset.name}_{i}")
            pair_out = st.text_area("Output", pair.output, height=200, key=f"t2_pair_out_{preset.name}_{i}")
            pairs.append(ExamplePair(pair_in, pair_out))
    text = st.text_area("Input", preset.default_input, height=180, key=f"t2_input_{preset.name}")
    if st.button("Generate", type="primary", key="t2_go"):
        with st.spinner("Generating…"):
            st.session_state["t2"] = run_few_shot(lm, preset, text, k, instructions, example_pairs=pairs,
                                                  temperature=temperature)
    if result := st.session_state.get("t2"):
        show_result(result)

# ---------- ③ Fine-tuned ----------
from style_lab.adapter import (  # noqa: E402
    ADAPTER_BASE_MODEL, ADAPTER_DIR, TRAIN_COMMAND, adapter_available, load_loss_history)
from style_lab.dataset import train_test_split  # noqa: E402
from style_lab.pipelines import (  # noqa: E402
    THREE_WAY_K, fine_tuned_inputs, run_fine_tuned_comparison, run_three_way_comparison)
from style_lab.text_model import AdaptedQwenTextModel  # noqa: E402

FREE_TEXT = "Type your own notes…"
FREE_TEXT_DEFAULT = "schedule a meeting with John about the Q4 planning"  # notebook 2's free-text test


@st.cache_resource(show_spinner="Loading the 0.5B base model with the adapter…")
def get_adapted_model() -> AdaptedQwenTextModel:
    # Its own copy of the 0.5B weights: loading an adapter modifies the model it is loaded into,
    # so sharing the sidebar's model would change tabs ① and ② too. Loaded only on first Generate.
    return AdaptedQwenTextModel(ADAPTER_BASE_MODEL, ADAPTER_DIR)


with tab3:
    st.header("③ Fine-tuned")
    st.markdown(
        "Fine-tuning teaches the style through training instead of through the prompt. A small **adapter** "
        "(LoRA, ~4 MB) was trained on the dataset's 180 training pairs on top of the 0.5B **base model**. "
        "Both models below get the same short prompt (system prompt + notes, no examples): any difference "
        "is what training added. Inputs are the 20 **test pairs**, which the adapter never saw."
    )
    if model_id != ADAPTER_BASE_MODEL:
        st.info(f"The sidebar says **{model_id}**, but this tab always uses **{ADAPTER_BASE_MODEL}**: an adapter "
                "only works with the base model it was trained on (it stores small changes to that model's weights).")
    st.caption("The adapter is email-only, so this tab has no style preset dropdown.")

    if not adapter_available():
        st.warning(f"**No adapter found** in `{ADAPTER_DIR.relative_to(Path(__file__).parent)}`. Train it with\n\n"
                   f"```\n{TRAIN_COMMAND}\n```\n\nIt takes about 80 minutes on an M2 CPU and writes the adapter "
                   "and its loss history there. Then reload this page.")
    else:
        test_pairs = fine_tuned_inputs()
        labels = [f"Test pair {i}: {p.input.splitlines()[0][:70]}" for i, p in enumerate(test_pairs, start=1)]
        choice = st.selectbox("Input", [*labels, FREE_TEXT], key="t3_choice")
        if choice == FREE_TEXT:
            text, reference = st.text_area("Your notes", FREE_TEXT_DEFAULT, height=120, key="t3_free"), None
        else:
            pair = test_pairs[labels.index(choice)]
            text, reference = pair.input, pair.output
            st.code(text, language=None, wrap_lines=True)

        if st.button("Generate base vs fine-tuned", type="primary", key="t3_go"):
            tuned = get_adapted_model()
            with st.spinner("Generating with the base model, then with the adapter…"):
                st.session_state["t3"] = run_fine_tuned_comparison(
                    tuned.base_view(), tuned, text, reference=reference, temperature=temperature)
        if comparison := st.session_state.get("t3"):
            left, right = st.columns(2)
            with left:
                st.subheader("Base model")
                show_result(comparison.base)
            with right:
                st.subheader("Base model + adapter")
                show_result(comparison.fine_tuned)
            if comparison.reference is not None:
                with st.expander("Reference answer (from the test pair)", expanded=True):
                    st.code(comparison.reference, language=None, wrap_lines=True)

        st.subheader("Compare all three techniques")
        st.markdown(f"Zero-shot, few-shot (k = {THREE_WAY_K}) and fine-tuned on the same input, all on the 0.5B "
                    "model. Watch the prompt token counts: zero-shot is cheap, few-shot is expensive, "
                    "fine-tuned is cheap again, yet keeps the style.")
        if st.button("Run the three-way comparison", key="t3_three_go"):
            tuned = get_adapted_model()
            with st.spinner("Generating three outputs…"):
                st.session_state["t3_three"] = run_three_way_comparison(
                    tuned.base_view(), tuned, text, temperature=temperature)
        if results := st.session_state.get("t3_three"):
            for col, result in zip(st.columns(3), results):
                with col:
                    st.subheader(result.technique)
                    show_result(result)

        st.subheader("What training did")
        if history := load_loss_history():
            cfg = history["config"]
            st.markdown("Training loss: how wrong the model's next-token guesses on the reference emails were "
                        "(lower = closer). It falls fast because the dataset's emails follow a few strict templates.")
            st.line_chart({"loss": [s["loss"] for s in history["steps"]]}, x_label="optimizer step",
                          y_label="training loss")
            st.caption(f"{cfg['train_pairs']} training pairs · {cfg['epochs']} epochs · LoRA r={cfg['lora_r']} on "
                       f"{', '.join(cfg['target_modules'])} · lr {cfg['learning_rate']} · effective batch "
                       f"{cfg['batch_size'] * cfg['grad_accum']} · final loss {history['steps'][-1]['loss']}")
        else:
            st.info("No loss history found next to the adapter.")
        train = train_test_split().train
        n = st.slider("Browse training pairs", 1, len(train), 1, key="t3_browse",
                      help="The adapter learned from these 180 pairs (the 20 test pairs above were held out).")
        left, right = st.columns(2)
        left.markdown("**Notes (input)**")
        left.code(train[n - 1].input, language=None, wrap_lines=True)
        right.markdown("**Email (output)**")
        right.code(train[n - 1].output, language=None, wrap_lines=True)

# ---------- ④ Image ----------
import io  # noqa: E402

from PIL import Image  # noqa: E402

from style_lab.depth import DepthEstimator  # noqa: E402
from style_lab.image_grid import (  # noqa: E402
    COMPARISON_GUIDANCE, COMPARISON_STRENGTH, GUIDANCES, MANIFEST_PATH, REFERENCE_IMAGES, STRENGTHS,
    GridImage, GridIndex)
from style_lab.live_generation import DepthImg2Img, live_generation_available  # noqa: E402

UPLOAD = "Upload your own…"
GRID_HOWTO = ("The image grid hasn't been rendered yet, so the images below are placeholders. Rendering needs a GPU: "
              "follow the how-to-run at the top of `scripts/render_grid.py` on a free Colab T4 (~15 min), then "
              "commit the resulting `data/grid/` folder. The depth map above already works.")


@st.cache_resource(show_spinner="Loading Depth Anything V2 Small (first run downloads ~100 MB)…")
def get_depth_estimator(device: str) -> DepthEstimator:
    return DepthEstimator(device)


@st.cache_resource(show_spinner="Loading SD1.5 + depth ControlNet (first run downloads ~5 GB)…")
def get_live_generator(device: str) -> DepthImg2Img:
    return DepthImg2Img(device)


@st.cache_data(show_spinner="Computing the depth map…", max_entries=8)
def depth_map_for(image_bytes: bytes, device: str) -> Image.Image:
    return get_depth_estimator(device).depth_map(Image.open(io.BytesIO(image_bytes)))


def load_for_tab4(data: bytes) -> tuple[Image.Image, bytes]:
    """Shrink to at most 512 px (the grid's size); returns the image and its JPEG bytes (the depth cache key)."""
    image = Image.open(io.BytesIO(data)).convert("RGB")
    image.thumbnail((512, 512))
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=90)
    return image, buf.getvalue()


def show_grid_image(found, caption: str) -> None:
    if isinstance(found, GridImage):
        st.image(str(found.path), caption=caption, width="stretch")
        st.caption(f"seed {found.seed} · {found.steps} steps · strength {found.strength} · guidance {found.guidance}"
                   f" · depth conditioning {'on' if found.depth_conditioned else 'off'}  \n"
                   + " · ".join(f"{k}: `{v}`" for k, v in found.models.items()))
    else:
        st.info(f"🖼️ **Placeholder: {caption}**\n\nImage {found.reason}. Render it with `scripts/render_grid.py`.")


with tab4:
    st.header("④ Image")
    st.markdown(
        "Image style transfer keeps a reference image's **content** (its composition: what is near, what is far) "
        "and changes its **style** to match a text prompt. The composition is carried by a **depth map**, and "
        "**depth conditioning** steers generation with it. Start with the depth map, then explore how "
        "strength and guidance shape the result."
    )
    gpu = pick_device()
    depth_device = "cuda" if gpu.startswith("cuda") else "cpu"  # the depth model is small; CPU is fine

    st.subheader("1. Reference image and its depth map")
    labels = {f"{r.title}, {r.artist}": key for key, r in REFERENCE_IMAGES.items()}
    choice = st.radio("Reference image", [*labels, UPLOAD], horizontal=True, key="t4_ref")
    grid_key, image, image_bytes, depth = None, None, None, None
    if choice == UPLOAD:
        upload = st.file_uploader("Your image (JPEG or PNG)", type=["jpg", "jpeg", "png"], key="t4_upload")
        if upload:
            image, image_bytes = load_for_tab4(upload.getvalue())
    else:
        grid_key = labels[choice]
        image, image_bytes = load_for_tab4(REFERENCE_IMAGES[grid_key].path.read_bytes())
    if image is None:
        st.caption("Upload an image to see its depth map.")
    else:
        left, right = st.columns(2)
        left.image(image, caption="Reference image", width="stretch")
        depth = depth_map_for(image_bytes, depth_device)
        right.image(depth, caption="Depth map (Depth Anything V2 Small; brighter = nearer)", width="stretch")
        if grid_key:
            ref = REFERENCE_IMAGES[grid_key]
            st.caption(f"*{ref.title}* by {ref.artist}, public domain, via [Wikimedia Commons]({ref.source_url}).")

    st.subheader("2. Explore strength and guidance")
    if grid_key is None:
        grid_key = st.selectbox("The grid is precomputed for the two reference images; pick one",
                                list(REFERENCE_IMAGES), format_func=lambda k: REFERENCE_IMAGES[k].title,
                                key="t4_grid_ref")
    index = GridIndex.load(MANIFEST_PATH)
    if index.is_empty:
        st.warning(GRID_HOWTO)
    prompt = st.radio("Prompt", REFERENCE_IMAGES[grid_key].prompts, key=f"t4_prompt_{grid_key}")

    controls, shown, notes = st.columns([1, 2, 1.2])
    with controls:
        strength = st.select_slider("Strength", STRENGTHS, value=COMPARISON_STRENGTH, key="t4_strength",
                                    help="Snaps to the precomputed values.")
        guidance = st.select_slider("Guidance", GUIDANCES, value=COMPARISON_GUIDANCE, key="t4_guidance",
                                    help="Snaps to the precomputed values.")
    with shown:
        show_grid_image(index.lookup(grid_key, prompt, strength, guidance), f"strength {strength}, guidance {guidance}")
    with notes:
        st.markdown(
            "**Strength** is how far generation may move away from the reference image's pixels: 0 would return "
            "it unchanged, 1 would ignore it. Low values keep its colours and brushwork; high values keep little "
            "more than what the depth map carries.\n\n"
            "**Guidance** is how closely generation follows the prompt. Low values drift freely and look softer; "
            "high values follow the prompt harder, often with harsher, oversaturated colours."
        )

    if st.toggle("Show the full 4 × 3 grid", key="t4_full_grid"):
        st.caption("Rows: strength (moving further from the reference image). "
                   "Columns: guidance (following the prompt more closely).")
        for s in STRENGTHS:
            for col, g in zip(st.columns(len(GUIDANCES)), GUIDANCES):
                with col:
                    show_grid_image(index.lookup(grid_key, prompt, s, g), f"strength {s}, guidance {g}")

    st.subheader("3. What the depth map adds")
    st.markdown(f"Same prompt, seed and steps at strength {COMPARISON_STRENGTH} / guidance {COMPARISON_GUIDANCE}, "
                "with and without depth conditioning. Without it, only the reference image's noised pixels hold "
                "the composition, so figures, horizon and perspective drift.")
    with_col, without_col = st.columns(2)
    with with_col:
        show_grid_image(index.lookup(grid_key, prompt, COMPARISON_STRENGTH, COMPARISON_GUIDANCE, True),
                        "with depth conditioning")
    with without_col:
        show_grid_image(index.lookup(grid_key, prompt, COMPARISON_STRENGTH, COMPARISON_GUIDANCE, False),
                        "without depth conditioning")

    st.subheader("4. Generate live (GPU only)")
    if not live_generation_available(gpu):
        st.caption(f"Live generation needs a GPU device (running on **{gpu}**; set `{DEVICE_ENV_VAR}=cuda`). "
                   "On a laptop CPU one image takes many minutes and SD1.5 does not fit comfortably in 8 GB, "
                   "which is why this tab uses a precomputed grid (ADR 0002).")
    elif image is None:
        st.caption("Pick or upload a reference image first.")
    else:
        live_prompt = st.text_input("Prompt", prompt, key="t4_live_prompt")
        live_depth = st.checkbox("Depth conditioning", True, key="t4_live_depth")
        st.caption(f"Uses the reference image above at strength {strength} / guidance {guidance}.")
        if st.button("Generate live", type="primary", key="t4_live_go"):
            with st.spinner("Generating…"):
                st.session_state["t4_live"] = get_live_generator(gpu).generate(
                    image, depth, live_prompt, strength, guidance, depth_conditioned=live_depth)
        if (live := st.session_state.get("t4_live")) is not None:
            st.image(live, caption=live_prompt, width="stretch")

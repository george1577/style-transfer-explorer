"""Style Transfer Explorer: the Streamlit UI. The only file that knows about Streamlit."""
from pathlib import Path

import streamlit as st

from style_lab.pipelines import TextResult, run_zero_shot
from style_lab.presets import PRESETS
from style_lab.text_model import DEVICE_ENV_VAR, MODELS, QwenTextModel, pick_device

DOC_LINK = "docs/how-style-transfer-works.md"
RESULT_KEYS = ("t1",)

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
    st.markdown("**Learning path**\n1. Zero-shot\n\n"
                f"Read the companion guide: `{DOC_LINK}`")

tab1, tab4 = st.tabs(["① Zero-shot", "④ Image"])

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

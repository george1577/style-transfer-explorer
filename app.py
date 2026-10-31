"""Style Transfer Explorer: the Streamlit UI. The only file that knows about Streamlit."""
from pathlib import Path

import streamlit as st

from style_lab.pipelines import TextResult, run_few_shot, run_zero_shot
from style_lab.presets import PRESETS
from style_lab.prompts import ExamplePair
from style_lab.text_model import DEVICE_ENV_VAR, MODELS, QwenTextModel

DOC_LINK = "docs/how-style-transfer-works.md"
RESULT_KEYS = ("t1", "t2")

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
    st.markdown("**Learning path**\n1. Zero-shot\n2. Few-shot\n\n"
                f"Read the companion guide: `{DOC_LINK}`")

tab1, tab2 = st.tabs(["① Zero-shot", "② Few-shot"])

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

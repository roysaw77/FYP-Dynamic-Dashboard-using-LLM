"""
main.py — Streamlit entry point and sidebar page router
=======================================================
The app is split into two sidebar pages that share one session:

  📊 Dashboard         PandasAI generation, generated SQL, correctness, latency
                       (generation logic lives in dashboard.py)
  🔎 Analysis Insight   AutoGen multi-agent discussion over exported charts
                       (page implementation lives in analysis.py)

Run from the repository root:
    venv\\Scripts\\python.exe -m streamlit run src/main.py
"""

import io
from pathlib import Path
import sys

import pandas as pd
import streamlit as st

# Make src/ importable no matter where Streamlit was launched from.
src_dir = str(Path(__file__).resolve().parent)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from dashboard import (
    DEFAULT_MODEL,
    MODELS,
    get_api_key,
    run_dashboard_analysis,
)


# ── Shared session state ──────────────────────────────────────────────────────

def _init_session_state() -> None:
    """Create the keys the pages rely on, once per session."""
    if "dataframes" not in st.session_state:
        st.session_state.dataframes = {}
    if "file_signatures" not in st.session_state:
        st.session_state.file_signatures = {}
    # Store last result for agent analysis
    if "last_result" not in st.session_state:
        st.session_state.last_result = None
    if "last_question" not in st.session_state:
        st.session_state.last_question = ""


def _render_result_block(result: dict) -> None:
    """Render a stored PandasAI result in the main page."""
    st.subheader("Response")
    if result.get("is_chart_export") and hasattr(result.get("response_obj"), "value"):
        st.image(result["response_obj"].value)
    else:
        st.code(result.get("response_text", ""))

    st.subheader("Generated Code")
    st.code(result.get("generated_code", "No code generated."))

    st.subheader("Correctness")
    st.code(result.get("correctness", "UNKNOWN"))

    st.subheader("Response time")
    st.code(f"{result.get('response_time', 0.0):.2f} seconds")


# ── Page: 📊 Dashboard (generation) ───────────────────────────────────────────

def render_dashboard_page() -> None:
    """PandasAI generation page — the original main.py UI."""
    st.set_page_config(page_title="PandasAI CSV", page_icon="💬", layout="wide")
    st.title("PandasAI CSV")

    # File Uploader
    uploaded_files = st.file_uploader(
        "Upload CSV files", type=["csv"], accept_multiple_files=True, key="csv_uploader_main"
    )

    uploaded_names = []
    if uploaded_files:
        for file in uploaded_files:
            uploaded_names.append(file.name)
            signature = f"{file.name}:{file.size}"
            if st.session_state.file_signatures.get(file.name) != signature:
                data = io.BytesIO(file.getvalue())
                st.session_state.dataframes[file.name] = pd.read_csv(data)
                st.session_state.file_signatures[file.name] = signature

        # Evict files that were un-uploaded
        for name in list(st.session_state.dataframes.keys()):
            if name not in uploaded_names:
                st.session_state.dataframes.pop(name, None)
                st.session_state.file_signatures.pop(name, None)

    # Dataset selection
    selected_files = st.multiselect(
        "Datasets for PandasAI",
        options=uploaded_names,
        default=uploaded_names,
    )

    # Model and question inputs
    model_keys = list(MODELS.keys())
    default_idx = model_keys.index(DEFAULT_MODEL) if DEFAULT_MODEL in model_keys else 0

    model_id = st.selectbox(
        "Model",
        options=model_keys,
        format_func=lambda x: MODELS[x],
        index=default_idx,
    )
    question = st.text_input("Question")
    api_key = get_api_key()

    # Execution (Dashboard page Run button)
    if st.button("Run", type="primary"):
        if not st.session_state.dataframes:
            st.error("Upload at least one CSV file.")
        elif not selected_files:
            st.error("Select at least one dataset.")
        elif not question.strip():
            st.error("Enter a question.")
        elif not api_key:
            st.error("NVIDIA API Key not found. Please set NVIDIA_API_KEY in .env (locally) or Streamlit Secrets (on Cloud).")
        else:
            selected_dataframes = [st.session_state.dataframes[name] for name in selected_files]
            try:
                with st.spinner("Running PandasAI and validating result..."):
                    result = run_dashboard_analysis(
                        question=question,
                        dataframes=selected_dataframes,
                        model_id=model_id.strip() if model_id else None,
                        api_key=api_key,
                    )
                # Persist result in session state so it survives reruns
                st.session_state.last_result = result
                st.session_state.last_question = question
            except Exception as exc:
                st.error(str(exc))

    # ── Dashboard result (from Run button) ───────────────────────────────────
    if st.session_state.last_result is not None:
        _render_result_block(st.session_state.last_result)


# ── Sidebar navigation ────────────────────────────────────────────────────────

_init_session_state()

nav = st.navigation(
    [
        st.Page(render_dashboard_page, title="Dashboard", icon="📊", default=True),
        st.Page("analysis.py", title="Analysis Insight", icon="🔎"),
    ],
    position="sidebar",
)
nav.run()

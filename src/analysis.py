"""
analysis.py  —  AutoGen Multi-Agent Chart & Result Analysis
=============================================================
Uses Microsoft AutoGen (pyautogen 0.2.x) to run a structured discussion
among specialised agents after PandasAI produces a result.

Agent Roles
-----------
ChartAnalyst   : Describes what the chart / data shows visually and factually.
DataInterpreter: Draws deeper data insights and statistical observations.
Critic         : Challenges the findings, highlights limitations / edge-cases.
Summarizer     : Synthesises the debate into a final actionable conclusion.
Coordinator    : UserProxyAgent that seeds the discussion (human_input_mode=NEVER).

Public APIs
-----------
  render_agent_analysis(question, result, api_key)  – embedded agent panel
  render_analysis_page()                             – the 🔎 Analysis Insight page; registered in
                                                       main.py's sidebar navigation and still runnable
                                                       on its own with `streamlit run src/analysis.py`
"""

from __future__ import annotations

import base64
import io
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st
from dotenv import load_dotenv
from litellm import completion as litellm_completion

# ── Environment ───────────────────────────────────────────────────────────────
load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=True)

# NVIDIA NIM OpenAI-compatible base URL
NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"

# Models for AutoGen (OpenAI-compatible)
VISION_MODEL_ID  = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"
TEXT_MODEL_ID    = "nvidia/nemotron-3.5-lightning-30b-a3b"

AVAILABLE_MODELS = {
    "nvidia/nemotron-3.5-lightning-30b-a3b":       "Nemotron-3.5-30B",
    "meta/llama-3.1-405b-instruct":                "Llama-3.1-405B",
    "meta/llama-3.1-8b-instruct":                  "Llama-3.1-8B",
    "mistralai/mistral-large-3-675b-instruct-2512":"Mistral-Large-3-675B",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_api_key() -> str:
    key = os.getenv("NVIDIA_API_KEY", "").strip()
    if not key:
        try:
            if hasattr(st, "secrets") and "NVIDIA_API_KEY" in st.secrets:
                key = st.secrets["NVIDIA_API_KEY"].strip()
        except Exception:
            pass
    return key


def _image_to_data_uri(image) -> str:
    """
    Return a base64 data URI for a filesystem path, raw bytes, or a Streamlit
    UploadedFile (which is what the chart uploader hands us).
    """
    if hasattr(image, "getvalue"):  # st.UploadedFile
        raw = image.getvalue()
        mime = getattr(image, "type", None) or "image/png"
    elif isinstance(image, (bytes, bytearray)):
        raw = bytes(image)
        mime = "image/png"
    else:
        path = Path(image)
        raw = path.read_bytes()
        mime = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
        }.get(path.suffix.lower(), "image/png")
    return f"data:{mime};base64,{base64.b64encode(raw).decode()}"


# ── AutoGen Configuration ─────────────────────────────────────────────────────

def _build_llm_config(model_id: str, api_key: str) -> Dict[str, Any]:
    """Build AutoGen llm_config using NVIDIA NIM OpenAI-compatible endpoint."""
    return {
        "config_list": [{
            "model":    model_id,
            "api_key":  api_key,
            "base_url": NIM_BASE_URL,
        }],
        "temperature": 0.4,
        "cache_seed":  None,   # disable disk cache so each run is fresh
    }


# ── AutoGen Multi-Agent Panel ─────────────────────────────────────────────────

def run_autogen_discussion(
    topic: str,
    context: str,
    model_id: str,
    api_key: str,
    max_rounds: int = 6,
) -> List[Dict[str, str]]:
    """
    Spin up an AutoGen GroupChat with 4 specialised agents + 1 coordinator.

    Parameters
    ----------
    topic    : Short title of what is being analysed.
    context  : The full text context for the agents (chart description / data).
    model_id : NVIDIA NIM model identifier (without nvidia_nim/ prefix).
    api_key  : NVIDIA API key.
    max_rounds: Maximum discussion rounds.

    Returns
    -------
    List of {name, content} dicts representing the conversation history.
    """
    import autogen

    llm_cfg = _build_llm_config(model_id, api_key)
    
    

    # ── Agent Definitions ─────────────────────────────────────────────────────
    chart_analyst = autogen.AssistantAgent(
        name="ChartAnalyst",
        system_message=(
            "You are a Chart Analyst specialising in ML/NLP benchmark visualisations. "
            "Your role is to ACCURATELY DESCRIBE what the data or chart shows: "
            "axes, model names, values, trends. Be factual and precise. "
            "Do NOT make up numbers. Respond in 3-5 concise sentences."
        ),
        llm_config=llm_cfg,
    )

    data_interpreter = autogen.AssistantAgent(
        name="DataInterpreter",
        system_message=(
            "You are a Data Interpreter specialising in Text-to-SQL system evaluation. "
            "After the ChartAnalyst describes the data, you draw deeper insights: "
            "What do the numbers actually mean for practitioners? "
            "Are there surprising correlations or patterns? "
            "Respond in 3-5 sentences with specific references to the numbers mentioned."
        ),
        llm_config=llm_cfg,
    )

    critic = autogen.AssistantAgent(
        name="Critic",
        system_message=(
            "You are a rigorous AI Research Critic. "
            "Your role is to challenge the previous analyses: "
            "What limitations exist in the benchmark? Are the conclusions over-stated? "
            "What confounding factors might explain the results? "
            "Be constructive but skeptical. Respond in 3-4 sentences."
        ),
        llm_config=llm_cfg,
    )

    summarizer = autogen.AssistantAgent(
        name="Summarizer",
        system_message=(
            "You are a Technical Summarizer. Your role is to synthesise the discussion "
            "into a final verdict: Which model wins and why? What should a practitioner do "
            "given these results? Provide 2-3 actionable recommendations. "
            "End your message with the word TERMINATE."
        ),
        llm_config=llm_cfg,
    )

    # Coordinator seeds the conversation; never asks for human input
    coordinator = autogen.UserProxyAgent(
        name="Coordinator",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=0,
        code_execution_config=False,
        is_termination_msg=lambda msg: "TERMINATE" in str(msg.get("content", "")),
    )

    # ── Group Chat ────────────────────────────────────────────────────────────
    group_chat = autogen.GroupChat(
        agents=[coordinator, chart_analyst, data_interpreter, critic, summarizer],
        messages=[],
        max_round=max_rounds,
        speaker_selection_method="round_robin",
        allow_repeat_speaker=False,
    )

    manager = autogen.GroupChatManager(
        groupchat=group_chat,
        llm_config=llm_cfg,
        is_termination_msg=lambda msg: "TERMINATE" in str(msg.get("content", "")),
    )

    # Seed message from coordinator
    seed_msg = (
        f"## Analysis Topic: {topic}\n\n"
        f"{context}\n\n"
        "ChartAnalyst, please begin by describing what you see in the data above."
    )

    coordinator.initiate_chat(
        manager,
        message=seed_msg,
        clear_history=True,
    )

    # Collect messages excluding the coordinator's seed
    history = []
    for msg in group_chat.messages:
        name = msg.get("name") or msg.get("role", "Unknown")
        content = str(msg.get("content", "")).replace("TERMINATE", "").strip()
        if name == "Coordinator" or not content:
            continue
        history.append({"name": name, "content": content})

    return history


# ── Vision: describe chart with LiteLLM before AutoGen discussion ──────────────

def _describe_chart_with_vision(
    image,
    question: str,
    api_key: str,
) -> str:
    """
    Use a vision model (via LiteLLM) to produce a textual description of the
    chart that AutoGen text agents can then discuss.

    ``image`` may be a path, raw bytes, or a Streamlit UploadedFile.
    """
    data_uri = _image_to_data_uri(image)
    messages = [{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": data_uri}},
            {
                "type": "text",
                "text": (
                    f"The user asked: \"{question}\"\n\n"
                    "Please provide a detailed description of this chart image: "
                    "what is plotted on each axis, which models appear, their exact values, "
                    "and any visible trends. Be precise and exhaustive — this description "
                    "will be passed to other AI agents for deeper analysis."
                ),
            },
        ],
    }]
    resp = litellm_completion(
        model=f"nvidia_nim/{VISION_MODEL_ID}",
        messages=messages,
        api_key=api_key,
    )
    return resp.choices[0].message.content.strip()


# ── Agent Name Styling ────────────────────────────────────────────────────────

AGENT_STYLES = {
    "ChartAnalyst":    {"emoji": "🔍", "color": "#60a5fa", "bg": "rgba(96,165,250,0.08)", "border": "rgba(96,165,250,0.3)"},
    "DataInterpreter": {"emoji": "📈", "color": "#34d399", "bg": "rgba(52,211,153,0.08)", "border": "rgba(52,211,153,0.3)"},
    "Critic":          {"emoji": "🎯", "color": "#f87171", "bg": "rgba(248,113,113,0.08)", "border": "rgba(248,113,113,0.3)"},
    "Summarizer":      {"emoji": "✅", "color": "#a78bfa", "bg": "rgba(167,139,250,0.08)", "border": "rgba(167,139,250,0.3)"},
}

def _render_message(name: str, content: str, idx: int) -> str:
    style = AGENT_STYLES.get(name, {"emoji": "🤖", "color": "#334155", "bg": "rgba(148,163,184,0.10)", "border": "rgba(148,163,184,0.35)"})
    return (
        f'<div style="background:{style["bg"]};border:1px solid {style["border"]};'
        f'border-radius:12px;padding:1rem 1.4rem;margin-bottom:1rem;">'
        f'<div style="color:{style["color"]};font-weight:600;font-size:0.9rem;margin-bottom:0.5rem;">'
        f'{style["emoji"]} {name}</div>'
        f'<div style="color:#1f2937;font-size:0.93rem;line-height:1.75;white-space:pre-wrap;">{content}</div>'
        f'</div>'
    )


# ── Streamlit Component — embedded in main.py ─────────────────────────────────

def render_agent_analysis(
    *,
    question: str,
    result: dict,
    api_key: Optional[str] = None,
) -> None:
    """
    Render an AutoGen multi-agent discussion panel inside main.py
    after PandasAI has produced a result.

    Parameters
    ----------
    question : Original user question.
    result   : Dict from run_dashboard_analysis() with keys:
               is_chart_export, response_obj, response_text, generated_code.
    api_key  : NVIDIA API key (falls back to env var).
    """
    api_key = api_key or _get_api_key()

    # CSS (injected once)
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        .ag-panel {
            background: rgba(15,12,41,0.6);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 16px; padding: 1.5rem 2rem; margin-top: 1.5rem;
        }
        .ag-header {
            font-size: 1.2rem; font-weight: 700;
            background: linear-gradient(90deg,#a78bfa,#60a5fa);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            background-clip: text; margin-bottom: 0.3rem;
        }
        .ag-sub { color: rgba(255,255,255,0.45); font-size:0.82rem; margin-bottom:1.2rem; }
        .chip-row { display:flex;gap:0.5rem;flex-wrap:wrap;margin-top:0.7rem; }
        .chip {
            background:rgba(139,92,246,0.15);border:1px solid rgba(139,92,246,0.35);
            border-radius:999px;padding:0.15rem 0.7rem;font-size:0.78rem;color:#c4b5fd;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown('<div class="ag-panel">', unsafe_allow_html=True)
    st.markdown('<div class="ag-header">🤖 AutoGen Multi-Agent Analysis</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="ag-sub">4 specialised AI agents discuss and debate your result: '
        'ChartAnalyst → DataInterpreter → Critic → Summarizer</div>',
        unsafe_allow_html=True,
    )

    is_chart = (
        result.get("is_chart_export", False)
        and hasattr(result.get("response_obj"), "value")
    )
    chart_path = result["response_obj"].value if is_chart else None

    # Model selector
    col_m, col_r = st.columns([3, 1])
    with col_m:
        model_labels = list(AVAILABLE_MODELS.values())
        model_keys   = list(AVAILABLE_MODELS.keys())
        default_idx  = model_keys.index(TEXT_MODEL_ID) if TEXT_MODEL_ID in model_keys else 0
        sel_label    = st.selectbox(
            "Discussion model",
            model_labels,
            index=default_idx,
            key="autogen_model_sel",
        )
        chosen_model = model_keys[model_labels.index(sel_label)]

    with col_r:
        max_rounds = st.number_input(
            "Rounds", min_value=4, max_value=12, value=6, step=2, key="autogen_rounds"
        )

    extra_q = st.text_input(
        "Optional: focus question for the agents",
        placeholder="e.g. Should we prefer Mistral over Llama for hard tasks?",
        key="autogen_focus_q",
    )

    go = st.button("🚀 Start Agent Discussion", key="autogen_run_btn", type="primary")
    st.markdown("</div>", unsafe_allow_html=True)

    if go:
        if not api_key:
            st.error("NVIDIA_API_KEY not found. Add it to .env or Streamlit secrets.")
            return

        # Build context for agents
        with st.spinner("🔍 Vision model reading the result…" if is_chart else "📋 Preparing context…"):
            if is_chart:
                try:
                    vision_desc = _describe_chart_with_vision(chart_path, question, api_key)
                    context = (
                        f"**Original Question:** {question}\n\n"
                        f"**Chart Description (from Vision Model):**\n{vision_desc}"
                    )
                    if extra_q.strip():
                        context += f"\n\n**Focus Question:** {extra_q.strip()}"
                except Exception as e:
                    st.error(f"Vision model failed: {e}")
                    return
            else:
                context = (
                    f"**Original Question:** {question}\n\n"
                    f"**PandasAI Result:**\n{result.get('response_text', '')}\n\n"
                    f"**Generated Code:**\n```python\n{result.get('generated_code', '')}\n```"
                )
                if extra_q.strip():
                    context += f"\n\n**Focus Question:** {extra_q.strip()}"

        topic = f"PandasAI Result — \"{question[:60]}{'…' if len(question) > 60 else ''}\""

        with st.spinner(f"🤖 AutoGen agents are discussing ({int(max_rounds)} rounds)…"):
            t0 = time.time()
            try:
                messages = run_autogen_discussion(
                    topic=topic,
                    context=context,
                    model_id=chosen_model,
                    api_key=api_key,
                    max_rounds=int(max_rounds),
                )
                elapsed = time.time() - t0
            except Exception as e:
                st.error(f"AutoGen error: {e}")
                return

        if not messages:
            st.warning("No messages returned from agents.")
            return

        # Render discussion
        st.markdown("---")
        st.markdown("#### 💬 Agent Discussion")

        if is_chart:
            with st.expander("📷 Vision Model Chart Description (context given to agents)", expanded=False):
                st.markdown(f"> {vision_desc}")

        html_blocks = "".join(
            _render_message(m["name"], m["content"], i)
            for i, m in enumerate(messages)
        )
        st.markdown(html_blocks, unsafe_allow_html=True)

        st.markdown(
            f'<div class="chip-row">'
            f'<span class="chip">⏱ {elapsed:.1f}s total</span>'
            f'<span class="chip">💬 {len(messages)} messages</span>'
            f'<span class="chip">🤖 {sel_label}</span>'
            f'<span class="chip">{"🖼 Chart" if is_chart else "📋 Tabular"}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )


# ── 🔎 Analysis Insight Page (sidebar page + standalone) ─────────────────────

def render_analysis_page() -> None:
    """
    🔎 Analysis Insight page — upload a chart image and let AutoGen agents debate it.

    Registered as a page in main.py's sidebar navigation
    (`streamlit run src/main.py`), and still runnable on its own:
        streamlit run src/analysis.py
    """
    st.set_page_config(page_title="Analysis Insight", page_icon="🔎", layout="wide")

    # Light theme: white page background, matching the Dashboard page.
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
        .stApp { background: #ffffff; }
        .hero { text-align:center; padding:2rem 1rem 1rem; }
        .hero h1 {
            font-size:2.4rem; font-weight:700;
            background:linear-gradient(90deg,#7c3aed,#2563eb,#059669);
            -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;
        }
        .hero p { color:#6b7280; font-size:1rem; }
        .chip-row { display:flex;gap:0.5rem;flex-wrap:wrap;margin-top:0.6rem; }
        .chip {
            background:#f5f3ff; border:1px solid #ddd6fe;
            border-radius:999px; padding:0.2rem 0.8rem; font-size:0.8rem; color:#6d28d9;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="hero">'
        '<h1>🔎 Analysis Insight</h1>'
        '<p>Upload a chart image — 4 AI agents will discuss and debate what it means.</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    # Sidebar
    with st.sidebar:
        st.markdown("## ⚙️ Settings")
        m_labels = list(AVAILABLE_MODELS.values())
        m_keys   = list(AVAILABLE_MODELS.keys())
        default_i = m_keys.index(TEXT_MODEL_ID) if TEXT_MODEL_ID in m_keys else 0
        sel_m    = st.selectbox("Discussion Model", m_labels, index=default_i)
        chosen_m = m_keys[m_labels.index(sel_m)]
        rounds   = st.slider("Discussion rounds", 4, 12, 6, 2)
        st.markdown("---")
        st.markdown(
            "**Agent Roles**\n\n"
            "🔍 **ChartAnalyst** — describes the data\n\n"
            "📈 **DataInterpreter** — extracts insights\n\n"
            "🎯 **Critic** — challenges the findings\n\n"
            "✅ **Summarizer** — final verdict\n"
        )

    api_key = _get_api_key()

    col_l, col_r = st.columns([1, 1], gap="large")

    with col_l:
        st.markdown("### 🖼️ Chart Image")
        uploaded_chart = st.file_uploader(
            "Upload a chart image",
            type=["png", "jpg", "jpeg", "webp"],
            accept_multiple_files=False,
            key="analysis_chart_upload",
        )
        st.caption("Drag an image here or click **Browse files** to pick one from your computer.")
        if uploaded_chart is not None:
            st.image(uploaded_chart, use_container_width=True)
            kb = uploaded_chart.size / 1024
            st.markdown(
                f'<div class="chip-row"><span class="chip">📁 {uploaded_chart.name}</span>'
                f'<span class="chip">💾 {kb:.1f} KB</span></div>',
                unsafe_allow_html=True,
            )

    with col_r:
        st.markdown("### 💬 Configure Discussion")

        user_prompt = st.text_area(
            "Topic",
            placeholder="e.g. Which model gives the best accuracy for the least latency?",
            height=130,
        )

        go = st.button("🚀 Start Agent Discussion", type="primary", use_container_width=True)

    # Results
    if go:
        if uploaded_chart is None:
            st.warning("Upload a chart image first.")
            return
        if not api_key:
            st.error("NVIDIA_API_KEY not found. Add it to .env or Streamlit secrets.")
            return
        if not user_prompt.strip():
            st.warning("Type a topic for the agents first.")
            return

        chart_name = uploaded_chart.name
        ph = st.empty()
        with ph.container():
            with st.spinner("🔍 Vision model reading the chart…"):
                try:
                    vision_desc = _describe_chart_with_vision(uploaded_chart, user_prompt, api_key)
                except Exception as e:
                    st.error(f"Vision model error: {e}")
                    return

            context = (
                f"**Chart:** {chart_name}\n\n"
                f"**Chart Description (Vision Model):**\n{vision_desc}"
            )

            with st.spinner(f"🤖 {rounds} rounds of agent discussion…"):
                t0 = time.time()
                try:
                    messages = run_autogen_discussion(
                        topic=user_prompt,
                        context=context,
                        model_id=chosen_m,
                        api_key=api_key,
                        max_rounds=rounds,
                    )
                    elapsed = time.time() - t0
                except Exception as e:
                    st.error(f"AutoGen error: {e}")
                    return

        if not messages:
            st.warning("No messages returned from agents.")
            return

        st.markdown("---")
        st.markdown("### 💬 Agent Discussion")

        with st.expander("📷 Vision Description (context fed to agents)", expanded=False):
            st.markdown(f"> {vision_desc}")

        html_blocks = "".join(
            _render_message(m["name"], m["content"], i)
            for i, m in enumerate(messages)
        )
        st.markdown(html_blocks, unsafe_allow_html=True)

        st.markdown(
            f'<div class="chip-row">'
            f'<span class="chip">⏱ {elapsed:.1f}s total</span>'
            f'<span class="chip">💬 {len(messages)} messages</span>'
            f'<span class="chip">🤖 {sel_m}</span>'
            f'<span class="chip">📊 {chart_name}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    render_analysis_page()

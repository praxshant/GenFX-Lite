"""
GenFX Lite — Streamlit UI
Full pipeline visualization: Prompt → JSON → Image → Blender Render.
Includes full health checks, structured diagnostics, and file extraction capabilities.
"""

import json
import logging
import sys
import shutil
from pathlib import Path
from PIL import Image
import io
import os

import streamlit as st

# ── Path Setup ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app import config
from app.pipeline import run_pipeline
from app.health import check_runtime_health

logging.basicConfig(level=logging.INFO)

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GenFX Lite — AI VFX Pipeline",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS Injection (single block, all styles) ──────────────────────────────────
st.markdown(
    """
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@300;400;500&display=swap" rel="stylesheet">

    <style>
    /* ── Design Tokens ─────────────────────────────────────── */
    :root {
        --bg-primary:    #0F0F0F;
        --bg-surface:    #1A1A1A;
        --bg-elevated:   #242424;
        --accent:        #E8D5B7;
        --accent-dim:    #8B7355;
        --text-primary:  #F0EDE8;
        --text-secondary:#8A8780;
        --text-tertiary: #4A4845;
        --border:        #2A2A2A;
        --green:         #4CAF7D;
        --amber:         #D4A853;
        --red:           #C0392B;
        --green-bg:      #1A3A2A;
        --amber-bg:      #3A2E1A;
        --red-bg:        #3A1A1A;
        --pending-bg:    #1F1F1F;
    }

    /* ── Global Reset ──────────────────────────────────────── */
    html, body, [class*="css"] {
        font-family: 'DM Mono', monospace;
        background-color: var(--bg-primary) !important;
        color: var(--text-primary) !important;
    }

    .main .block-container {
        max-width: 1200px;
        padding: 2rem 2rem 4rem 2rem;
        background: var(--bg-primary);
    }

    /* ── Sidebar ───────────────────────────────────────────── */
    [data-testid="stSidebar"] {
        background: var(--bg-surface) !important;
        border-right: 0.5px solid var(--border) !important;
        width: 250px !important;
    }
    [data-testid="stSidebar"] > div:first-child {
        padding: 2rem 1.25rem;
    }

    /* ── Typography ────────────────────────────────────────── */
    h1, h2, h3 {
        font-family: 'DM Serif Display', serif !important;
        color: var(--text-primary) !important;
        letter-spacing: -0.02em;
    }
    h1 { font-size: 2.8rem !important; line-height: 1.1 !important; }
    h2 { font-size: 1.6rem !important; }
    h3 { font-size: 1.2rem !important; }

    p, li, label, span {
        font-family: 'DM Mono', monospace !important;
        color: var(--text-secondary) !important;
    }

    /* ── Main Title Block ─────────────────────────────────── */
    .genfx-title {
        font-family: 'DM Serif Display', serif;
        font-size: 3.2rem;
        font-weight: 400;
        color: var(--text-primary);
        line-height: 1.05;
        margin-bottom: 0.25rem;
    }
    .genfx-subtitle {
        font-family: 'DM Mono', monospace;
        font-size: 0.82rem;
        color: var(--text-tertiary);
        letter-spacing: 0.18em;
        text-transform: uppercase;
        margin-bottom: 2rem;
    }

    /* ── Sidebar Wordmark ─────────────────────────────────── */
    .sidebar-wordmark {
        font-family: 'DM Mono', monospace;
        font-size: 1.1rem;
        font-weight: 500;
        color: var(--accent);
        letter-spacing: 0.25em;
        text-transform: uppercase;
    }
    .sidebar-subtitle {
        font-family: 'DM Mono', monospace;
        font-size: 0.72rem;
        color: var(--text-tertiary);
        letter-spacing: 0.1em;
        margin-top: 2px;
    }
    .sidebar-divider {
        border: none;
        border-top: 0.5px solid var(--border);
        margin: 1.2rem 0;
    }
    .sidebar-section-label {
        font-family: 'DM Mono', monospace;
        font-size: 0.68rem;
        color: var(--text-tertiary);
        letter-spacing: 0.15em;
        text-transform: uppercase;
        margin-bottom: 0.75rem;
    }

    /* ── Status Badges (sidebar) ─────────────────────────── */
    .status-row {
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 0.55rem;
        font-family: 'DM Mono', monospace;
        font-size: 0.78rem;
    }
    .status-dot-ok     { color: var(--green);  font-size: 1rem; }
    .status-dot-fb     { color: var(--amber);  font-size: 1rem; }
    .status-dot-pend   { color: var(--text-tertiary); font-size: 1rem; }
    .status-dot-run    { color: var(--accent); font-size: 1rem;
                         animation: pulse 1.2s ease-in-out infinite; }
    .status-label      { color: var(--text-secondary); }
    .status-val-ok     { color: var(--green);   margin-left: auto; font-size: 0.68rem; }
    .status-val-fb     { color: var(--amber);   margin-left: auto; font-size: 0.68rem; }
    .status-val-pend   { color: var(--text-tertiary); margin-left: auto; font-size: 0.68rem; }
    .status-val-run    { color: var(--accent);  margin-left: auto; font-size: 0.68rem; }

    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50%       { opacity: 0.25; }
    }

    /* ── Inline Status Badge (cards) ─────────────────────── */
    .badge {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        padding: 2px 10px;
        border-radius: 4px;
        font-family: 'DM Mono', monospace;
        font-size: 0.72rem;
        letter-spacing: 0.06em;
        font-weight: 500;
    }
    .badge-ok      { background: var(--green-bg);   color: var(--green); }
    .badge-fallback{ background: var(--amber-bg);   color: var(--amber); }
    .badge-pending { background: var(--pending-bg); color: var(--text-tertiary); }
    .badge-error   { background: var(--red-bg);     color: var(--red); }

    /* ── Output Cards ─────────────────────────────────────── */
    .output-card {
        background: var(--bg-surface);
        border: 0.5px solid var(--border);
        border-radius: 8px;
        padding: 24px;
        position: relative;
        overflow: hidden;
        box-shadow: 0 1px 3px rgba(0,0,0,0.4), 0 4px 16px rgba(0,0,0,0.2);
        transition: background 150ms ease, transform 150ms ease,
                    box-shadow 150ms ease;
        margin-bottom: 1rem;
    }
    .output-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: var(--accent);
        border-radius: 8px 8px 0 0;
    }
    .output-card:hover {
        background: var(--bg-elevated);
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.5), 0 8px 32px rgba(0,0,0,0.3);
    }
    .card-stage-num {
        font-family: 'DM Mono', monospace;
        font-size: 0.68rem;
        color: var(--text-tertiary);
        letter-spacing: 0.15em;
        text-transform: uppercase;
        margin-bottom: 4px;
    }
    .card-heading {
        font-family: 'DM Serif Display', serif;
        font-size: 1.25rem;
        color: var(--text-primary);
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 8px;
    }
    .card-caption {
        font-family: 'DM Mono', monospace;
        font-size: 0.72rem;
        color: var(--text-tertiary);
        margin-top: 0.6rem;
        letter-spacing: 0.04em;
    }
    .card-img {
        width: 100%;
        border-radius: 4px;
        display: block;
    }

    /* ── Input Area ───────────────────────────────────────── */
    textarea {
        background: var(--bg-surface) !important;
        border: 0.5px solid var(--border) !important;
        border-radius: 6px !important;
        color: var(--text-primary) !important;
        font-family: 'DM Mono', monospace !important;
        font-size: 0.9rem !important;
        box-shadow: inset 0 1px 4px rgba(0,0,0,0.3) !important;
        transition: border-color 150ms ease !important;
    }
    textarea:focus {
        border-color: var(--accent-dim) !important;
        outline: none !important;
    }

    /* ── Run Button ───────────────────────────────────────── */
    .stButton > button {
        width: 100%;
        background: var(--accent) !important;
        color: #0F0F0F !important;
        font-family: 'DM Mono', monospace !important;
        font-size: 0.82rem !important;
        font-weight: 500 !important;
        letter-spacing: 0.2em !important;
        text-transform: uppercase !important;
        border: none !important;
        border-radius: 6px !important;
        padding: 0.75rem 1.5rem !important;
        transition: filter 150ms ease, transform 150ms ease !important;
        cursor: pointer !important;
    }
    .stButton > button:hover {
        filter: brightness(1.12) !important;
        transform: translateY(-1px) !important;
    }
    
    .stButton > button:disabled {
        opacity: 0.5 !important;
        cursor: not-allowed !important;
        pointer-events: none !important;
    }

    /* ── Download Buttons Overrides ───────────────────────── */
    div[data-testid="stDownloadButton"] > button {
        background: var(--bg-elevated) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border) !important;
        font-family: 'DM Mono', monospace !important;
        letter-spacing: 0.05em !important;
        padding: 0.4rem 0.8rem !important;
        transition: border-color 150ms;
    }
    div[data-testid="stDownloadButton"] > button:hover {
        border-color: var(--accent) !important;
        color: var(--accent) !important;
    }

    /* ── JSON viewer override ─────────────────────────────── */
    [data-testid="stJson"] {
        background: var(--bg-primary) !important;
        border: 0.5px solid var(--border) !important;
        border-radius: 6px !important;
        font-family: 'DM Mono', monospace !important;
        font-size: 0.78rem !important;
    }

    /* ── Pipeline Flow Diagram ────────────────────────────── */
    .pipeline-flow {
        display: flex;
        align-items: center;
        justify-content: center;
        flex-wrap: wrap;
        gap: 0;
        padding: 1.5rem 0;
        font-family: 'DM Mono', monospace;
        font-size: 0.75rem;
        letter-spacing: 0.06em;
    }
    .flow-node {
        background: var(--bg-surface);
        border: 0.5px solid var(--border);
        border-radius: 6px;
        padding: 7px 14px;
        color: var(--text-secondary);
        white-space: nowrap;
    }
    .flow-node.active { border-color: var(--accent); color: var(--accent); }
    .flow-arrow {
        color: var(--text-tertiary);
        padding: 0 6px;
        font-size: 0.85rem;
    }

    /* ── Divider ─────────────────────────────────────────── */
    hr {
        border-color: var(--border) !important;
        margin: 1.5rem 0 !important;
    }
    
    /* ── Diagnostics Text ────────────────────────────────── */
    .diag-text {
        font-family: 'DM Mono', monospace;
        font-size: 0.75rem;
        color: var(--amber);
        margin-bottom: 5px;
    }

    /* ── Scrollbar ───────────────────────────────────────── */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: var(--bg-primary); }
    ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

    /* ── Hide Streamlit chrome ───────────────────────────── */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    [data-testid="stToolbar"] { display: none; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Health Checks ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def cached_health() -> dict:
    """Run full API + asset health probes once every 5 minutes."""
    return check_runtime_health()


def get_health() -> dict[str, bool]:
    """Legacy key-only check — kept for any existing references."""
    return {
        "openrouter": bool(config.OPENROUTER_API_KEY and config.OPENROUTER_API_KEY.startswith("sk-or")),
        "openai": bool(config.OPENAI_API_KEY and config.OPENAI_API_KEY.startswith("sk-")),
        "hf": bool(config.HUGGINGFACE_API_KEY and config.HUGGINGFACE_API_KEY.startswith("hf_")),
        "blender": bool(shutil.which(config.BLENDER_PATH)),
        "fallbacks": all(
            p.exists() for p in [
                config.FALLBACK_JSON_PATH,
                config.FALLBACK_IMAGE_PATH,
                config.FALLBACK_RENDER_PATH
            ]
        )
    }

# ── Helper HTML Generator ─────────────────────────────────────────────────────

def badge_html(status: str) -> str:
    """Return an HTML badge string for a given pipeline status value."""
    cfg = {
        "ok":       ("●", "badge-ok",       "OK"),
        "fallback": ("●", "badge-fallback",  "FALLBACK"),
        "pending":  ("○", "badge-pending",   "PENDING"),
        "running":  ("●", "badge-ok",        "RUNNING"),
        "error":    ("●", "badge-error",     "ERROR"),
    }
    dot, cls, label = cfg.get(status, ("○", "badge-pending", status.upper()))
    return f'<span class="badge {cls}">{dot} {label}</span>'


def sidebar_status_row(label: str, status: str) -> str:
    """Return an HTML sidebar status row for a pipeline stage."""
    dot_cls = {
        "ok":      "status-dot-ok",
        "fallback":"status-dot-fb",
        "pending": "status-dot-pend",
        "running": "status-dot-run",
    }.get(status, "status-dot-pend")

    val_cls = {
        "ok":      "status-val-ok",
        "fallback":"status-val-fb",
        "pending": "status-val-pend",
        "running": "status-val-run",
    }.get(status, "status-val-pend")

    dot = "●" if status in ("ok", "fallback", "running") else "○"

    return (
        f'<div class="status-row">'
        f'  <span class="{dot_cls}">{dot}</span>'
        f'  <span class="status-label">{label}</span>'
        f'  <span class="{val_cls}">{status.upper()}</span>'
        f'</div>'
    )


# ── Session State ─────────────────────────────────────────────────────────────
if "pipeline_result" not in st.session_state:
    st.session_state.pipeline_result = None
if "pipeline_status" not in st.session_state:
    st.session_state.pipeline_status = {"llm": "pending", "image": "pending", "render": "pending"}
if "is_running" not in st.session_state:
    st.session_state.is_running = False

def run_pipeline_action():
    if st.session_state.scene_prompt.strip():
        st.session_state.is_running = True
        st.session_state.pipeline_status = {"llm": "running", "image": "pending", "render": "pending"}


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<div class="sidebar-wordmark">GENFX</div>'
        '<div class="sidebar-subtitle">VFX Pipeline Prototype</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section-label">Pipeline Status</div>', unsafe_allow_html=True)

    status = st.session_state.pipeline_status
    st.markdown(
        sidebar_status_row("LLM Parser", status["llm"])
        + sidebar_status_row("Image Gen", status["image"])
        + sidebar_status_row("Blender", status["render"]),
        unsafe_allow_html=True,
    )

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

    # Live API + asset health (cached 5 min)
    rt = cached_health()
    st.markdown('<div class="sidebar-section-label">System Checks</div>', unsafe_allow_html=True)
    st.markdown(
        sidebar_status_row("OpenRouter API", "ok" if rt["llm"]["ok"] else "fallback")
        + sidebar_status_row("HF Image API", "ok" if rt["image"]["ok"] else "fallback")
        + sidebar_status_row("Blender Exec", "ok" if rt["blender"]["ok"] else "fallback")
        + sidebar_status_row("Assets Intact", "ok" if rt["assets"]["ok"] else "fallback"),
        unsafe_allow_html=True,
    )

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)
    st.markdown(
        '<div class="sidebar-section-label">Stack</div>'
        '<p style="font-size:0.72rem;color:#4A4845;line-height:1.8;">'
        "OpenAI GPT-4o-mini<br>"
        "HuggingFace SDXL<br>"
        "Blender 3.6+ (Cycles)<br>"
        "Streamlit 1.32+"
        "</p>",
        unsafe_allow_html=True,
    )

# ── Main Content ──────────────────────────────────────────────────────────────
st.markdown(
    '<h1 class="genfx-title">GenFX Lite</h1>'
    '<p class="genfx-subtitle">Prompt → LLM → JSON → SDXL → Image → Blender → Render</p>',
    unsafe_allow_html=True,
)

# Pipeline flow diagram (always visible)
st.markdown(
    '<div class="pipeline-flow">'
    '  <div class="flow-node">Prompt</div><span class="flow-arrow">→</span>'
    '  <div class="flow-node">LLM Parser</div><span class="flow-arrow">→</span>'
    '  <div class="flow-node">Scene JSON</div><span class="flow-arrow">→</span>'
    '  <div class="flow-node">SDXL</div><span class="flow-arrow">→</span>'
    '  <div class="flow-node">Image</div><span class="flow-arrow">→</span>'
    '  <div class="flow-node">Blender</div><span class="flow-arrow">→</span>'
    '  <div class="flow-node">Render</div>'
    '</div>',
    unsafe_allow_html=True,
)

st.markdown("<hr>", unsafe_allow_html=True)

# ── Input Section ─────────────────────────────────────────────────────────────
user_prompt = st.text_area(
    label="Describe your scene",
    placeholder="e.g. cinematic desert at golden hour with dust storms and dramatic light",
    height=100,
    label_visibility="visible",
    disabled=st.session_state.is_running,
    key="scene_prompt",
)

if st.button("Run Pipeline  →", key="run_btn", disabled=st.session_state.is_running):
    if user_prompt.strip():
        st.session_state.is_running = True
        st.session_state.pipeline_status = {"llm": "running", "image": "pending", "render": "pending"}
        st.rerun()
    else:
        st.warning("Please enter a scene description before running the pipeline.")

# ── Pipeline Execution Intercept ──────────────────────────────────────────────
if st.session_state.is_running:
    with st.spinner("Executing Pipeline Stages..."):
        result = run_pipeline(user_prompt.strip())
        st.session_state.pipeline_result = result
        st.session_state.pipeline_status = result["status"]
    st.session_state.is_running = False
    st.rerun()

# ── Results Display ───────────────────────────────────────────────────────────
result = st.session_state.pipeline_result

if result and not st.session_state.is_running:
    st.markdown("<hr>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3, gap="medium")

    # ── Card 1: Scene JSON ────────────────────────────────────────────────────
    with col1:
        s1 = result["status"].get("llm", "pending")
        st.markdown(
            f'<div class="output-card">'
            f'  <div class="card-stage-num">Stage 01</div>'
            f'  <div class="card-heading">Scene JSON {badge_html(s1)}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        if result.get("scene_json"):
            json_str = json.dumps(result["scene_json"], indent=2)
            st.json(result["scene_json"])
            st.download_button(
                label="⬇️ Download JSON",
                data=json_str,
                file_name=f"scene_{result.get('run_id', 'output')}.json",
                mime="application/json",
            )
            
        st.markdown(
            f'<div class="card-caption">Parsed securely via unified dataclass layer.</div>',
            unsafe_allow_html=True,
        )

    # ── Card 2: Generated Image ───────────────────────────────────────────────
    with col2:
        s2 = result["status"].get("image", "pending")
        st.markdown(
            f'<div class="output-card">'
            f'  <div class="card-stage-num">Stage 02</div>'
            f'  <div class="card-heading">Generated Image {badge_html(s2)}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        img_path = result.get("image_path")
        if img_path:
            resolved_img_path = Path(img_path).resolve()
            if resolved_img_path.exists():
                st.write("Image Size:", os.path.getsize(resolved_img_path))
                try:
                    with open(resolved_img_path, "rb") as f:
                        img_bytes = f.read()
                    img_obj = Image.open(io.BytesIO(img_bytes))
                    st.image(img_obj, width="stretch")
                except Exception as e:
                    st.error(f"Failed to load image: {e}")
                
                with open(resolved_img_path, "rb") as file:
                    st.download_button(
                        label="⬇️ Download Image",
                        data=file,
                        file_name=f"img_{result.get('run_id', 'output')}.png",
                        mime="image/png",
                    )
                
        st.markdown(
            '<div class="card-caption">SDXL validation logic complete.</div>',
            unsafe_allow_html=True,
        )

    # ── Card 3: Blender Render ────────────────────────────────────────────────
    with col3:
        s3 = result["status"].get("render", "pending")
        st.markdown(
            f'<div class="output-card">'
            f'  <div class="card-stage-num">Stage 03</div>'
            f'  <div class="card-heading">Blender Render {badge_html(s3)}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        render_path = result.get("render_path")
        if render_path:
            resolved_render_path = Path(render_path).resolve()
            
            # Debug info to prove the file is exactly where it should be
            st.write("Render path:", str(resolved_render_path))
            st.write("Exists:", resolved_render_path.exists())
            
            if resolved_render_path.exists():
                st.write("Size:", os.path.getsize(resolved_render_path))
                try:
                    with open(resolved_render_path, "rb") as f:
                        render_bytes = f.read()
                    render_obj = Image.open(io.BytesIO(render_bytes))
                    st.image(render_obj, width="stretch")
                except Exception as e:
                    st.error(f"Failed to load render: {e}")
                
                with open(resolved_render_path, "rb") as file:
                    st.download_button(
                        label="⬇️ Download Render",
                        data=file,
                        file_name=f"render_{result.get('run_id', 'output')}.png",
                        mime="image/png",
                    )
                
        st.markdown(
            '<div class="card-caption">Engine: Cycles · Samples: 32</div>',
            unsafe_allow_html=True,
        )

    # ── Pipeline summary bar & Diagnostics ────────────────────────────────────
    st.markdown("<hr>", unsafe_allow_html=True)
    total_fallbacks = sum(1 for v in result["status"].values() if v == "fallback")
    total_ok = sum(1 for v in result["status"].values() if v == "ok")

    summary_color = "#4CAF7D" if total_fallbacks == 0 else "#D4A853"
    summary_msg = (
        f"Pipeline complete — [{result.get('run_id')}] — {total_ok}/3 stages live, {total_fallbacks}/3 fallback."
        if total_fallbacks > 0
        else f"Pipeline complete — [{result.get('run_id')}] — all 3 stages live."
    )
    st.markdown(
        f'<p style="font-family:\'DM Mono\',monospace;font-size:0.78rem;'
        f'color:{summary_color};text-align:center;letter-spacing:0.06em;">'
        f"⬡ {summary_msg}</p>",
        unsafe_allow_html=True,
    )

    if total_fallbacks > 0:
        with st.expander("Diagnostics"):
            st.markdown("### Fallback Trace")
            diag = result.get("diagnostics", {})
            if result["status"]["llm"] == "fallback":
                st.markdown(f'<div class="diag-text">LLM Parser: {diag.get("llm")}</div>', unsafe_allow_html=True)
            if result["status"]["image"] == "fallback":
                st.markdown(f'<div class="diag-text">Image Generator: {diag.get("image")}</div>', unsafe_allow_html=True)
            if result["status"]["render"] == "fallback":
                st.markdown(f'<div class="diag-text">Blender Runner: {diag.get("render")}</div>', unsafe_allow_html=True)

else:
    if not st.session_state.is_running:
        # Empty state
        st.markdown(
            '<div style="text-align:center;padding:4rem 0;">'
            '  <p style="font-family:\'DM Mono\',monospace;font-size:0.85rem;'
            '     color:#2A2A2A;letter-spacing:0.12em;text-transform:uppercase;">'
            "     Enter a scene description and click Run Pipeline to begin."
            "  </p>"
            "</div>",
            unsafe_allow_html=True,
        )

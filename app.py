import html
import os
import sys
import traceback
from datetime import datetime
from typing import Any, Dict, Optional

import streamlit as st


# ------------------------------------------------------------------------
# Project path configuration
# ------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from rag.rag_pipeline import RAGPipeline, RAGResponse  # noqa: E402
from utils.config import settings  # noqa: E402
from utils.logger import get_logger  # noqa: E402


logger = get_logger(__name__)


# ------------------------------------------------------------------------
# Application constants
# ------------------------------------------------------------------------
DISCLAIMER = (
    "This assistant answers using only the documents you have uploaded and "
    "can make mistakes. It does not provide legal, medical, or financial "
    "advice and is not a substitute for contacting Aetna Member Services "
    "directly for coverage decisions."
)


# ------------------------------------------------------------------------
# Styling
# ------------------------------------------------------------------------
def _inject_custom_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --primary: #6D28D9;
            --primary-dark: #5521B5;
            --primary-soft: #F3EEFF;
            --primary-softer: #FAF8FF;

            --success: #15803D;
            --success-soft: #F0FDF4;

            --warning: #B45309;
            --warning-soft: #FFFBEB;

            --danger: #B91C1C;
            --danger-soft: #FEF2F2;

            --text-primary: #18181B;
            --text-secondary: #52525B;
            --text-muted: #71717A;

            --border: #E4E4E7;
            --border-soft: #F1F1F3;

            --surface: #FFFFFF;
            --surface-soft: #FAFAFB;

            --shadow-sm: 0 1px 2px rgba(24, 24, 27, 0.04);
            --shadow-md: 0 8px 24px rgba(24, 24, 27, 0.07);
            --shadow-primary: 0 8px 20px rgba(109, 40, 217, 0.18);
        }

        html,
        body,
        [class*="css"] {
            font-family:
                Inter,
                ui-sans-serif,
                -apple-system,
                BlinkMacSystemFont,
                "Segoe UI",
                sans-serif;
        }

        html {
            scroll-behavior: smooth;
        }

        body {
            color: var(--text-primary);
            background: var(--surface);
        }

        /* -------------------------------------------------------------
           Theme hardening
           Streamlit will fall back to a dark theme on some browsers/OS
           settings unless every surface is pinned explicitly. Without
           this, the fixed bottom chat bar renders black and sidebar
           text renders near-white (invisible) against our light
           backgrounds.
        ------------------------------------------------------------- */
        [data-testid="stAppViewContainer"],
        [data-testid="stAppViewBlockContainer"],
        [data-testid="stMain"] {
            background: var(--surface) !important;
            color: var(--text-primary) !important;
        }

        [data-testid="stBottomBlockContainer"],
        [data-testid="stBottom"],
        .stChatFloatingInputContainer {
            background: var(--surface) !important;
            border-top: 1px solid var(--border);
        }

        section[data-testid="stSidebar"],
        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] span,
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] small,
        section[data-testid="stSidebar"] div {
            color: var(--text-primary);
        }

        section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
        section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] small,
        section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] span {
            color: var(--text-secondary) !important;
        }

        [data-testid="stExpander"] summary,
        [data-testid="stExpander"] summary p,
        [data-testid="stExpander"] summary span {
            color: var(--text-primary) !important;
            font-weight: 600;
        }

        .stApp {
            background:
                radial-gradient(
                    circle at 75% 0%,
                    rgba(109, 40, 217, 0.055),
                    transparent 26rem
                ),
                var(--surface);
        }

        header[data-testid="stHeader"] {
            background: rgba(255, 255, 255, 0.82);
            backdrop-filter: blur(14px);
            border-bottom: 1px solid rgba(228, 228, 231, 0.75);
        }

        .block-container {
            max-width: 1260px;
            padding-top: 2rem;
            padding-bottom: 2rem;
        }

        footer {
            visibility: hidden;
        }

        /* -------------------------------------------------------------
           Professional application header
        ------------------------------------------------------------- */
        .app-header-shell {
            position: relative;
            overflow: hidden;
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1.5rem;
            padding: 1.45rem 1.6rem;
            margin-bottom: 1.25rem;
            border: 1px solid #E9E2F5;
            border-radius: 20px;
            background:
                linear-gradient(
                    135deg,
                    rgba(255, 255, 255, 0.98),
                    rgba(248, 245, 255, 0.96)
                );
            box-shadow: var(--shadow-md);
        }

        .app-header-shell::after {
            content: "";
            position: absolute;
            width: 180px;
            height: 180px;
            top: -90px;
            right: -40px;
            border-radius: 999px;
            background: rgba(109, 40, 217, 0.07);
            pointer-events: none;
        }

        .app-header-content {
            position: relative;
            z-index: 1;
            display: flex;
            align-items: center;
            gap: 1rem;
        }

        .app-logo {
            width: 52px;
            height: 52px;
            flex-shrink: 0;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: 16px;
            color: #FFFFFF;
            font-size: 1.45rem;
            background: linear-gradient(145deg, #7C3AED, #5B21B6);
            box-shadow: var(--shadow-primary);
        }

        .app-heading-group {
            min-width: 0;
        }

        .app-eyebrow {
            margin-bottom: 0.25rem;
            color: var(--primary);
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.1em;
            text-transform: uppercase;
        }

        .app-title {
            margin: 0;
            color: var(--text-primary);
            font-size: clamp(1.3rem, 2.4vw, 1.8rem);
            font-weight: 750;
            letter-spacing: -0.035em;
            line-height: 1.2;
        }

        .app-subtitle {
            max-width: 770px;
            margin: 0.45rem 0 0;
            color: var(--text-secondary);
            font-size: 0.92rem;
            line-height: 1.55;
        }

        .secure-indicator {
            position: relative;
            z-index: 1;
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            flex-shrink: 0;
            padding: 0.5rem 0.8rem;
            border: 1px solid #D9F0DF;
            border-radius: 999px;
            color: var(--success);
            background: var(--success-soft);
            font-size: 0.78rem;
            font-weight: 650;
            white-space: nowrap;
        }

        .secure-dot {
            width: 7px;
            height: 7px;
            border-radius: 999px;
            background: #22C55E;
            box-shadow: 0 0 0 4px rgba(34, 197, 94, 0.13);
            animation: pulse-dot 2.2s ease-in-out infinite;
        }

        @keyframes pulse-dot {
            0%, 100% {
                box-shadow: 0 0 0 4px rgba(34, 197, 94, 0.13);
            }
            50% {
                box-shadow: 0 0 0 7px rgba(34, 197, 94, 0.2);
            }
        }

        /* -------------------------------------------------------------
           Token dashboard
        ------------------------------------------------------------- */
        .token-section-title {
            margin: 0 0 0.6rem;
            color: var(--text-primary);
            font-size: 0.88rem;
            font-weight: 700;
        }

        [data-testid="stMetric"] {
            padding: 0.8rem;
            border: 1px solid var(--border);
            border-radius: 12px;
            background: #FFFFFF;
            box-shadow: var(--shadow-sm);
            transition: transform 150ms ease, box-shadow 150ms ease;
        }

        [data-testid="stMetric"]:hover {
            transform: translateY(-1px);
            box-shadow: var(--shadow-md);
        }

        [data-testid="stMetricLabel"] {
            color: var(--text-muted);
        }

        [data-testid="stMetricValue"] {
            color: var(--text-primary);
            font-size: 1.3rem;
        }

        /* -------------------------------------------------------------
           Sidebar
        ------------------------------------------------------------- */
        section[data-testid="stSidebar"] {
            width: 340px !important;
            background:
                linear-gradient(
                    180deg,
                    #FCFBFF 0%,
                    #F8F6FC 100%
                );
            border-right: 1px solid #E9E4EF;
        }

        section[data-testid="stSidebar"] > div {
            padding-top: 1.35rem;
        }

        .sidebar-brand {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            padding: 0.15rem 0 1rem;
            border-bottom: 1px solid var(--border);
            margin-bottom: 1rem;
        }

        .sidebar-brand-icon {
            width: 40px;
            height: 40px;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: 12px;
            color: #FFFFFF;
            background: linear-gradient(145deg, #7C3AED, #5B21B6);
            box-shadow: 0 7px 16px rgba(109, 40, 217, 0.18);
        }

        .sidebar-brand-title {
            color: var(--text-primary);
            font-size: 0.95rem;
            font-weight: 700;
            line-height: 1.2;
        }

        .sidebar-brand-caption {
            margin-top: 0.15rem;
            color: var(--text-muted);
            font-size: 0.73rem;
        }

        .sidebar-section-label {
            margin: 0.4rem 0 0.35rem;
            color: var(--text-primary);
            font-size: 0.84rem;
            font-weight: 700;
        }

        .sidebar-help {
            margin-bottom: 0.8rem;
            color: var(--text-secondary);
            font-size: 0.77rem;
            line-height: 1.55;
        }

        [data-testid="stFileUploaderDropzone"] {
            min-height: 148px;
            padding: 1.1rem;
            border: 1.5px dashed #CFC2DF;
            border-radius: 14px;
            background: rgba(255, 255, 255, 0.75);
            transition: border-color 150ms ease, background 150ms ease;
        }

        [data-testid="stFileUploaderDropzone"]:hover {
            border-color: var(--primary);
            background: var(--primary-softer);
        }

        [data-testid="stFileUploaderDropzone"] button {
            border: 1px solid #D6CCDF;
            border-radius: 9px;
            background: #FFFFFF;
            color: var(--text-primary);
        }

        [data-testid="stFileUploaderDropzone"] svg {
            fill: var(--primary) !important;
        }

        [data-testid="stFileUploaderFile"] {
            border: 1px solid var(--border);
            border-radius: 10px;
            background: #FFFFFF;
        }

        /* -------------------------------------------------------------
           Buttons
        ------------------------------------------------------------- */
        div.stButton > button,
        div.stDownloadButton > button {
            min-height: 2.6rem;
            border: 1px solid #DDD6E5;
            border-radius: 10px;
            color: var(--text-primary);
            background: #FFFFFF;
            font-weight: 600;
            box-shadow: var(--shadow-sm);
            transition:
                transform 150ms ease,
                border-color 150ms ease,
                background 150ms ease,
                box-shadow 150ms ease;
        }

        div.stButton > button:hover,
        div.stDownloadButton > button:hover {
            transform: translateY(-1px);
            border-color: #BCA9D2;
            color: var(--primary);
            background: #FCFAFF;
            box-shadow: 0 5px 14px rgba(24, 24, 27, 0.07);
        }

        div.stButton > button:active,
        div.stDownloadButton > button:active {
            transform: translateY(0);
        }

        div.stButton > button[kind="primary"] {
            border-color: var(--primary);
            color: #FFFFFF;
            background: linear-gradient(135deg, #7C3AED, #6525C6);
            box-shadow: var(--shadow-primary);
        }

        div.stButton > button[kind="primary"]:hover {
            border-color: var(--primary-dark);
            color: #FFFFFF;
            background: linear-gradient(135deg, #6D28D9, #5521B5);
        }

        /* -------------------------------------------------------------
           Document status
        ------------------------------------------------------------- */
        .document-status {
            display: flex;
            align-items: flex-start;
            gap: 0.7rem;
            padding: 0.8rem;
            border: 1px solid;
            border-radius: 12px;
            margin: 0.75rem 0 0.9rem;
            transition: transform 150ms ease, box-shadow 150ms ease;
        }

        .document-status:hover {
            transform: translateY(-1px);
            box-shadow: var(--shadow-sm);
        }

        .document-status.ready {
            border-color: #CDEBD5;
            background: var(--success-soft);
        }

        .document-status.empty {
            border-color: #F3DFB0;
            background: var(--warning-soft);
        }

        .document-status-icon {
            font-size: 1rem;
            line-height: 1.3;
        }

        .document-status-title {
            color: var(--text-primary);
            font-size: 0.79rem;
            font-weight: 700;
        }

        .document-status-description {
            margin-top: 0.1rem;
            color: var(--text-secondary);
            font-size: 0.72rem;
            line-height: 1.4;
        }

        /* -------------------------------------------------------------
           Empty state
        ------------------------------------------------------------- */
        .empty-state {
            position: relative;
            overflow: hidden;
            max-width: 780px;
            margin: 3rem auto;
            padding: 2.2rem 2rem;
            border: 1px solid #E8E0F2;
            border-radius: 22px;
            text-align: center;
            background:
                linear-gradient(
                    145deg,
                    rgba(255, 255, 255, 0.98),
                    rgba(249, 246, 255, 0.97)
                );
            box-shadow: 0 14px 40px rgba(58, 35, 79, 0.07);
            animation: fadeInUp 260ms ease;
        }

        .empty-state-icon {
            width: 66px;
            height: 66px;
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 0 auto 1.15rem;
            border: 1px solid #E5D9F5;
            border-radius: 20px;
            font-size: 1.7rem;
            background: #FFFFFF;
            box-shadow: 0 10px 25px rgba(109, 40, 217, 0.12);
        }

        .empty-state h2 {
            margin: 0;
            color: var(--text-primary);
            font-size: 1.45rem;
            font-weight: 750;
        }

        .empty-state p {
            max-width: 590px;
            margin: 0.7rem auto 0;
            color: var(--text-secondary);
            font-size: 0.91rem;
            line-height: 1.65;
        }

        .empty-state-hints {
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            justify-content: center;
            gap: 0.5rem;
            margin-top: 1.2rem;
        }

        .empty-state-hint {
            padding: 0.4rem 0.8rem;
            border: 1px solid #E5D9F5;
            border-radius: 999px;
            color: var(--primary);
            background: var(--primary-soft);
            font-size: 0.76rem;
            font-weight: 600;
        }

        /* -------------------------------------------------------------
           Chat messages and input
        ------------------------------------------------------------- */
        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translateY(8px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        [data-testid="stChatMessage"] {
            max-width: 950px;
            padding: 1.05rem 1.15rem;
            margin: 0 auto 0.85rem;
            border: 1px solid var(--border-soft);
            border-radius: 16px;
            background: #FFFFFF;
            box-shadow: 0 3px 12px rgba(24, 24, 27, 0.035);
            animation: fadeInUp 200ms ease;
        }

        [data-testid="stChatMessage"] p {
            color: var(--text-primary);
            line-height: 1.7;
        }

        [data-testid="stChatInput"] {
            max-width: 950px;
            margin-left: auto;
            margin-right: auto;
        }

        [data-testid="stChatInput"] textarea {
            color: var(--text-primary) !important;
        }

        [data-testid="stChatInput"] > div {
            border: 1px solid #D9D1E2;
            border-radius: 15px;
            background: rgba(255, 255, 255, 0.97);
            box-shadow: 0 8px 25px rgba(24, 24, 27, 0.08);
            transition: box-shadow 150ms ease, border-color 150ms ease;
        }

        [data-testid="stChatInput"] > div:focus-within {
            border-color: var(--primary);
            box-shadow:
                0 0 0 3px rgba(109, 40, 217, 0.1),
                0 8px 25px rgba(24, 24, 27, 0.08);
        }

        /* -------------------------------------------------------------
           Citation cards
        ------------------------------------------------------------- */
        .citation-card {
            position: relative;
            overflow: hidden;
            padding: 0.85rem 1rem;
            margin-bottom: 0.65rem;
            border: 1px solid #E3D9F0;
            border-radius: 12px;
            background: linear-gradient(135deg, #FBF9FE, #FFFFFF);
            transition: transform 150ms ease, box-shadow 150ms ease;
        }

        .citation-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 18px rgba(109, 40, 217, 0.12);
        }

        .citation-card::before {
            content: "";
            position: absolute;
            top: 0;
            bottom: 0;
            left: 0;
            width: 4px;
            background: linear-gradient(180deg, #7C3AED, #A78BFA);
        }

        .citation-card .cite-meta {
            padding-left: 0.2rem;
            color: #4C1D75;
            font-size: 0.79rem;
            font-weight: 700;
        }

        .citation-card .cite-preview {
            padding-left: 0.2rem;
            margin-top: 0.35rem;
            color: var(--text-secondary);
            font-size: 0.78rem;
            line-height: 1.55;
        }

        [data-testid="stExpander"] {
            overflow: hidden;
            border: 1px solid var(--border);
            border-radius: 12px;
            background: #FFFFFF;
        }

        [data-testid="stAlert"] {
            border-radius: 12px;
        }

        /* -------------------------------------------------------------
           Footer
        ------------------------------------------------------------- */
        .app-footer {
            max-width: 950px;
            margin: 2.5rem auto 0;
            padding: 1rem 1.1rem;
            border: 1px solid var(--border);
            border-radius: 12px;
            color: var(--text-muted);
            background: var(--surface-soft);
            font-size: 0.74rem;
            line-height: 1.55;
        }

        .app-footer strong {
            color: var(--text-secondary);
        }

        @media (max-width: 900px) {
            .block-container {
                padding-top: 1.2rem;
                padding-left: 1rem;
                padding-right: 1rem;
            }

            .app-header-shell {
                align-items: flex-start;
                flex-direction: column;
                padding: 1.2rem;
            }
        }

        @media (max-width: 600px) {
            .app-header-content {
                align-items: flex-start;
            }

            .app-logo {
                width: 44px;
                height: 44px;
                border-radius: 13px;
            }

            .app-title {
                font-size: 1.2rem;
            }

            .app-subtitle {
                font-size: 0.82rem;
            }

            .empty-state {
                margin-top: 1.5rem;
                padding: 1.65rem 1rem;
            }

            [data-testid="stChatMessage"] {
                padding: 0.85rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ------------------------------------------------------------------------
# Error handling
# ------------------------------------------------------------------------
def _friendly_error(
    user_message: str,
    exc: Exception,
) -> None:
    logger.exception(user_message)

    st.error(f"⚠️ {user_message}")

    with st.expander("Technical details for support"):
        st.code(f"{type(exc).__name__}: {exc}")


def _render_startup_failure(exc: Exception) -> None:
    st.markdown("## ⚠️ The assistant could not start")

    st.write(
        "There is a configuration issue that must be fixed before the "
        "application can run. This is a setup issue, not a problem with "
        "your question."
    )

    st.error(str(exc))

    with st.expander("Full technical details"):
        st.code(traceback.format_exc())

    st.stop()


# ------------------------------------------------------------------------
# Token usage helpers
# ------------------------------------------------------------------------
def _default_token_usage() -> Dict[str, Any]:
    return {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "thinking_tokens": 0,
        "cached_tokens": 0,
        "model": "",
    }


def _safe_token_value(
    usage: Optional[Dict[str, Any]],
    key: str,
) -> int:
    if not usage:
        return 0

    try:
        return int(usage.get(key, 0) or 0)
    except (TypeError, ValueError):
        return 0


def _reset_token_usage() -> None:
    st.session_state.last_token_usage = _default_token_usage()

    st.session_state.session_input_tokens = 0
    st.session_state.session_output_tokens = 0
    st.session_state.session_total_tokens = 0
    st.session_state.session_thinking_tokens = 0
    st.session_state.session_cached_tokens = 0


def _update_token_usage(
    usage_metadata: Optional[Dict[str, Any]],
) -> None:
    usage = usage_metadata or {}

    input_tokens = _safe_token_value(
        usage,
        "input_tokens",
    )

    output_tokens = _safe_token_value(
        usage,
        "output_tokens",
    )

    total_tokens = _safe_token_value(
        usage,
        "total_tokens",
    )

    thinking_tokens = _safe_token_value(
        usage,
        "thinking_tokens",
    )

    cached_tokens = _safe_token_value(
        usage,
        "cached_tokens",
    )

    if total_tokens == 0:
        total_tokens = input_tokens + output_tokens

    st.session_state.last_token_usage = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "thinking_tokens": thinking_tokens,
        "cached_tokens": cached_tokens,
        "model": usage.get("model", ""),
    }

    st.session_state.session_input_tokens += input_tokens
    st.session_state.session_output_tokens += output_tokens
    st.session_state.session_total_tokens += total_tokens
    st.session_state.session_thinking_tokens += thinking_tokens
    st.session_state.session_cached_tokens += cached_tokens


# ------------------------------------------------------------------------
# Session state
# ------------------------------------------------------------------------
def _init_session_state() -> None:
    if "rag_pipeline" not in st.session_state:
        with st.spinner(
            "Starting up the Aetna Virtual Assistant..."
        ):
            try:
                st.session_state.rag_pipeline = RAGPipeline()
            except Exception as exc:
                _render_startup_failure(exc)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "documents_ingested" not in st.session_state:
        try:
            st.session_state.documents_ingested = (
                st.session_state.rag_pipeline.retriever.has_documents()
            )
        except Exception:
            st.session_state.documents_ingested = False

    if "pending_question" not in st.session_state:
        st.session_state.pending_question = None

    if "last_token_usage" not in st.session_state:
        st.session_state.last_token_usage = _default_token_usage()

    if "session_input_tokens" not in st.session_state:
        st.session_state.session_input_tokens = 0

    if "session_output_tokens" not in st.session_state:
        st.session_state.session_output_tokens = 0

    if "session_total_tokens" not in st.session_state:
        st.session_state.session_total_tokens = 0

    if "session_thinking_tokens" not in st.session_state:
        st.session_state.session_thinking_tokens = 0

    if "session_cached_tokens" not in st.session_state:
        st.session_state.session_cached_tokens = 0


# ------------------------------------------------------------------------
# Application header
# ------------------------------------------------------------------------
def _render_header() -> None:
    header_html = """
<div class="app-header-shell">
    <div class="app-header-content">
        <div class="app-logo">✦</div>

        <div class="app-heading-group">
            <div class="app-eyebrow">
                Aetna Document Intelligence
            </div>

            <h1 class="app-title">
                Insurance Virtual Assistant
            </h1>

            <p class="app-subtitle">
                Ask questions about your uploaded insurance documents
                and receive clear answers supported by relevant sources.
            </p>
        </div>
    </div>

    <div class="secure-indicator">
        <span class="secure-dot"></span>
        Secure document workspace
    </div>
</div>
"""

    st.html(header_html)



# ------------------------------------------------------------------------
# Token usage interface
# ------------------------------------------------------------------------
def _render_token_usage() -> None:
    usage = st.session_state.last_token_usage

    input_tokens = _safe_token_value(
        usage,
        "input_tokens",
    )

    output_tokens = _safe_token_value(
        usage,
        "output_tokens",
    )

    total_tokens = _safe_token_value(
        usage,
        "total_tokens",
    )

    thinking_tokens = _safe_token_value(
        usage,
        "thinking_tokens",
    )

    cached_tokens = _safe_token_value(
        usage,
        "cached_tokens",
    )

    session_total = int(
        st.session_state.session_total_tokens
    )

    model_name = usage.get("model", "")

    if not model_name:
        model_name = getattr(
            st.session_state.rag_pipeline,
            "_chat_model_label",
            "Not available",
        )

    with st.expander(
        "Token usage and model information",
        expanded=False,
    ):
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                label="Last input",
                value=f"{input_tokens:,}",
                help=(
                    "Tokens used by the question, system prompt, "
                    "and retrieved document context."
                ),
            )

        with col2:
            st.metric(
                label="Last output",
                value=f"{output_tokens:,}",
                help="Tokens generated in the latest answer.",
            )

        with col3:
            st.metric(
                label="Last request",
                value=f"{total_tokens:,}",
                help=(
                    "Total tokens reported for the latest request."
                ),
            )

        with col4:
            st.metric(
                label="Session total",
                value=f"{session_total:,}",
                help=(
                    "Cumulative token usage in the current chat."
                ),
            )

        details = [
            f"Model: {model_name}",
            f"Thinking tokens: {thinking_tokens:,}",
            f"Cached tokens: {cached_tokens:,}",
        ]

        st.caption(" | ".join(details))

        st.info(
            "Session total represents token usage during the current chat. "
            "It does not represent the remaining API account quota."
        )


# ------------------------------------------------------------------------
# PDF ingestion
# ------------------------------------------------------------------------
def _handle_ingestion(uploaded_files) -> None:
    if not uploaded_files:
        st.warning("Please select at least one PDF document.")
        return

    os.makedirs(
        settings.pdf_upload_dir,
        exist_ok=True,
    )

    saved_paths = []

    for uploaded_file in uploaded_files:
        safe_filename = os.path.basename(
            uploaded_file.name
        )

        save_path = os.path.join(
            settings.pdf_upload_dir,
            safe_filename,
        )

        try:
            with open(save_path, "wb") as file:
                file.write(uploaded_file.getbuffer())

            saved_paths.append(save_path)

        except OSError as exc:
            _friendly_error(
                f"Could not save {safe_filename}.",
                exc,
            )
            return

    with st.spinner(
        f"Processing {len(saved_paths)} document(s)..."
    ):
        try:
            chunk_count = (
                st.session_state.rag_pipeline.ingest_documents(
                    saved_paths
                )
            )

            st.session_state.documents_ingested = (
                chunk_count > 0
            )

            if chunk_count > 0:
                st.success(
                    f"✅ Indexed {chunk_count:,} chunks from "
                    f"{len(saved_paths)} document(s)."
                )
            else:
                st.warning(
                    "No readable text was found in the uploaded "
                    "documents."
                )

        except Exception as exc:
            _friendly_error(
                "Something went wrong while processing your documents.",
                exc,
            )

# ------------------------------------------------------------------------
# Sidebar token usage
# ------------------------------------------------------------------------
def _render_sidebar_token_usage() -> None:
    usage = st.session_state.get(
        "last_token_usage",
        {},
    )

    input_tokens = _safe_token_value(
        usage,
        "input_tokens",
    )

    output_tokens = _safe_token_value(
        usage,
        "output_tokens",
    )

    total_tokens = _safe_token_value(
        usage,
        "total_tokens",
    )

    thinking_tokens = _safe_token_value(
        usage,
        "thinking_tokens",
    )

    cached_tokens = _safe_token_value(
        usage,
        "cached_tokens",
    )

    session_total = int(
        st.session_state.get(
            "session_total_tokens",
            0,
        )
    )

    model_name = usage.get("model", "")

    if not model_name:
        model_name = getattr(
            st.session_state.rag_pipeline,
            "_chat_model_label",
            "Not available",
        )

    with st.expander(
        "Token usage",
        expanded=False,
    ):
        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                label="Last input",
                value=f"{input_tokens:,}",
                help=(
                    "Tokens used by the question, system prompt, "
                    "and retrieved document context."
                ),
            )

        with col2:
            st.metric(
                label="Last output",
                value=f"{output_tokens:,}",
                help="Tokens generated in the latest answer.",
            )

        col3, col4 = st.columns(2)

        with col3:
            st.metric(
                label="Last request",
                value=f"{total_tokens:,}",
                help="Total tokens used by the latest request.",
            )

        with col4:
            st.metric(
                label="Session total",
                value=f"{session_total:,}",
                help=(
                    "Total token usage during the current chat."
                ),
            )

        st.caption(
            f"**Model:** {model_name}"
        )

        if thinking_tokens > 0:
            st.caption(
                f"**Thinking tokens:** {thinking_tokens:,}"
            )

        if cached_tokens > 0:
            st.caption(
                f"**Cached tokens:** {cached_tokens:,}"
            )

        st.caption(
            "Session total represents token usage during the "
            "current chat. It does not represent the remaining "
            "API account quota."
        )

# ------------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------------
def _render_sidebar() -> None:
    with st.sidebar:
        # Sidebar heading
        st.markdown(
            """
            <div class="sidebar-brand">
                <div class="sidebar-brand-icon">✦</div>
                <div>
                    <div class="sidebar-brand-title">
                        Document Workspace
                    </div>
                    <div class="sidebar-brand-caption">
                        Secure plan document assistant
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Upload section
        st.markdown(
            """
            <div class="sidebar-section-label">
                Upload documents
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="sidebar-help">
                Add your Evidence of Coverage, Summary of Benefits and
                Coverage, or other insurance PDF documents.
            </div>
            """,
            unsafe_allow_html=True,
        )

        uploaded_files = st.file_uploader(
            label="Choose PDF file(s)",
            type=["pdf"],
            accept_multiple_files=True,
            label_visibility="collapsed",
            help="You can select and upload multiple PDF documents.",
            key="sidebar_pdf_file_uploader",
        )

        if uploaded_files:
            if st.button(
                label="Process documents",
                type="primary",
                use_container_width=True,
                key="process_documents_button",
            ):
                _handle_ingestion(uploaded_files)

        # Read the number of indexed document chunks
        try:
            doc_count = (
                st.session_state.rag_pipeline
                .chroma_manager
                .document_count()
            )

        except Exception as exc:
            logger.exception(
                "Could not read the document count"
            )

            doc_count = 0

            with st.expander("Document status details"):
                st.code(
                    f"{type(exc).__name__}: {exc}"
                )

        # Document status
        if doc_count > 0:
            st.markdown(
                f"""
                <div class="document-status ready">
                    <div class="document-status-icon">✓</div>
                    <div>
                        <div class="document-status-title">
                            Knowledge base ready
                        </div>
                        <div class="document-status-description">
                            {doc_count:,} document chunks are available
                            for retrieval.
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        else:
            st.markdown(
                """
                <div class="document-status empty">
                    <div class="document-status-icon">○</div>
                    <div>
                        <div class="document-status-title">
                            No documents processed
                        </div>
                        <div class="document-status-description">
                            Upload and process at least one PDF to
                            begin asking questions.
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.divider()

        # Workspace actions
        st.markdown(
            """
            <div class="sidebar-section-label">
                Workspace actions
            </div>
            """,
            unsafe_allow_html=True,
        )

        col1, col2 = st.columns(2)

        with col1:
            if st.button(
                label="New chat",
                use_container_width=True,
                help=(
                    "Start a new conversation without deleting documents."
                ),
                key="new_chat_button",
            ):
                st.session_state.chat_history = []
                st.session_state.pending_question = None

                _reset_token_usage()

                st.rerun()

        with col2:
            if st.button(
                label="Clear docs",
                use_container_width=True,
                help="Delete all indexed document content.",
                key="clear_documents_button",
            ):
                try:
                    (
                        st.session_state.rag_pipeline
                        .chroma_manager
                        .clear_collection()
                    )

                    st.session_state.documents_ingested = False
                    st.session_state.chat_history = []
                    st.session_state.pending_question = None

                    _reset_token_usage()

                    st.rerun()

                except Exception as exc:
                    _friendly_error(
                        "Could not clear the document collection.",
                        exc,
                    )

        # Export conversation
        if st.session_state.chat_history:
            transcript_lines = []

            for message in st.session_state.chat_history:
                role = message.get(
                    "role",
                    "assistant",
                )

                role_label = (
                    "You"
                    if role == "user"
                    else "Assistant"
                )

                content = str(
                    message.get(
                        "content",
                        "",
                    )
                )

                transcript_lines.append(
                    f"{role_label}: {content}"
                )

            transcript = "\n\n".join(
                transcript_lines
            )

            st.download_button(
                label="Export conversation",
                data=transcript,
                file_name=(
                    "aetna_chat_"
                    f"{datetime.now().strftime('%Y%m%d_%H%M')}.txt"
                ),
                mime="text/plain",
                use_container_width=True,
                key="export_conversation_button",
            )

        st.write("")

        # System details
        with st.expander("System information"):
            st.caption(
                f"**Chat provider:** "
                f"{settings.resolved_chat_provider}  \n"
                f"**Embedding provider:** "
                f"{settings.embedding_provider}  \n"
                f"**Retrieval depth:** "
                f"Top {settings.retrieval_top_k} sources"
            )

            st.divider()

        st.markdown(
            """
            <div class="sidebar-section-label">
                Usage information
            </div>
            """,
            unsafe_allow_html=True,
        )

        _render_sidebar_token_usage()


# ------------------------------------------------------------------------
# Empty state
# ------------------------------------------------------------------------
def _render_empty_state() -> None:
    render_html = """
<div class="empty-state">
    <div class="empty-state-icon">💬</div>

    <h2>Ask questions about your documents</h2>

    <p>
        Upload and process one or more PDF documents from the
        sidebar. You can then ask one question or submit multiple
        questions by entering one question per line.
    </p>

    <div class="empty-state-hints">
        <span class="empty-state-hint">"What is my deductible?"</span>
        <span class="empty-state-hint">"Is physical therapy covered?"</span>
        <span class="empty-state-hint">"What's my out-of-pocket max?"</span>
    </div>
</div>
"""
    st.html(render_html)


# ------------------------------------------------------------------------
# Citation rendering
# ------------------------------------------------------------------------
def _render_citations(sources) -> None:
    if not sources:
        return

    with st.expander(
        f"Supporting sources ({len(sources)})"
    ):
        for index, chunk in enumerate(
            sources,
            start=1,
        ):
            source_file = html.escape(
                str(
                    getattr(
                        chunk,
                        "source_file",
                        "Unknown document",
                    )
                )
            )

            page_number = html.escape(
                str(
                    getattr(
                        chunk,
                        "page_number",
                        "Unknown",
                    )
                )
            )

            chunk_text = str(
                getattr(
                    chunk,
                    "text",
                    "",
                )
            )

            raw_preview = chunk_text[:400]

            if len(chunk_text) > 400:
                raw_preview += "..."

            preview = html.escape(raw_preview)

            st.markdown(
                f"""
                <div class="citation-card">
                    <div class="cite-meta">
                        Source {index} · {source_file} ·
                        Page {page_number}
                    </div>

                    <div class="cite-preview">
                        {preview}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ------------------------------------------------------------------------
# Per-message token usage
# ------------------------------------------------------------------------
def _render_message_token_usage(
    usage_metadata: Optional[Dict[str, Any]],
) -> None:
    usage = usage_metadata or {}

    input_tokens = _safe_token_value(
        usage,
        "input_tokens",
    )

    output_tokens = _safe_token_value(
        usage,
        "output_tokens",
    )

    total_tokens = _safe_token_value(
        usage,
        "total_tokens",
    )

    thinking_tokens = _safe_token_value(
        usage,
        "thinking_tokens",
    )

    cached_tokens = _safe_token_value(
        usage,
        "cached_tokens",
    )

    if total_tokens == 0:
        total_tokens = input_tokens + output_tokens

    if total_tokens <= 0:
        return

    token_parts = [
        f"{input_tokens:,} input",
        f"{output_tokens:,} output",
        f"{total_tokens:,} total",
    ]

    if thinking_tokens > 0:
        token_parts.append(
            f"{thinking_tokens:,} thinking"
        )

    if cached_tokens > 0:
        token_parts.append(
            f"{cached_tokens:,} cached"
        )

    st.caption(
        "Token usage: " + " · ".join(token_parts)
    )


# ------------------------------------------------------------------------
# Chat history
# ------------------------------------------------------------------------
def _render_chat_history() -> None:
    for message in st.session_state.chat_history:
        role = message.get(
            "role",
            "assistant",
        )

        avatar = (
            "🤖"
            if role == "assistant"
            else None
        )

        with st.chat_message(
            role,
            avatar=avatar,
        ):
            st.markdown(
                str(
                    message.get(
                        "content",
                        "",
                    )
                )
            )

            if (
                role == "assistant"
                and message.get("sources")
            ):
                _render_citations(
                    message["sources"]
                )

            if role == "assistant":
                _render_message_token_usage(
                    message.get(
                        "usage_metadata",
                        {},
                    )
                )


# ------------------------------------------------------------------------
# Question processing
# ------------------------------------------------------------------------
def _answer_question(question: str) -> None:
    cleaned_question = question.strip()

    if not cleaned_question:
        return

    # Add user message to history
    st.session_state.chat_history.append(
        {
            "role": "user",
            "content": cleaned_question,
        }
    )

    # Show user message immediately
    with st.chat_message("user"):
        st.markdown(cleaned_question)

    # Show assistant processing UI
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Searching your documents and drafting an answer..."):
            try:
                response: RAGResponse = (
                    st.session_state.rag_pipeline.ask(
                        cleaned_question
                    )
                )

            except Exception as exc:
                logger.exception(
                    "RAG pipeline failed to answer"
                )

                response = RAGResponse(
                    answer=(
                        "⚠️ Sorry, something went wrong while generating "
                        "an answer. Please try again in a moment."
                    ),
                    sources=[],
                    usage_metadata={},
                )

                with st.expander("Technical details (for support)"):
                    st.code(f"{type(exc).__name__}: {exc}")

                st.session_state.chat_history.append(
                    {
                        "role": "assistant",
                        "content": response.answer,
                        "sources": [],
                        "usage_metadata": {},
                        "error_details": (
                            f"{type(exc).__name__}: {exc}"
                        ),
                    }
                )

                st.markdown(response.answer)
                return

        usage_metadata = response.usage_metadata or {}

        _update_token_usage(usage_metadata)

        # Display answer immediately
        st.markdown(response.answer)

        # Display citations if available
        if response.sources:
            _render_citations(response.sources)

    # Save assistant message in history
    st.session_state.chat_history.append(
        {
            "role": "assistant",
            "content": response.answer,
            "sources": response.sources,
            "usage_metadata": usage_metadata,
        }
    )

# ------------------------------------------------------------------------
# Chat input
# ------------------------------------------------------------------------
def _render_chat_input() -> None:
    pending_question = (
        st.session_state.pending_question
    )

    if pending_question:
        st.session_state.pending_question = None
        _answer_question(pending_question)
        st.rerun()

    user_input = st.chat_input(
        "Ask one or multiple questions, one question per line"
    )

    if not user_input:
        return

    questions = [
        question.strip()
        for question in user_input.splitlines()
        if question.strip()
    ]

    if not questions:
        return

    with st.spinner(
        f"Analyzing your {len(questions)} question(s)"
    ):
        for question in questions:
            _answer_question(question)

    st.rerun()


# ------------------------------------------------------------------------
# Footer
# ------------------------------------------------------------------------
def _render_footer() -> None:
    safe_disclaimer = html.escape(
        DISCLAIMER
    )

    st.markdown(
        f"""
        <div class="app-footer">
            <strong>Important information:</strong>
            {safe_disclaimer}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ------------------------------------------------------------------------
# Main application
# ------------------------------------------------------------------------
def run_app() -> None:
    st.set_page_config(
        page_title="Aetna Insurance Virtual Assistant",
        page_icon="💜",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    _inject_custom_css()
    _init_session_state()
    _render_sidebar()
    _render_header()

    if not st.session_state.chat_history:
        _render_empty_state()
    else:
        _render_chat_history()

    _render_chat_input()
    _render_footer()


if __name__ == "__main__":
    run_app()
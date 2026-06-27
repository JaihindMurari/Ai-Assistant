"""
utils/helpers.py
-----------------
Shared utility functions used across multiple Streamlit pages: session-state
initialization, file persistence, export helpers, and small formatting
utilities. Keeping these here avoids duplicating logic between pages/upload.py,
pages/chat.py, and pages/settings.py.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st

logger = logging.getLogger(__name__)


def init_session_state() -> None:
    """Initialize all Streamlit session_state keys used across the app, once."""
    defaults: Dict[str, Any] = {
        "chat_history": [],          # list of {"role", "content", "sources", "confidence", "timestamp"}
        "processed_sources": [],     # list of filenames/labels already embedded
        "pending_files": [],         # uploaded files awaiting processing
        "pasted_text_buffer": "",
        "top_k": 4,
        "temperature": 0.2,
        "max_tokens": 1024,
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "total_estimated_cost": 0.0,
        "last_full_text": "",        # text of most recently processed document, for summary/quiz/etc.
        "last_doc_name": "",
        "compare_doc_a": None,
        "compare_doc_b": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def save_uploaded_file(uploaded_file, upload_dir: str) -> str:
    """Persist a Streamlit UploadedFile object to disk and return its path."""
    Path(upload_dir).mkdir(parents=True, exist_ok=True)
    dest_path = Path(upload_dir) / uploaded_file.name
    with open(dest_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    logger.info("Saved uploaded file to %s", dest_path)
    return str(dest_path)


def add_chat_message(role: str, content: str, sources: List[dict] | None = None, confidence: float = 0.0) -> None:
    """Append a message to the chat history in session state."""
    st.session_state.chat_history.append(
        {
            "role": role,
            "content": content,
            "sources": sources or [],
            "confidence": confidence,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
    )


def track_usage(input_tokens: int, output_tokens: int, cost: float) -> None:
    """Accumulate running token/cost totals in session state."""
    st.session_state.total_input_tokens += input_tokens
    st.session_state.total_output_tokens += output_tokens
    st.session_state.total_estimated_cost += cost


def export_chat_history_as_markdown() -> str:
    """Render the current chat history as a downloadable Markdown transcript."""
    lines = ["# AI Document Assistant - Chat Export", ""]
    for msg in st.session_state.chat_history:
        role_label = "**You**" if msg["role"] == "user" else "**Assistant**"
        lines.append(f"### {role_label}  \n_{msg['timestamp']}_")
        lines.append(msg["content"])
        if msg.get("sources"):
            lines.append("\n**Sources:**")
            for s in msg["sources"]:
                lines.append(f"- {s.get('source', 'Unknown')} (Page {s.get('page', '?')})")
        lines.append("\n---\n")
    return "\n".join(lines)


def export_chat_history_as_json() -> str:
    """Render chat history as a JSON string for export."""
    return json.dumps(st.session_state.chat_history, indent=2)


def format_confidence_label(score: float) -> str:
    """Map a numeric confidence score to a human-readable label."""
    if score >= 0.75:
        return "High"
    if score >= 0.5:
        return "Medium"
    return "Low"


def truncate(text: str, max_chars: int = 4000) -> str:
    """Truncate long text for prompts that don't need the full document."""
    return text if len(text) <= max_chars else text[:max_chars] + "..."

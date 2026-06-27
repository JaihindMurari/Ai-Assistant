"""
pages/settings.py
------------------
Settings page: provider/model status, retrieval & generation parameters,
usage/cost stats, vector database management, and application info.
"""

from __future__ import annotations

import streamlit as st

from config import settings
from utils.helpers import init_session_state

st.set_page_config(page_title="Settings - AI Document Assistant", page_icon="⚙️", layout="wide")
init_session_state()

st.title("⚙️ Settings")

if "pipeline" not in st.session_state or st.session_state.pipeline is None:
    st.warning("The RAG pipeline isn't initialized yet. Go back to the main page first.", icon="⚠️")
    st.stop()

pipeline = st.session_state.pipeline

# ---------------------------------------------------------------------------
# Provider status
# ---------------------------------------------------------------------------
st.subheader("🔌 LLM Provider")
col1, col2 = st.columns(2)
with col1:
    st.metric("Active Provider", settings.llm_provider.upper())
    st.metric("API Key Configured", "✅ Yes" if settings.has_valid_api_key() else "❌ No")
with col2:
    model_name = settings.openai_model if settings.llm_provider == "openai" else settings.gemini_model
    st.metric("Model", model_name)
    st.metric("Embedding Model", settings.embedding_model.split("/")[-1])

if not settings.has_valid_api_key():
    st.error(
        f"No valid API key found for provider '{settings.llm_provider}'. "
        "Set the appropriate key in your .env file (OPENAI_API_KEY or GOOGLE_API_KEY)."
    )

st.caption("To switch providers, edit `LLM_PROVIDER` in your `.env` file and restart the app.")

st.divider()

# ---------------------------------------------------------------------------
# Retrieval & generation parameters
# ---------------------------------------------------------------------------
st.subheader("🎛️ Retrieval & Generation Parameters")
col1, col2, col3 = st.columns(3)
with col1:
    st.session_state.top_k = st.slider("Top K (chunks retrieved)", 1, 10, st.session_state.top_k)
with col2:
    st.session_state.temperature = st.slider("Temperature", 0.0, 1.0, st.session_state.temperature, step=0.05)
with col3:
    st.session_state.max_tokens = st.slider("Max Tokens", 128, 4096, st.session_state.max_tokens, step=128)

st.caption("These settings apply immediately to new questions asked on the Chat page.")

st.divider()

# ---------------------------------------------------------------------------
# Usage & cost
# ---------------------------------------------------------------------------
st.subheader("📊 Usage Statistics")
col1, col2, col3 = st.columns(3)
col1.metric("Total Input Tokens", f"{st.session_state.total_input_tokens:,}")
col2.metric("Total Output Tokens", f"{st.session_state.total_output_tokens:,}")
col3.metric("Estimated API Cost", f"${st.session_state.total_estimated_cost:.4f}")
st.caption("Cost estimates use approximate public pricing and are for guidance only, not billing-accurate.")

st.divider()

# ---------------------------------------------------------------------------
# Vector database management
# ---------------------------------------------------------------------------
st.subheader("🗄️ Vector Database")
col1, col2 = st.columns(2)
col1.metric("Total Chunks Stored", pipeline.vector_store.count())
col2.metric("Documents Indexed", len(pipeline.vector_store.list_sources()))

confirm = st.checkbox("I understand this will permanently delete all embedded documents.")
if st.button("🗑️ Clear Entire Vector Database", type="primary", disabled=not confirm):
    pipeline.vector_store.clear()
    st.session_state.processed_sources = []
    st.session_state.last_full_text = ""
    st.session_state.last_doc_name = ""
    st.success("Vector database cleared.")
    st.rerun()

st.divider()

# ---------------------------------------------------------------------------
# Application info
# ---------------------------------------------------------------------------
st.subheader("ℹ️ Application Information")
st.markdown(
    f"""
- **App**: AI Document Assistant (RAG)
- **Vector DB**: ChromaDB, persisted at `{settings.vector_db_dir}`
- **Collection**: `{settings.collection_name}`
- **Chunk size / overlap**: {settings.chunk_size} / {settings.chunk_overlap}
- **Embedding model**: `{settings.embedding_model}`
"""
)

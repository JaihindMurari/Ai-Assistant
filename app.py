"""
app.py
------
Main entry point for the AI Document Assistant.

Responsibilities:
  - Bootstraps the RAG pipeline (embedder, vector store, retriever, LLM, chain)
    once and caches it via st.cache_resource.
  - Renders the sidebar (global controls available on every page).
  - Renders the home page: upload, paste-text, ask-a-question, and answer
    display, so the core workflow works end-to-end without navigating away.

Run with: streamlit run app.py
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import streamlit as st

from config import settings
from rag.chain import RAGChain
from rag.embeddings import SentenceTransformerEmbeddings
from rag.llm import LLMError, get_llm_provider
from rag.loader import DocumentLoadError, PDFLoader, load_pasted_text
from rag.retriever import Retriever
from rag.splitter import TextSplitter
from rag.vector_store import VectorStore
import streamlit as st
import google.generativeai as genai

genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
from utils.helpers import (
    add_chat_message,
    format_confidence_label,
    init_session_state,
    save_uploaded_file,
    track_usage,
    truncate,
)

logger = logging.getLogger(__name__)


@dataclass
class Pipeline:
    """Bundles every RAG component the UI layer needs to call into."""

    vector_store: VectorStore
    retriever: Retriever
    chain: RAGChain


@st.cache_resource(show_spinner="Initializing AI Document Assistant (loading embedding model)...")
def build_pipeline(_embedding_model: str, _vector_db_dir: str, _collection_name: str) -> Pipeline:
    """
    Build and cache the RAG pipeline. Cached by Streamlit's resource cache so
    the embedding model and Chroma client are only created once per session,
    not on every script rerun.
    """
    embedder = SentenceTransformerEmbeddings(model_name=_embedding_model)
    vector_store = VectorStore(
        persist_dir=_vector_db_dir,
        collection_name=_collection_name,
        embedder=embedder,
    )
    retriever = Retriever(vector_store, top_k=settings.default_top_k)

    llm = get_llm_provider(
        provider=settings.llm_provider,
        openai_api_key=settings.openai_api_key,
        openai_model=settings.openai_model,
        google_api_key=settings.google_api_key,
        gemini_model=settings.gemini_model,
    )
    chain = RAGChain(retriever=retriever, llm=llm)

    return Pipeline(vector_store=vector_store, retriever=retriever, chain=chain)


def main() -> None:
    st.set_page_config(
        page_title="AI Document Assistant",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    init_session_state()

    # ---- Custom CSS: dark-mode-safe, subtle visual polish ----
    st.markdown(
        """
        <style>
        .stChatMessage { border-radius: 12px; }
        .block-container { padding-top: 2rem; }
        div[data-testid="stMetricValue"] { font-size: 1.4rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # ---- Initialize pipeline (or show setup error) ----
    if "pipeline" not in st.session_state or st.session_state.pipeline is None:
        try:
            st.session_state.pipeline = build_pipeline(
                settings.embedding_model, settings.vector_db_dir, settings.collection_name
            )
        except Exception as exc:
            logger.exception("Failed to initialize pipeline")
            st.error(f"Failed to initialize the application: {exc}")
            st.stop()

    pipeline = st.session_state.pipeline

    render_sidebar(pipeline)
    render_home(pipeline)


def render_sidebar(pipeline: Pipeline) -> None:
    with st.sidebar:
        st.title("🧠 AI Document Assistant")
        st.caption("RAG-powered Q&A over your documents")

        if not settings.has_valid_api_key():
            st.error(f"⚠️ No API key set for provider '{settings.llm_provider}'. Add it to your .env file.")

        st.divider()

        with st.expander("📎 Quick Upload PDFs", expanded=False):
            quick_files = st.file_uploader(
                "Upload PDFs", type=["pdf"], accept_multiple_files=True, key="sidebar_uploader"
            )
            if quick_files and st.button("Process", key="sidebar_process_pdfs"):
                _process_pdfs(pipeline, quick_files)

        with st.expander("✍️ Quick Paste Text", expanded=False):
            quick_text = st.text_area("Paste text", height=120, key="sidebar_paste")
            if st.button("Process Text", key="sidebar_process_text") and quick_text.strip():
                _process_pasted_text(pipeline, quick_text)

        if st.button("🗑️ Clear Database", key="sidebar_clear_db"):
            pipeline.vector_store.clear()
            st.session_state.processed_sources = []
            st.toast("Vector database cleared.", icon="🗑️")
            st.rerun()

        st.divider()
        st.subheader("🎛️ Model Settings")
        st.caption(f"Embedding model: `{settings.embedding_model.split('/')[-1]}`")
        st.session_state.temperature = st.slider(
            "Temperature", 0.0, 1.0, st.session_state.temperature, step=0.05, key="sidebar_temp"
        )
        st.session_state.top_k = st.slider("Top K", 1, 10, st.session_state.top_k, key="sidebar_topk")
        st.session_state.max_tokens = st.slider(
            "Max Tokens", 128, 4096, st.session_state.max_tokens, step=128, key="sidebar_maxtok"
        )

        st.divider()
        st.subheader("💬 Chat History")
        st.caption(f"{len(st.session_state.chat_history)} message(s) this session")
        if st.button("Clear Chat History", key="sidebar_clear_chat"):
            st.session_state.chat_history = []
            st.rerun()

        st.divider()
        with st.expander("ℹ️ Application Information"):
            st.caption(f"Provider: {settings.llm_provider}")
            st.caption(f"Vector DB: ChromaDB @ `{settings.vector_db_dir}`")
            st.caption(f"Chunks indexed: {pipeline.vector_store.count()}")
            st.caption(f"Documents: {len(pipeline.vector_store.list_sources())}")

        st.divider()
        st.page_link("pages/upload.py", label="📄 Full Upload Page")
        st.page_link("pages/chat.py", label="💬 Full Chat Page")
        st.page_link("pages/settings.py", label="⚙️ Full Settings Page")


def render_home(pipeline: Pipeline) -> None:
    st.title("🧠 AI Document Assistant")
    st.markdown(
        "Upload PDFs or paste text, then ask questions and get accurate, "
        "cited answers grounded only in your documents — no hallucination."
    )

    doc_count = pipeline.vector_store.count()
    col1, col2, col3 = st.columns(3)
    col1.metric("Documents Indexed", len(pipeline.vector_store.list_sources()))
    col2.metric("Chunks Stored", doc_count)
    col3.metric("Chat Messages", len(st.session_state.chat_history))

    st.divider()

    # ------------------------------------------------------------------
    # Upload / Paste section
    # ------------------------------------------------------------------
    st.subheader("1️⃣ Add Documents")
    tab_upload, tab_paste = st.tabs(["📎 Upload PDFs", "✍️ Paste Text"])

    with tab_upload:
        files = st.file_uploader("Drag & drop PDF files", type=["pdf"], accept_multiple_files=True, key="home_upload")
        if files and st.button("🚀 Process Documents", type="primary", key="home_process"):
            _process_pdfs(pipeline, files)

    with tab_paste:
        text = st.text_area("Paste text here", height=150, key="home_paste_text")
        label = st.text_input("Label for this text", value="Pasted Text", key="home_paste_label")
        if st.button("🚀 Process Pasted Text", type="primary", key="home_process_text") and text.strip():
            _process_pasted_text(pipeline, text, label)

    st.divider()

    # ------------------------------------------------------------------
    # Ask a question
    # ------------------------------------------------------------------
    st.subheader("2️⃣ Ask a Question")

    if doc_count == 0:
        st.info("Add at least one document above before asking questions.", icon="📭")
    else:
        question = st.text_input("Your question", key="home_question", placeholder="What is this document about?")
        if st.button("🔎 Get Answer", type="primary", key="home_ask") and question.strip():
            with st.spinner("Retrieving context and generating answer..."):
                try:
                    result = pipeline.chain.answer_question(
                        question,
                        top_k=st.session_state.top_k,
                        temperature=st.session_state.temperature,
                        max_tokens=st.session_state.max_tokens,
                    )

                    st.markdown("#### ✅ Answer")
                    st.markdown(result.answer)

                    if result.sources:
                        label = format_confidence_label(result.confidence)
                        st.caption(f"Confidence: **{label}** ({result.confidence:.0%})")

                        st.markdown("#### 📑 Retrieved Sources")
                        for s in result.sources:
                            with st.expander(f"{s.source} — Page {s.page} (score: {s.score:.2f})"):
                                st.text(truncate(s.text, 800))

                    if result.llm_response:
                        track_usage(
                            result.llm_response.input_tokens,
                            result.llm_response.output_tokens,
                            result.llm_response.estimated_cost_usd,
                        )

                    sources_payload = [
                        {"source": s.source, "page": s.page, "score": s.score, "text": s.text}
                        for s in result.sources
                    ]
                    add_chat_message("user", question)
                    add_chat_message("assistant", result.answer, sources=sources_payload, confidence=result.confidence)

                    col1, col2 = st.columns(2)
                    col1.download_button("⬇️ Download Answer", data=result.answer, file_name="answer.txt")
                    col2.caption("💡 Tip: use the copy icon in the code block below to copy the answer.")
                    st.code(result.answer, language=None)

                except LLMError as exc:
                    st.error(f"LLM error: {exc}")
                except Exception as exc:
                    logger.exception("Unexpected error answering question")
                    st.error(f"Unexpected error: {exc}")

    st.divider()

    # ------------------------------------------------------------------
    # Chat history (condensed view; full experience on pages/chat.py)
    # ------------------------------------------------------------------
    st.subheader("🕘 Recent Chat History")
    if not st.session_state.chat_history:
        st.caption("No questions asked yet.")
    else:
        for msg in st.session_state.chat_history[-6:]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
        st.caption("See the full Chat page (sidebar link) for export, quiz, summary, and translation tools.")


def _process_pdfs(pipeline: Pipeline, files) -> None:
    """Shared PDF processing logic used by both the sidebar and home page uploaders."""
    loader = PDFLoader(ocr_enabled=True, tesseract_cmd=settings.tesseract_cmd or None)
    splitter = TextSplitter(chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)

    progress = st.progress(0, text="Starting...")
    added_total = 0

    for i, f in enumerate(files):
        progress.progress(int((i / len(files)) * 100), text=f"Processing {f.name}...")
        try:
            with st.spinner(f"Extracting {f.name}..."):
                path = save_uploaded_file(f, settings.upload_dir)
                pages = loader.load(path, source_name=f.name)
            chunks = splitter.split_pages(pages)
            added = pipeline.vector_store.add_chunks(chunks)
            added_total += added

            if f.name not in st.session_state.processed_sources:
                st.session_state.processed_sources.append(f.name)

            st.session_state.last_full_text = "\n\n".join(p.text for p in pages)
            st.session_state.last_doc_name = f.name

            st.toast(f"✅ {f.name} processed", icon="✅")
        except DocumentLoadError as exc:
            st.toast(f"❌ {f.name}: {exc}", icon="❌")
        except Exception as exc:
            logger.exception("Error processing %s", f.name)
            st.error(f"Unexpected error on {f.name}: {exc}")

    progress.progress(100, text="Done.")
    if added_total:
        st.success(f"Embedded {added_total} new chunk(s).")
    st.rerun()


def _process_pasted_text(pipeline: Pipeline, text: str, label: str = "Pasted Text") -> None:
    """Shared pasted-text processing logic used by both the sidebar and home page."""
    try:
        with st.spinner("Embedding pasted text..."):
            pages = load_pasted_text(text, source_name=label)
            splitter = TextSplitter(chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)
            chunks = splitter.split_pages(pages)
            added = pipeline.vector_store.add_chunks(chunks)

        if label not in st.session_state.processed_sources:
            st.session_state.processed_sources.append(label)

        st.session_state.last_full_text = text
        st.session_state.last_doc_name = label

        st.success(f"Embedded {added} new chunk(s) from '{label}'.")
        st.rerun()
    except DocumentLoadError as exc:
        st.error(str(exc))


if __name__ == "__main__":
    main()

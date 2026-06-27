"""
pages/upload.py
----------------
Streamlit page for uploading PDFs, pasting raw text, and running the
extract -> clean -> split -> embed -> store pipeline with progress feedback.
"""

from __future__ import annotations

import logging

import streamlit as st

from config import settings
from rag.loader import DocumentLoadError, PDFLoader, load_pasted_text
from rag.splitter import TextSplitter
from utils.helpers import init_session_state, save_uploaded_file

logger = logging.getLogger(__name__)

st.set_page_config(page_title="Upload - AI Document Assistant", page_icon="📄", layout="wide")
init_session_state()

st.title("📄 Upload Documents")
st.caption("Upload PDF files or paste plain text. Documents are chunked, embedded, and stored locally in ChromaDB.")

if "pipeline" not in st.session_state or st.session_state.pipeline is None:
    st.warning("The RAG pipeline isn't initialized yet. Go back to the main page first.", icon="⚠️")
    st.stop()

pipeline = st.session_state.pipeline

tab_upload, tab_paste = st.tabs(["📎 Upload PDFs", "✍️ Paste Text"])

# ---------------------------------------------------------------------------
# Tab 1: PDF Upload
# ---------------------------------------------------------------------------
with tab_upload:
    uploaded_files = st.file_uploader(
        "Drag & drop one or more PDF files here",
        type=["pdf"],
        accept_multiple_files=True,
        help="Scanned/image-only PDFs are supported via OCR if Tesseract is installed.",
    )

    if uploaded_files:
        st.write(f"**{len(uploaded_files)} file(s) selected:**")
        for f in uploaded_files:
            st.write(f"- {f.name} ({f.size / 1024:.1f} KB)")

        if st.button("🚀 Process PDFs", type="primary", key="process_pdfs"):
            loader = PDFLoader(ocr_enabled=True, tesseract_cmd=settings.tesseract_cmd or None)
            splitter = TextSplitter(chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)

            progress = st.progress(0, text="Starting...")
            total_chunks_added = 0
            errors = []

            for i, uploaded_file in enumerate(uploaded_files):
                pct = int(((i) / len(uploaded_files)) * 100)
                progress.progress(pct, text=f"Processing {uploaded_file.name}...")
                try:
                    with st.spinner(f"Extracting text from {uploaded_file.name}..."):
                        path = save_uploaded_file(uploaded_file, settings.upload_dir)
                        pages = loader.load(path, source_name=uploaded_file.name)

                    with st.spinner(f"Chunking and embedding {uploaded_file.name}..."):
                        chunks = splitter.split_pages(pages)
                        added = pipeline.vector_store.add_chunks(chunks)
                        total_chunks_added += added

                    if uploaded_file.name not in st.session_state.processed_sources:
                        st.session_state.processed_sources.append(uploaded_file.name)

                    # Keep full text available for summary/quiz/etc. bonus features
                    st.session_state.last_full_text = "\n\n".join(p.text for p in pages)
                    st.session_state.last_doc_name = uploaded_file.name

                    st.toast(f"✅ {uploaded_file.name} processed ({added} new chunks)", icon="✅")

                except DocumentLoadError as exc:
                    errors.append(f"{uploaded_file.name}: {exc}")
                    st.toast(f"❌ Failed: {uploaded_file.name}", icon="❌")
                except Exception as exc:
                    logger.exception("Unexpected error processing %s", uploaded_file.name)
                    errors.append(f"{uploaded_file.name}: Unexpected error - {exc}")

            progress.progress(100, text="Done.")

            if total_chunks_added:
                st.success(f"Successfully embedded {total_chunks_added} new chunk(s) into the vector store.")
            if errors:
                st.error("Some files failed to process:\n\n" + "\n".join(f"- {e}" for e in errors))

# ---------------------------------------------------------------------------
# Tab 2: Paste Text
# ---------------------------------------------------------------------------
with tab_paste:
    pasted = st.text_area(
        "Paste your text here",
        height=250,
        placeholder="Paste an article, notes, or any text you want to ask questions about...",
        key="pasted_text_input",
    )
    label = st.text_input("Give this text a label (used as the source name)", value="Pasted Text")

    if st.button("🚀 Process Pasted Text", type="primary", key="process_pasted"):
        if not pasted.strip():
            st.error("Please paste some text first.")
        else:
            try:
                with st.spinner("Chunking and embedding text..."):
                    pages = load_pasted_text(pasted, source_name=label or "Pasted Text")
                    splitter = TextSplitter(chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)
                    chunks = splitter.split_pages(pages)
                    added = pipeline.vector_store.add_chunks(chunks)

                if label not in st.session_state.processed_sources:
                    st.session_state.processed_sources.append(label)

                st.session_state.last_full_text = pasted
                st.session_state.last_doc_name = label

                st.success(f"Embedded {added} new chunk(s) from '{label}'.")
            except DocumentLoadError as exc:
                st.error(str(exc))

# ---------------------------------------------------------------------------
# Currently indexed documents + deletion
# ---------------------------------------------------------------------------
st.divider()
st.subheader("📚 Indexed Documents")

sources = pipeline.vector_store.list_sources()
if not sources:
    st.info("No documents indexed yet.")
else:
    for src in sources:
        col1, col2 = st.columns([4, 1])
        col1.write(f"📄 {src}")
        if col2.button("🗑️ Delete", key=f"delete_{src}"):
            deleted = pipeline.vector_store.delete_source(src)
            if src in st.session_state.processed_sources:
                st.session_state.processed_sources.remove(src)
            st.success(f"Deleted {deleted} chunk(s) for '{src}'.")
            st.rerun()

st.caption(f"Total chunks in vector database: **{pipeline.vector_store.count()}**")

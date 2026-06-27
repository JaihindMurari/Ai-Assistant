"""
pages/chat.py
--------------
Main Q&A chat interface: ask questions, view answers with citations and
confidence scores, inspect retrieved chunks, and use document-level bonus
features (summary, key points, quiz, flashcards, translate).
"""

from __future__ import annotations

import logging

import streamlit as st

from rag.llm import LLMError
from utils.helpers import (
    add_chat_message,
    export_chat_history_as_json,
    export_chat_history_as_markdown,
    format_confidence_label,
    init_session_state,
    track_usage,
    truncate,
)

logger = logging.getLogger(__name__)

st.set_page_config(page_title="Chat - AI Document Assistant", page_icon="💬", layout="wide")
init_session_state()

st.title("💬 Chat with Your Documents")

if "pipeline" not in st.session_state or st.session_state.pipeline is None:
    st.warning("The RAG pipeline isn't initialized yet. Go back to the main page first.", icon="⚠️")
    st.stop()

pipeline = st.session_state.pipeline
doc_count = pipeline.vector_store.count()

if doc_count == 0:
    st.info("No documents indexed yet. Head to the **Upload** page to add PDFs or pasted text.", icon="📭")

# ---------------------------------------------------------------------------
# Chat history
# ---------------------------------------------------------------------------
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            label = format_confidence_label(msg["confidence"])
            st.caption(f"Confidence: **{label}** ({msg['confidence']:.0%})")
            with st.expander(f"📑 View {len(msg['sources'])} source(s)"):
                for s in msg["sources"]:
                    st.markdown(f"**{s.get('source', 'Unknown')}** — Page {s.get('page', '?')} (score: {s.get('score', 0):.2f})")
                    st.text(truncate(s.get("text", ""), 500))
                    st.divider()

# ---------------------------------------------------------------------------
# Question input
# ---------------------------------------------------------------------------
question = st.chat_input("Ask a question about your documents...", disabled=(doc_count == 0))

if question:
    add_chat_message("user", question)
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = pipeline.chain.answer_question(
                    question,
                    top_k=st.session_state.top_k,
                    temperature=st.session_state.temperature,
                    max_tokens=st.session_state.max_tokens,
                )
                st.markdown(result.answer)

                sources_payload = [
                    {"source": s.source, "page": s.page, "score": s.score, "text": s.text} for s in result.sources
                ]

                if result.sources:
                    label = format_confidence_label(result.confidence)
                    st.caption(f"Confidence: **{label}** ({result.confidence:.0%})")
                    with st.expander(f"📑 View {len(result.sources)} source(s)"):
                        for s in result.sources:
                            st.markdown(f"**{s.source}** — Page {s.page} (score: {s.score:.2f})")
                            st.text(truncate(s.text, 500))
                            st.divider()

                if result.llm_response:
                    track_usage(
                        result.llm_response.input_tokens,
                        result.llm_response.output_tokens,
                        result.llm_response.estimated_cost_usd,
                    )

                add_chat_message("assistant", result.answer, sources=sources_payload, confidence=result.confidence)

                col1, col2 = st.columns(2)
                col1.download_button(
                    "⬇️ Download Answer",
                    data=result.answer,
                    file_name="answer.txt",
                    mime="text/plain",
                    key=f"dl_{len(st.session_state.chat_history)}",
                )
                col2.code(result.answer, language=None)  # acts as a built-in "copy" affordance via the code block

            except LLMError as exc:
                st.error(f"LLM error: {exc}")
            except Exception as exc:
                logger.exception("Unexpected error answering question")
                st.error(f"Unexpected error: {exc}")

# ---------------------------------------------------------------------------
# Export chat history
# ---------------------------------------------------------------------------
if st.session_state.chat_history:
    st.divider()
    col1, col2, col3 = st.columns(3)
    col1.download_button(
        "⬇️ Export Chat (Markdown)",
        data=export_chat_history_as_markdown(),
        file_name="chat_export.md",
        mime="text/markdown",
    )
    col2.download_button(
        "⬇️ Export Chat (JSON)",
        data=export_chat_history_as_json(),
        file_name="chat_export.json",
        mime="application/json",
    )
    if col3.button("🧹 Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

# ---------------------------------------------------------------------------
# Document-level bonus features
# ---------------------------------------------------------------------------
st.divider()
st.subheader("✨ Document Tools")

if not st.session_state.last_full_text:
    st.caption("Process a document on the Upload page to enable summary, quiz, and other tools.")
else:
    st.caption(f"Tools below apply to the most recently processed document: **{st.session_state.last_doc_name}**")
    tool_tabs = st.tabs(["📝 Summary", "🔑 Key Points", "❓ Quiz", "🗂️ Flashcards", "🌐 Translate"])

    text_for_tools = truncate(st.session_state.last_full_text, 8000)

    with tool_tabs[0]:
        if st.button("Generate Summary"):
            with st.spinner("Summarizing..."):
                try:
                    resp = pipeline.chain.summarize(text_for_tools)
                    st.markdown(resp.text)
                    track_usage(resp.input_tokens, resp.output_tokens, resp.estimated_cost_usd)
                except LLMError as exc:
                    st.error(str(exc))

    with tool_tabs[1]:
        if st.button("Extract Key Points"):
            with st.spinner("Extracting key points..."):
                try:
                    resp = pipeline.chain.extract_key_points(text_for_tools)
                    st.markdown(resp.text)
                    track_usage(resp.input_tokens, resp.output_tokens, resp.estimated_cost_usd)
                except LLMError as exc:
                    st.error(str(exc))

    with tool_tabs[2]:
        num_q = st.slider("Number of questions", 3, 10, 5, key="quiz_num_q")
        if st.button("Generate Quiz"):
            with st.spinner("Generating quiz..."):
                try:
                    resp = pipeline.chain.generate_quiz(text_for_tools, num_questions=num_q)
                    st.markdown(resp.text)
                    track_usage(resp.input_tokens, resp.output_tokens, resp.estimated_cost_usd)
                except LLMError as exc:
                    st.error(str(exc))

    with tool_tabs[3]:
        num_cards = st.slider("Number of flashcards", 4, 15, 8, key="flashcard_num")
        if st.button("Generate Flashcards"):
            with st.spinner("Generating flashcards..."):
                try:
                    resp = pipeline.chain.generate_flashcards(text_for_tools, num_cards=num_cards)
                    st.markdown(resp.text)
                    track_usage(resp.input_tokens, resp.output_tokens, resp.estimated_cost_usd)
                except LLMError as exc:
                    st.error(str(exc))

    with tool_tabs[4]:
        target_lang = st.selectbox(
            "Target language", ["Spanish", "French", "German", "Hindi", "Japanese", "Arabic", "Portuguese"]
        )
        last_answer = next(
            (m["content"] for m in reversed(st.session_state.chat_history) if m["role"] == "assistant"), None
        )
        if not last_answer:
            st.caption("Ask a question first, then translate the latest answer here.")
        elif st.button("Translate Last Answer"):
            with st.spinner("Translating..."):
                try:
                    resp = pipeline.chain.translate_answer(last_answer, target_lang)
                    st.markdown(resp.text)
                    track_usage(resp.input_tokens, resp.output_tokens, resp.estimated_cost_usd)
                except LLMError as exc:
                    st.error(str(exc))

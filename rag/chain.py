"""
rag/chain.py
------------
Orchestrates the full RAG flow: retrieve context -> build prompt -> call LLM
-> return a structured answer with citations. Also hosts the document-level
bonus features (summary, key points, quiz, flashcards, translation) since
they share the same "retrieve-or-use-all-context then prompt LLM" pattern.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List

from rag.llm import BaseLLMProvider, LLMResponse
from rag.retriever import Retriever, RetrievedChunk

logger = logging.getLogger(__name__)

NOT_FOUND_MESSAGE = "I could not find that information in the uploaded documents."

QA_PROMPT_TEMPLATE = """You are an intelligent assistant answering questions about uploaded documents.

Answer ONLY using the provided context below. Do not use outside knowledge.
If the answer is not present in the context, respond exactly with:
"{not_found}"

Never hallucinate facts. Always cite the page number(s) when available, in the
format (Source: <name>, Page <n>) immediately after the relevant statement.

Context:
{context}

Question: {question}

Answer:"""


@dataclass
class RAGAnswer:
    """Final structured result returned to the UI layer."""

    answer: str
    sources: List[RetrievedChunk] = field(default_factory=list)
    confidence: float = 0.0
    llm_response: LLMResponse | None = None


class RAGChain:
    """Ties together retrieval and generation for question answering."""

    def __init__(self, retriever: Retriever, llm: BaseLLMProvider):
        self.retriever = retriever
        self.llm = llm

    def answer_question(self, question: str, top_k: int, temperature: float, max_tokens: int) -> RAGAnswer:
        """Run the full retrieve -> prompt -> generate pipeline for a question."""
        if not question.strip():
            return RAGAnswer(answer="Please enter a question.", sources=[], confidence=0.0)

        chunks = self.retriever.retrieve(question, top_k=top_k)

        if not chunks:
            return RAGAnswer(answer=NOT_FOUND_MESSAGE, sources=[], confidence=0.0)

        context = self.retriever.build_context(chunks)
        prompt = QA_PROMPT_TEMPLATE.format(not_found=NOT_FOUND_MESSAGE, context=context, question=question)

        llm_response = self.llm.generate(prompt, temperature=temperature, max_tokens=max_tokens)
        confidence = self.retriever.average_confidence(chunks)

        return RAGAnswer(
            answer=llm_response.text or NOT_FOUND_MESSAGE,
            sources=chunks,
            confidence=confidence,
            llm_response=llm_response,
        )

    # ------------------------------------------------------------------
    # Bonus document-level features
    # ------------------------------------------------------------------
    def summarize(self, full_text: str, max_tokens: int = 512) -> LLMResponse:
        """Generate a concise summary of the supplied document text."""
        prompt = (
            "Summarize the following document in clear, concise prose. "
            "Focus on the main ideas and conclusions. Limit the summary to 200-300 words.\n\n"
            f"Document:\n{full_text}\n\nSummary:"
        )
        return self.llm.generate(prompt, temperature=0.3, max_tokens=max_tokens)

    def extract_key_points(self, full_text: str, max_tokens: int = 512) -> LLMResponse:
        """Extract key points as a bulleted list."""
        prompt = (
            "Extract the most important key points from the following document. "
            "Return them as a concise bulleted list (max 10 bullets).\n\n"
            f"Document:\n{full_text}\n\nKey Points:"
        )
        return self.llm.generate(prompt, temperature=0.3, max_tokens=max_tokens)

    def generate_quiz(self, full_text: str, num_questions: int = 5, max_tokens: int = 800) -> LLMResponse:
        """Generate a short multiple-choice quiz based on the document."""
        prompt = (
            f"Create a {num_questions}-question multiple-choice quiz based on the following document. "
            "For each question, provide 4 options labeled A-D and indicate the correct answer at the end "
            "of each question like 'Correct Answer: B'.\n\n"
            f"Document:\n{full_text}\n\nQuiz:"
        )
        return self.llm.generate(prompt, temperature=0.4, max_tokens=max_tokens)

    def generate_flashcards(self, full_text: str, num_cards: int = 8, max_tokens: int = 700) -> LLMResponse:
        """Generate question/answer flashcards based on the document."""
        prompt = (
            f"Create {num_cards} flashcards based on the following document. "
            "Format each as 'Q: <question>' on one line and 'A: <answer>' on the next line.\n\n"
            f"Document:\n{full_text}\n\nFlashcards:"
        )
        return self.llm.generate(prompt, temperature=0.4, max_tokens=max_tokens)

    def translate_answer(self, text: str, target_language: str, max_tokens: int = 512) -> LLMResponse:
        """Translate a given answer/text into the target language."""
        prompt = f"Translate the following text into {target_language}. Return only the translation:\n\n{text}"
        return self.llm.generate(prompt, temperature=0.2, max_tokens=max_tokens)

    def compare_documents(self, text_a: str, name_a: str, text_b: str, name_b: str, max_tokens: int = 800) -> LLMResponse:
        """Compare two documents and summarize similarities/differences."""
        prompt = (
            f"Compare the following two documents. Summarize key similarities and differences "
            f"in a structured format with two sections: 'Similarities' and 'Differences'.\n\n"
            f"Document A ({name_a}):\n{text_a[:6000]}\n\n"
            f"Document B ({name_b}):\n{text_b[:6000]}\n\nComparison:"
        )
        return self.llm.generate(prompt, temperature=0.3, max_tokens=max_tokens)

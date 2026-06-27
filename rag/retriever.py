"""
rag/retriever.py
-----------------
Translates a user question into retrieved context chunks. Kept separate
from vector_store.py so that retrieval-specific logic (top-k selection,
confidence aggregation, formatting for the LLM prompt) doesn't bloat the
storage layer.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

from rag.vector_store import VectorStore

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    """A chunk returned from retrieval, ready for display and prompting."""

    text: str
    source: str
    page: int
    score: float


class Retriever:
    """Performs semantic similarity search against the vector store."""

    def __init__(self, vector_store: VectorStore, top_k: int = 4):
        self.vector_store = vector_store
        self.top_k = top_k

    def retrieve(self, question: str, top_k: Optional[int] = None) -> List[RetrievedChunk]:
        """Retrieve the most relevant chunks for a given question."""
        k = top_k or self.top_k
        raw_results = self.vector_store.similarity_search(question, top_k=k)

        chunks = [
            RetrievedChunk(
                text=r["text"],
                source=r["metadata"].get("source", "Unknown"),
                page=r["metadata"].get("page", 0),
                score=r["score"],
            )
            for r in raw_results
        ]
        logger.info("Retrieved %d chunk(s) for question: %r", len(chunks), question[:60])
        return chunks

    @staticmethod
    def average_confidence(chunks: List[RetrievedChunk]) -> float:
        """Average similarity score across retrieved chunks, used as a confidence proxy."""
        if not chunks:
            return 0.0
        return round(sum(c.score for c in chunks) / len(chunks), 4)

    @staticmethod
    def build_context(chunks: List[RetrievedChunk]) -> str:
        """Format retrieved chunks into a single context block for the LLM prompt."""
        parts = []
        for i, chunk in enumerate(chunks, start=1):
            parts.append(f"[Source {i}: {chunk.source}, Page {chunk.page}]\n{chunk.text}")
        return "\n\n---\n\n".join(parts)


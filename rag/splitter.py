"""
rag/splitter.py
----------------
Splits page-level text into overlapping chunks suitable for embedding,
while preserving page number and source metadata on every chunk so that
citations can later point back to an exact page.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.loader import PageContent

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """A single chunk of text ready for embedding, with traceable metadata."""

    text: str
    source: str
    page_number: int
    chunk_id: str
    metadata: dict = field(default_factory=dict)


class TextSplitter:
    """Wraps LangChain's RecursiveCharacterTextSplitter with page-aware chunking."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def split_pages(self, pages: List[PageContent]) -> List[Chunk]:
        """Split a list of PageContent objects into Chunk objects."""
        chunks: List[Chunk] = []

        for page in pages:
            pieces = self._splitter.split_text(page.text)
            for idx, piece in enumerate(pieces):
                chunk_id = self._make_chunk_id(page.source, page.page_number, idx, piece)
                chunks.append(
                    Chunk(
                        text=piece,
                        source=page.source,
                        page_number=page.page_number,
                        chunk_id=chunk_id,
                        metadata={
                            "source": page.source,
                            "page": page.page_number,
                            "chunk_index": idx,
                        },
                    )
                )

        logger.info("Split %d page(s) into %d chunk(s).", len(pages), len(chunks))
        return chunks

    @staticmethod
    def _make_chunk_id(source: str, page: int, idx: int, text: str) -> str:
        """Deterministic ID so re-processing the same content doesn't duplicate vectors."""
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        safe_source = source.replace(" ", "_")
        return f"{safe_source}_p{page}_c{idx}_{digest}"

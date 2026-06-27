"""
rag/vector_store.py
--------------------
Thin wrapper around a persistent ChromaDB collection. Handles:
  - storing chunk embeddings with metadata
  - avoiding duplicate embeddings (via deterministic chunk IDs)
  - similarity search
  - keyword (substring) search, for the "search without asking questions" bonus feature
  - deleting by source document or clearing the whole collection
"""

from __future__ import annotations

import logging
from typing import List, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from rag.embeddings import SentenceTransformerEmbeddings
from rag.splitter import Chunk

logger = logging.getLogger(__name__)


class VectorStore:
    """Persistent ChromaDB-backed vector store for document chunks."""

    def __init__(
        self,
        persist_dir: str,
        collection_name: str,
        embedder: SentenceTransformerEmbeddings,
    ):
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.embedder = embedder

        self._client = chromadb.PersistentClient(
            path=persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(name=collection_name)

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------
    def add_chunks(self, chunks: List[Chunk]) -> int:
        """
        Embed and store chunks. Chunks whose chunk_id already exists in the
        collection are skipped, preventing duplicate embeddings when the
        same document is processed more than once.
        """
        if not chunks:
            return 0

        existing_ids = set(self._collection.get(ids=[c.chunk_id for c in chunks]).get("ids", []))
        new_chunks = [c for c in chunks if c.chunk_id not in existing_ids]

        if not new_chunks:
            logger.info("All %d chunk(s) already exist in the vector store; skipping.", len(chunks))
            return 0

        texts = [c.text for c in new_chunks]
        embeddings = self.embedder.embed_documents(texts)
        ids = [c.chunk_id for c in new_chunks]
        metadatas = [c.metadata for c in new_chunks]

        self._collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )
        logger.info(
            "Added %d new chunk(s) to collection '%s' (%d duplicate(s) skipped).",
            len(new_chunks),
            self.collection_name,
            len(chunks) - len(new_chunks),
        )
        return len(new_chunks)

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------
    def similarity_search(self, query: str, top_k: int = 4) -> List[dict]:
        """Return the top-k most semantically similar chunks to the query."""
        if self.count() == 0:
            return []

        query_embedding = self.embedder.embed_query(query)
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, max(self.count(), 1)),
        )
        return self._format_results(results)

    def keyword_search(self, keyword: str, limit: int = 20) -> List[dict]:
        """Simple case-insensitive substring search across stored documents."""
        if not keyword.strip() or self.count() == 0:
            return []

        all_docs = self._collection.get(include=["documents", "metadatas"])
        matches = []
        keyword_lower = keyword.lower()

        for doc, meta, doc_id in zip(all_docs["documents"], all_docs["metadatas"], all_docs["ids"]):
            if keyword_lower in doc.lower():
                matches.append({"id": doc_id, "text": doc, "metadata": meta})
            if len(matches) >= limit:
                break

        return matches

    def count(self) -> int:
        """Number of chunks currently stored."""
        return self._collection.count()

    def list_sources(self) -> List[str]:
        """Return the distinct set of source document names currently indexed."""
        if self.count() == 0:
            return []
        all_meta = self._collection.get(include=["metadatas"])
        sources = {m.get("source", "Unknown") for m in all_meta["metadatas"]}
        return sorted(sources)

    # ------------------------------------------------------------------
    # Delete operations
    # ------------------------------------------------------------------
    def delete_source(self, source: str) -> int:
        """Delete all chunks belonging to a specific source document."""
        existing = self._collection.get(where={"source": source})
        ids = existing.get("ids", [])
        if ids:
            self._collection.delete(ids=ids)
            logger.info("Deleted %d chunk(s) for source '%s'.", len(ids), source)
        return len(ids)

    def clear(self) -> None:
        """Delete the entire collection and recreate it empty (rebuild index)."""
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.get_or_create_collection(name=self.collection_name)
        logger.info("Cleared collection '%s'.", self.collection_name)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _format_results(results: dict) -> List[dict]:
        """Normalize Chroma's query() output into a flat list of dicts."""
        formatted = []
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        ids = results.get("ids", [[]])[0]

        for doc, meta, dist, _id in zip(docs, metas, distances, ids):
            # Chroma returns L2 distance for normalized embeddings; convert to
            # an approximate 0-1 similarity/confidence score for display.
            similarity = max(0.0, 1.0 - (dist / 2.0))
            formatted.append(
                {
                    "id": _id,
                    "text": doc,
                    "metadata": meta,
                    "score": round(similarity, 4),
                }
            )
        return formatted

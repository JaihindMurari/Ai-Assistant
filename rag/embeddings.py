"""
rag/embeddings.py
------------------
Wraps the sentence-transformers embedding model behind a small interface.
The model itself is cached as a singleton (loading it is expensive) and
embedding generation is batched for performance.
"""

from __future__ import annotations

import logging
from typing import List

from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)

_MODEL_CACHE: dict = {}


class SentenceTransformerEmbeddings(Embeddings):
    """
    LangChain-compatible embedding class backed by sentence-transformers.

    The underlying model is cached at the module level so that repeated
    instantiation (e.g. across Streamlit reruns) doesn't reload weights
    from disk every time.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = self._load_model(model_name)

    @staticmethod
    def _load_model(model_name: str):
        if model_name not in _MODEL_CACHE:
            logger.info("Loading embedding model '%s' (first load may take a moment)...", model_name)
            from sentence_transformers import SentenceTransformer

            _MODEL_CACHE[model_name] = SentenceTransformer(model_name)
            logger.info("Embedding model '%s' loaded and cached.", model_name)
        return _MODEL_CACHE[model_name]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of documents. Required by LangChain's Embeddings interface."""
        if not texts:
            return []
        embeddings = self._model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query string. Required by LangChain's Embeddings interface."""
        embedding = self._model.encode(
            [text],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embedding[0].tolist()

    def get_dimension(self) -> int:
        """Return the embedding vector dimension for this model."""
        return self._model.get_sentence_embedding_dimension()

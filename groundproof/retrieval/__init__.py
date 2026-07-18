"""Vector store port, temporal filtering, recency ranking, supersedence resolution (P1-P2)."""

from groundproof.retrieval.embeddings import (
    EmbeddingClient,
    MockEmbeddingClient,
    cosine_similarity,
)

__all__ = ["EmbeddingClient", "MockEmbeddingClient", "cosine_similarity"]

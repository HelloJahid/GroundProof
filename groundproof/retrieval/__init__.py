"""Vector store port, temporal filtering, recency ranking, supersedence resolution (P1-P2).

``ChromaVectorStore`` is intentionally NOT re-exported here — importing
chromadb is heavyweight, so wire it explicitly from
:mod:`groundproof.retrieval.chroma_store` where the live backend is needed.
"""

from groundproof.retrieval.embeddings import (
    EmbeddingClient,
    MockEmbeddingClient,
    cosine_similarity,
)
from groundproof.retrieval.indexing import index_documents
from groundproof.retrieval.store import InMemoryVectorStore, ScoredChunk, VectorStore

__all__ = [
    "EmbeddingClient",
    "InMemoryVectorStore",
    "MockEmbeddingClient",
    "ScoredChunk",
    "VectorStore",
    "cosine_similarity",
    "index_documents",
]

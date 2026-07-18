"""The vector store port: index dated chunks, query by similarity.

``VectorStore`` is the contract; ``InMemoryVectorStore`` (here) and
``ChromaVectorStore`` (in :mod:`groundproof.retrieval.chroma_store`) both
fulfil it and are exercised by the same contract tests — swapping one for the
other is a wiring change, never a pipeline change.

Every indexed chunk keeps its temporal metadata, so P2's as-of filtering is a
query-time concern, not a re-indexing one.
"""

from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

from groundproof.ingest.models import Chunk
from groundproof.retrieval.embeddings import cosine_similarity


class ScoredChunk(BaseModel):
    """One retrieval hit: the chunk plus its similarity score (higher = closer)."""

    model_config = ConfigDict(frozen=True)

    chunk: Chunk
    score: float


@runtime_checkable
class VectorStore(Protocol):
    """Anything that can index dated chunks and rank them against a query vector."""

    def index(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        """Add (or overwrite, by chunk_id) each chunk with its vector."""
        ...

    def query(self, vector: list[float], top_k: int = 5) -> list[ScoredChunk]:
        """Return up to ``top_k`` chunks, best score first."""
        ...

    def count(self) -> int:
        """Number of chunks currently indexed."""
        ...


class InMemoryVectorStore:
    """Exact-cosine, pure-Python store: the deterministic test workhorse.

    Keyed by ``chunk_id``, so re-indexing the same chunk overwrites instead of
    duplicating — the same idempotency Chroma's upsert gives the live backend.
    Ties in score break by ``chunk_id`` so result order is fully deterministic.
    """

    def __init__(self) -> None:
        self._entries: dict[str, tuple[Chunk, list[float]]] = {}

    def index(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        for chunk, vector in zip(chunks, vectors, strict=True):
            self._entries[chunk.chunk_id] = (chunk, vector)

    def query(self, vector: list[float], top_k: int = 5) -> list[ScoredChunk]:
        scored = [
            ScoredChunk(chunk=chunk, score=cosine_similarity(vector, stored_vector))
            for chunk, stored_vector in self._entries.values()
        ]
        scored.sort(key=lambda hit: (-hit.score, hit.chunk.chunk_id))
        return scored[:top_k]

    def count(self) -> int:
        return len(self._entries)

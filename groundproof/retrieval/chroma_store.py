"""Chroma-backed VectorStore: persistent, cosine-space, idempotent via upsert.

Deliberately kept OUT of the package façade (``groundproof.retrieval``
re-exports everything except this): importing chromadb is heavyweight, so only
the code that actually wires the live backend pays that cost. Collections are
created in cosine space, so Chroma's distance is ``1 - cosine_similarity`` and
scores stay comparable with the in-memory store's.
"""

from datetime import date
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings

from groundproof.ingest.models import Chunk
from groundproof.retrieval.store import ScoredChunk

_COSINE = {"hnsw:space": "cosine"}
# Telemetry off: the test suite (and this project's discipline) never dials out.
_SETTINGS = Settings(anonymized_telemetry=False)


class ChromaVectorStore:
    """VectorStore over a Chroma collection (must be created with cosine space)."""

    def __init__(self, collection: Any) -> None:
        self._collection = collection

    @classmethod
    def persistent(cls, path: Path, collection_name: str = "groundproof") -> "ChromaVectorStore":
        """Open (or create) a cosine-space collection persisted under ``path``."""
        client = chromadb.PersistentClient(path=str(path), settings=_SETTINGS)
        return cls(client.get_or_create_collection(collection_name, metadata=_COSINE))

    @classmethod
    def ephemeral(cls, collection_name: str = "groundproof") -> "ChromaVectorStore":
        """A throwaway in-process collection — used by contract tests."""
        client = chromadb.EphemeralClient(settings=_SETTINGS)
        return cls(client.get_or_create_collection(collection_name, metadata=_COSINE))

    def index(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        if not chunks:
            return
        if len(chunks) != len(vectors):
            raise ValueError(f"{len(chunks)} chunks but {len(vectors)} vectors")
        self._collection.upsert(
            ids=[chunk.chunk_id for chunk in chunks],
            embeddings=vectors,
            documents=[chunk.text for chunk in chunks],
            metadatas=[_to_metadata(chunk) for chunk in chunks],
        )

    def query(self, vector: list[float], top_k: int = 5) -> list[ScoredChunk]:
        n_results = min(top_k, self.count())
        if n_results == 0:
            return []
        result = self._collection.query(
            query_embeddings=[vector],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )
        return [
            ScoredChunk(chunk=_to_chunk(chunk_id, text, metadata), score=1.0 - distance)
            for chunk_id, text, metadata, distance in zip(
                result["ids"][0],
                result["documents"][0],
                result["metadatas"][0],
                result["distances"][0],
                strict=True,
            )
        ]

    def count(self) -> int:
        return self._collection.count()


def _to_metadata(chunk: Chunk) -> dict[str, Any]:
    """Flatten a chunk's provenance for Chroma; dates go in twice — ISO for
    humans, ordinal int so P2 can range-filter with ``$lte``."""
    metadata: dict[str, Any] = {
        "doc_id": chunk.doc_id,
        "source": chunk.source,
        "position": chunk.position,
        "observed_at": chunk.observed_at.isoformat(),
        "observed_at_ord": chunk.observed_at.toordinal(),
    }
    if chunk.valid_from is not None:
        metadata["valid_from"] = chunk.valid_from.isoformat()
    if chunk.valid_to is not None:
        metadata["valid_to"] = chunk.valid_to.isoformat()
    return metadata


def _to_chunk(chunk_id: str, text: str, metadata: dict[str, Any]) -> Chunk:
    """Rebuild the validated Chunk from what Chroma stored."""
    return Chunk(
        chunk_id=chunk_id,
        doc_id=metadata["doc_id"],
        source=metadata["source"],
        text=text,
        position=metadata["position"],
        observed_at=date.fromisoformat(metadata["observed_at"]),
        valid_from=(
            date.fromisoformat(metadata["valid_from"]) if "valid_from" in metadata else None
        ),
        valid_to=(date.fromisoformat(metadata["valid_to"]) if "valid_to" in metadata else None),
    )

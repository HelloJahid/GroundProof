"""End-to-end ingestion wiring: documents -> chunks -> vectors -> indexed store.

The last mile of Phase 1. Both collaborators arrive as ports, so the same
function serves tests (mock embedder + in-memory store) and the live demo
(real embedder + persistent Chroma) unchanged.
"""

from groundproof.ingest.chunker import DEFAULT_MAX_CHARS, chunk_corpus
from groundproof.ingest.models import Document
from groundproof.retrieval.embeddings import EmbeddingClient
from groundproof.retrieval.store import VectorStore


def index_documents(
    documents: list[Document],
    embedder: EmbeddingClient,
    store: VectorStore,
    batch_size: int = 64,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> int:
    """Chunk every document, embed in batches, index; returns chunks indexed."""
    chunks = chunk_corpus(documents, max_chars=max_chars)
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        store.index(batch, embedder.embed_texts([chunk.text for chunk in batch]))
    return len(chunks)

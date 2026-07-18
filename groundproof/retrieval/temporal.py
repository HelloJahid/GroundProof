"""Time-aware retrieval: as-of filtering plus a tunable recency prior (Hook A).

The store already refuses to show the future (``as_of`` filters inside the
query); this module ranks the survivors. Similarity says "how on-topic",
recency says "how fresh", and the final score blends the two with a tunable
weight. Freshness decays exponentially with a half-life: at the default
365 days, a year-old fact scores half a today-fact — old truth is
*discounted*, never erased. Every number involved is a pure function of the
chunk dates and two knobs; no model call anywhere.
"""

from datetime import date

from pydantic import BaseModel, ConfigDict

from groundproof.ingest.models import Chunk
from groundproof.retrieval.embeddings import EmbeddingClient
from groundproof.retrieval.store import VectorStore

DEFAULT_RECENCY_WEIGHT = 0.3
DEFAULT_HALF_LIFE_DAYS = 365.0


class RankedChunk(BaseModel):
    """One retrieval hit with its full scoring breakdown kept visible.

    ``similarity`` and ``recency`` are preserved alongside the blended
    ``score`` so traces and the eval harness can see *why* a chunk ranked
    where it did — observability over opacity.
    """

    model_config = ConfigDict(frozen=True)

    chunk: Chunk
    similarity: float
    recency: float
    score: float


def recency_score(observed_at: date, as_of: date, half_life_days: float) -> float:
    """Exponential freshness decay: 1.0 at age zero, 0.5 per half-life elapsed."""
    age_days = max((as_of - observed_at).days, 0)
    return 0.5 ** (age_days / half_life_days)


class TemporalRetriever:
    """Retrieve as of a moment: filter out the future, rank the past by blend.

    ``fetch_k`` over-fetches from the store before re-ranking, so the recency
    prior has a wider pool than the final ``top_k`` — a fresher chunk sitting
    just below the similarity cut-off can still win a slot.
    """

    def __init__(
        self,
        store: VectorStore,
        embedder: EmbeddingClient,
        *,
        fetch_k: int = 20,
        recency_weight: float = DEFAULT_RECENCY_WEIGHT,
        half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
    ) -> None:
        if not 0.0 <= recency_weight <= 1.0:
            raise ValueError(f"recency_weight must be in [0, 1], got {recency_weight}")
        if half_life_days <= 0:
            raise ValueError(f"half_life_days must be positive, got {half_life_days}")
        self._store = store
        self._embedder = embedder
        self._fetch_k = fetch_k
        self._recency_weight = recency_weight
        self._half_life_days = half_life_days

    def retrieve(
        self, question: str, *, as_of: date | None = None, top_k: int = 5
    ) -> list[RankedChunk]:
        vector = self._embedder.embed_texts([question])[0]
        hits = self._store.query(vector, top_k=max(self._fetch_k, top_k), as_of=as_of)
        if not hits:
            return []
        # No explicit as_of means "now" — and the freshest thing this store has
        # ever observed IS its now. Anchoring to data keeps ranking deterministic.
        anchor = as_of if as_of is not None else max(hit.chunk.observed_at for hit in hits)
        weight = self._recency_weight
        ranked = [
            RankedChunk(
                chunk=hit.chunk,
                similarity=hit.score,
                recency=recency_score(hit.chunk.observed_at, anchor, self._half_life_days),
                score=(1.0 - weight) * hit.score
                + weight * recency_score(hit.chunk.observed_at, anchor, self._half_life_days),
            )
            for hit in hits
        ]
        ranked.sort(key=lambda item: (-item.score, item.chunk.chunk_id))
        return ranked[:top_k]

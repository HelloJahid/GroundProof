"""P2: time-aware retrieval — the DoD test lives here (time travel, mocked store)."""

from datetime import date

import pytest

from groundproof.ingest import Chunk
from groundproof.retrieval import (
    InMemoryVectorStore,
    MockEmbeddingClient,
    TemporalRetriever,
    recency_score,
)

EMBEDDER = MockEmbeddingClient()


def make_chunk(chunk_id: str, text: str, observed_at: date) -> Chunk:
    doc_id = chunk_id.rsplit(":", 1)[0]
    return Chunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        source=f"https://example.test/{doc_id}.rst.txt",
        text=text,
        position=0,
        observed_at=observed_at,
    )


def store_with(chunks: list[Chunk]) -> InMemoryVectorStore:
    store = InMemoryVectorStore()
    store.index(chunks, EMBEDDER.embed_texts([chunk.text for chunk in chunks]))
    return store


VERSION_CHUNKS = [
    make_chunk(
        "python-whatsnew-3.12:0000",
        "Latest Version\n\nPython 3.12 is the latest stable release of Python.",
        date(2023, 10, 2),
    ),
    make_chunk(
        "python-whatsnew-3.13:0000",
        "Latest Version\n\nPython 3.13 is the latest stable release of Python.",
        date(2024, 10, 7),
    ),
]


class TestRecencyScore:
    def test_age_zero_scores_one(self):
        assert recency_score(date(2024, 6, 1), date(2024, 6, 1), 365.0) == 1.0

    def test_one_half_life_scores_half(self):
        assert recency_score(date(2023, 6, 2), date(2024, 6, 1), 365.0) == pytest.approx(
            0.5, abs=0.01
        )

    def test_older_always_scores_lower(self):
        anchor = date(2024, 6, 1)
        scores = [
            recency_score(observed, anchor, 365.0)
            for observed in [date(2024, 5, 1), date(2023, 5, 1), date(2020, 5, 1)]
        ]
        assert scores == sorted(scores, reverse=True)


class TestTimeTravel:
    """The Phase 2 Definition of Done, as an executable test."""

    def test_two_as_of_dates_give_different_correctly_dated_results(self):
        retriever = TemporalRetriever(store_with(VERSION_CHUNKS), EMBEDDER)
        question = "what is the latest stable version of python"

        past = retriever.retrieve(question, as_of=date(2024, 6, 1), top_k=1)
        assert past[0].chunk.chunk_id == "python-whatsnew-3.12:0000"
        assert past[0].chunk.observed_at == date(2023, 10, 2)
        assert "3.12" in past[0].chunk.text

        later = retriever.retrieve(question, as_of=date(2025, 6, 1), top_k=1)
        assert later[0].chunk.chunk_id == "python-whatsnew-3.13:0000"
        assert later[0].chunk.observed_at == date(2024, 10, 7)
        assert "3.13" in later[0].chunk.text

    def test_future_chunks_never_reach_ranking(self):
        retriever = TemporalRetriever(store_with(VERSION_CHUNKS), EMBEDDER)
        results = retriever.retrieve("latest python", as_of=date(2024, 6, 1), top_k=10)
        assert all(item.chunk.observed_at <= date(2024, 6, 1) for item in results)

    def test_no_as_of_anchors_recency_to_newest_hit(self):
        retriever = TemporalRetriever(store_with(VERSION_CHUNKS), EMBEDDER)
        results = retriever.retrieve("latest stable python", top_k=2)
        newest = next(r for r in results if r.chunk.chunk_id == "python-whatsnew-3.13:0000")
        assert newest.recency == 1.0


class TestRecencyPrior:
    def test_zero_weight_is_pure_similarity(self):
        retriever = TemporalRetriever(store_with(VERSION_CHUNKS), EMBEDDER, recency_weight=0.0)
        results = retriever.retrieve("latest stable python", as_of=date(2025, 6, 1), top_k=2)
        assert [item.score for item in results] == [item.similarity for item in results]

    def test_recency_breaks_similarity_ties_toward_fresh(self):
        retriever = TemporalRetriever(store_with(VERSION_CHUNKS), EMBEDDER)
        results = retriever.retrieve(
            "what is the latest stable version of python", as_of=date(2025, 6, 1), top_k=2
        )
        assert results[0].chunk.chunk_id == "python-whatsnew-3.13:0000"
        assert results[0].similarity == pytest.approx(results[1].similarity, abs=0.02)
        assert results[0].recency > results[1].recency

    def test_scoring_breakdown_is_preserved(self):
        retriever = TemporalRetriever(store_with(VERSION_CHUNKS), EMBEDDER, recency_weight=0.3)
        item = retriever.retrieve("latest python", as_of=date(2025, 6, 1), top_k=1)[0]
        assert item.score == pytest.approx(0.7 * item.similarity + 0.3 * item.recency)

    def test_invalid_knobs_are_rejected(self):
        store = store_with(VERSION_CHUNKS)
        with pytest.raises(ValueError, match="recency_weight"):
            TemporalRetriever(store, EMBEDDER, recency_weight=1.5)
        with pytest.raises(ValueError, match="half_life_days"):
            TemporalRetriever(store, EMBEDDER, half_life_days=0)

    def test_empty_store_returns_empty(self):
        retriever = TemporalRetriever(InMemoryVectorStore(), EMBEDDER)
        assert retriever.retrieve("anything", as_of=date(2024, 6, 1)) == []

"""P4: the A/B harness — tokens down, retention flat, scorecard renders."""

from datetime import date
from pathlib import Path

from groundproof.compress import QueryAwarePruner
from groundproof.evals import ABCase, load_ab_cases, render_ab_scorecard, run_ab
from groundproof.ingest import Chunk
from groundproof.retrieval import InMemoryVectorStore, MockEmbeddingClient, TemporalRetriever

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


PADDING = " ".join(f"Unrelated filler detail number {i} about build tooling." for i in range(30))

CHUNKS = [
    make_chunk(
        "python-whatsnew-3.12:0000",
        f"Removed\n\nThe distutils package was removed in Python 3.12. {PADDING}",
        date(2023, 10, 2),
    ),
    make_chunk(
        "python-whatsnew-3.9:0000",
        f"Added\n\nThe zoneinfo module was added in Python 3.9. {PADDING}",
        date(2020, 10, 5),
    ),
]

CASES = [
    ABCase(
        case_id="distutils-removed",
        question="is the distutils package removed",
        required_phrases=["distutils"],
    ),
    ABCase(
        case_id="zoneinfo-added",
        question="when was the zoneinfo module added",
        required_phrases=["zoneinfo"],
    ),
]


def make_retriever() -> TemporalRetriever:
    store = InMemoryVectorStore()
    store.index(CHUNKS, EMBEDDER.embed_texts([chunk.text for chunk in CHUNKS]))
    return TemporalRetriever(store, EMBEDDER)


class TestRunAB:
    def test_tokens_down_retention_flat(self):
        report = run_ab(
            CASES,
            retriever=make_retriever(),
            embedder=EMBEDDER,
            pruner=QueryAwarePruner(EMBEDDER, token_budget=60),
        )
        assert len(report.results) == 2
        for result in report.results:
            assert result.tokens_on < result.tokens_off
            assert result.retention_off == 1.0
            assert result.retention_on == 1.0
        assert report.mean_savings > 0.4
        assert report.retention_intact

    def test_scorecard_renders_the_diff(self):
        report = run_ab(
            CASES,
            retriever=make_retriever(),
            embedder=EMBEDDER,
            pruner=QueryAwarePruner(EMBEDDER, token_budget=60),
        )
        scorecard = render_ab_scorecard(report)
        assert "distutils-removed" in scorecard
        assert "zoneinfo-added" in scorecard
        assert "mean savings" in scorecard
        assert "intact" in scorecard


class TestGoldenDataset:
    def test_committed_ab_cases_load_and_validate(self):
        cases = load_ab_cases(Path("datasets/ab_compression.jsonl"))
        assert len(cases) >= 3
        assert all(case.required_phrases for case in cases)
        assert {case.case_id for case in cases} >= {"distutils-removed", "zoneinfo-added"}

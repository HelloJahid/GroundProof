"""P1 Step 5: one contract, two stores — the same tests run against both.

The fixture parametrizes every contract test over InMemoryVectorStore and
ChromaVectorStore (in-process EphemeralClient: no keys, no network, telemetry
off), so the two implementations cannot drift apart unnoticed.
"""

import uuid
from datetime import date

import pytest

from groundproof.ingest import Chunk, Document
from groundproof.retrieval import (
    InMemoryVectorStore,
    MockEmbeddingClient,
    ScoredChunk,
    VectorStore,
    index_documents,
)

EMBEDDER = MockEmbeddingClient()


def make_chunk(chunk_id: str, text: str, **overrides) -> Chunk:
    defaults = dict(
        chunk_id=chunk_id,
        doc_id="python-whatsnew-3.12",
        source="https://example.test/whatsnew/3.12.rst.txt",
        text=text,
        position=int(chunk_id.rsplit(":", 1)[-1]),
        observed_at=date(2023, 10, 2),
    )
    return Chunk(**{**defaults, **overrides})


def index_texts(store, chunks: list[Chunk]) -> None:
    store.index(chunks, EMBEDDER.embed_texts([chunk.text for chunk in chunks]))


@pytest.fixture(params=["memory", "chroma"])
def store(request):
    if request.param == "memory":
        return InMemoryVectorStore()
    from groundproof.retrieval.chroma_store import ChromaVectorStore

    # Chroma's EphemeralClient is a process-wide singleton per settings, so a
    # fixed collection name would leak state between tests — name each one uniquely.
    return ChromaVectorStore.ephemeral(f"contract-{uuid.uuid4().hex[:12]}")


class TestContract:
    def test_satisfies_the_protocol(self, store):
        assert isinstance(store, VectorStore)

    def test_empty_store_counts_zero_and_returns_no_hits(self, store):
        assert store.count() == 0
        assert store.query(EMBEDDER.embed_texts(["anything"])[0]) == []

    def test_index_then_count(self, store):
        index_texts(
            store,
            [
                make_chunk("python-whatsnew-3.12:0000", "distutils was removed."),
                make_chunk("python-whatsnew-3.12:0001", "zoneinfo was added."),
            ],
        )
        assert store.count() == 2

    def test_query_ranks_by_relevance(self, store):
        index_texts(
            store,
            [
                make_chunk("python-whatsnew-3.12:0000", "The distutils package was removed."),
                make_chunk("python-whatsnew-3.12:0001", "A banana smoothie needs ripe bananas."),
            ],
        )
        hits = store.query(EMBEDDER.embed_texts(["is distutils still available"])[0], top_k=2)
        assert [type(hit) for hit in hits] == [ScoredChunk, ScoredChunk]
        assert hits[0].chunk.chunk_id == "python-whatsnew-3.12:0000"
        assert hits[0].score > hits[1].score

    def test_top_k_limits_results_and_never_overreaches(self, store):
        index_texts(
            store,
            [
                make_chunk("python-whatsnew-3.12:0000", "alpha text"),
                make_chunk("python-whatsnew-3.12:0001", "beta text"),
            ],
        )
        query_vector = EMBEDDER.embed_texts(["alpha"])[0]
        assert len(store.query(query_vector, top_k=1)) == 1
        assert len(store.query(query_vector, top_k=10)) == 2

    def test_reindexing_same_chunk_id_overwrites(self, store):
        index_texts(store, [make_chunk("python-whatsnew-3.12:0000", "old text here")])
        index_texts(store, [make_chunk("python-whatsnew-3.12:0000", "new text here")])
        assert store.count() == 1
        hits = store.query(EMBEDDER.embed_texts(["new text here"])[0], top_k=1)
        assert hits[0].chunk.text == "new text here"

    def test_as_of_hides_chunks_from_the_future(self, store):
        index_texts(
            store,
            [
                make_chunk(
                    "python-whatsnew-3.12:0000", "same text", observed_at=date(2023, 10, 2)
                ),
                make_chunk(
                    "python-whatsnew-3.13:0000", "same text", observed_at=date(2024, 10, 7)
                ),
            ],
        )
        hits = store.query(EMBEDDER.embed_texts(["same text"])[0], top_k=5, as_of=date(2024, 6, 1))
        assert [hit.chunk.chunk_id for hit in hits] == ["python-whatsnew-3.12:0000"]

    def test_as_of_boundary_is_inclusive(self, store):
        index_texts(
            store,
            [make_chunk("python-whatsnew-3.12:0000", "boundary", observed_at=date(2023, 10, 2))],
        )
        hits = store.query(
            EMBEDDER.embed_texts(["boundary"])[0], top_k=5, as_of=date(2023, 10, 2)
        )
        assert len(hits) == 1

    def test_no_as_of_sees_everything(self, store):
        index_texts(
            store,
            [
                make_chunk("python-whatsnew-3.12:0000", "one", observed_at=date(2023, 10, 2)),
                make_chunk("python-whatsnew-3.13:0000", "two", observed_at=date(2024, 10, 7)),
            ],
        )
        assert len(store.query(EMBEDDER.embed_texts(["one two"])[0], top_k=5)) == 2

    def test_temporal_metadata_round_trips(self, store):
        index_texts(
            store,
            [
                make_chunk(
                    "python-whatsnew-3.12:0000",
                    "distutils was removed.",
                    valid_from=date(2023, 10, 2),
                    valid_to=date(2028, 10, 31),
                ),
                make_chunk("python-whatsnew-3.12:0001", "zoneinfo was added."),
            ],
        )
        hits = store.query(EMBEDDER.embed_texts(["distutils removed"])[0], top_k=2)
        windowed = next(h.chunk for h in hits if h.chunk.chunk_id.endswith("0000"))
        bare = next(h.chunk for h in hits if h.chunk.chunk_id.endswith("0001"))
        assert windowed.observed_at == date(2023, 10, 2)
        assert windowed.valid_from == date(2023, 10, 2)
        assert windowed.valid_to == date(2028, 10, 31)
        assert bare.valid_from is None
        assert bare.valid_to is None


class TestInMemoryDeterminism:
    def test_equal_scores_tie_break_by_chunk_id(self):
        store = InMemoryVectorStore()
        index_texts(
            store,
            [
                make_chunk("python-whatsnew-3.12:0001", "identical text"),
                make_chunk("python-whatsnew-3.12:0000", "identical text"),
            ],
        )
        hits = store.query(EMBEDDER.embed_texts(["identical text"])[0], top_k=2)
        assert [h.chunk.chunk_id for h in hits] == [
            "python-whatsnew-3.12:0000",
            "python-whatsnew-3.12:0001",
        ]


class TestEndToEndIngestion:
    def test_documents_to_indexed_store(self):
        documents = [
            Document(
                doc_id="python-whatsnew-3.12",
                source="https://example.test/whatsnew/3.12.rst.txt",
                title="What's New In Python 3.12",
                text="Removed\n=======\n\nThe distutils package was removed.\n",
                observed_at=date(2023, 10, 2),
            ),
            Document(
                doc_id="python-whatsnew-3.9",
                source="https://example.test/whatsnew/3.9.rst.txt",
                title="What's New In Python 3.9",
                text="Added\n=====\n\nThe zoneinfo module was added.\n",
                observed_at=date(2020, 10, 5),
            ),
        ]
        store = InMemoryVectorStore()
        indexed = index_documents(documents, EMBEDDER, store, batch_size=1)
        assert indexed == store.count() == 2
        hits = store.query(EMBEDDER.embed_texts(["is distutils available"])[0], top_k=1)
        assert hits[0].chunk.doc_id == "python-whatsnew-3.12"
        assert hits[0].chunk.observed_at == date(2023, 10, 2)

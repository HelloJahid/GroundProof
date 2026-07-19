"""P4: the query-aware pruner — budget respected, attribution preserved."""

from datetime import date

import pytest

from groundproof.compress import (
    QueryAwarePruner,
    estimate_tokens,
    split_chunk,
)
from groundproof.ingest import Chunk
from groundproof.retrieval import MockEmbeddingClient, RankedChunk

EMBEDDER = MockEmbeddingClient()


def make_ranked(chunk_id: str, text: str, observed_at: date = date(2023, 10, 2)) -> RankedChunk:
    doc_id = chunk_id.rsplit(":", 1)[0]
    chunk = Chunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        source=f"https://example.test/{doc_id}.rst.txt",
        text=text,
        position=0,
        observed_at=observed_at,
    )
    return RankedChunk(chunk=chunk, similarity=0.5, recency=1.0, score=0.5)


MIXED = make_ranked(
    "python-whatsnew-3.12:0000",
    "Removed\n\nThe distutils package was removed in Python 3.12. "
    "Meanwhile, a banana smoothie needs two ripe bananas and a blender. "
    "The venerable ensurepip module remains available for bootstrapping pip.",
)


class TestTokenEstimate:
    def test_scales_with_length_and_floors_at_one(self):
        assert estimate_tokens("") == 0
        assert estimate_tokens("hi") == 1
        assert estimate_tokens("x" * 400) == 100


class TestSplitting:
    def test_sentences_carry_full_attribution(self):
        sentences = split_chunk(MIXED.chunk)
        assert len(sentences) == 4  # title + three sentences
        for sentence in sentences:
            assert sentence.chunk_id == "python-whatsnew-3.12:0000"
            assert sentence.doc_id == "python-whatsnew-3.12"
            assert sentence.observed_at == date(2023, 10, 2)
        assert [s.order for s in sentences] == [0, 1, 2, 3]

    def test_hard_wrapped_lines_are_unwrapped(self):
        chunk = make_ranked("d:0000", "One sentence\nwrapped across\nlines.").chunk
        assert [s.text for s in split_chunk(chunk)] == ["One sentence wrapped across lines."]


class TestCompression:
    def test_budget_is_respected(self):
        pruner = QueryAwarePruner(EMBEDDER, token_budget=20)
        compressed = pruner.compress("is distutils removed", [MIXED])
        assert compressed.tokens_after <= 20
        assert compressed.tokens_before > compressed.tokens_after

    def test_query_relevant_sentence_survives_irrelevant_dropped(self):
        # 30 tokens: room for the attribution header plus the relevant sentence,
        # but not for the irrelevant one.
        pruner = QueryAwarePruner(EMBEDDER, token_budget=30)
        compressed = pruner.compress("is the distutils package removed", [MIXED])
        assert "distutils" in compressed.text
        assert "banana" not in compressed.text

    def test_attribution_header_survives_compression(self):
        pruner = QueryAwarePruner(EMBEDDER, token_budget=50)
        compressed = pruner.compress("is distutils removed", [MIXED])
        assert "(2023-10-02, python-whatsnew-3.12):" in compressed.text

    def test_deterministic(self):
        pruner = QueryAwarePruner(EMBEDDER, token_budget=30)
        assert pruner.compress("distutils", [MIXED]) == pruner.compress("distutils", [MIXED])

    def test_empty_evidence_compresses_to_empty(self):
        compressed = QueryAwarePruner(EMBEDDER).compress("anything", [])
        assert compressed.text == ""
        assert compressed.tokens_after == 0
        assert compressed.savings == 0.0

    def test_savings_property(self):
        pruner = QueryAwarePruner(EMBEDDER, token_budget=20)
        compressed = pruner.compress("is distutils removed", [MIXED])
        assert compressed.savings == pytest.approx(
            1 - compressed.tokens_after / compressed.tokens_before
        )

    def test_invalid_budget_rejected(self):
        with pytest.raises(ValueError, match="token_budget"):
            QueryAwarePruner(EMBEDDER, token_budget=0)

"""P1 Step 1: the temporal data model validates at the boundary and is immutable after."""

from datetime import date

import pytest
from pydantic import ValidationError

from groundproof.ingest import Chunk, Document


def make_document(**overrides) -> Document:
    defaults = dict(
        doc_id="py-3.12.0",
        source="https://docs.python.org/3/whatsnew/3.12.html",
        title="What's New In Python 3.12",
        text="Python 3.12.0 was released on October 2, 2023.",
        observed_at=date(2023, 10, 2),
    )
    return Document(**{**defaults, **overrides})


def make_chunk(**overrides) -> Chunk:
    defaults = dict(
        chunk_id="py-3.12.0:0000",
        doc_id="py-3.12.0",
        source="https://docs.python.org/3/whatsnew/3.12.html",
        text="Python 3.12.0 was released on October 2, 2023.",
        position=0,
        observed_at=date(2023, 10, 2),
    )
    return Chunk(**{**defaults, **overrides})


class TestDocument:
    def test_constructs_with_temporal_anchor(self):
        doc = make_document()
        assert doc.observed_at == date(2023, 10, 2)
        assert doc.metadata == {}

    def test_is_frozen(self):
        doc = make_document()
        with pytest.raises(ValidationError):
            doc.title = "tampered"

    def test_rejects_empty_text(self):
        with pytest.raises(ValidationError):
            make_document(text="")

    def test_observed_at_is_required(self):
        with pytest.raises(ValidationError):
            Document(
                doc_id="d", source="s", title="t", text="x"
            )


class TestChunk:
    def test_constructs_without_validity_window(self):
        chunk = make_chunk()
        assert chunk.valid_from is None
        assert chunk.valid_to is None

    def test_accepts_chronological_validity_window(self):
        chunk = make_chunk(valid_from=date(2023, 10, 2), valid_to=date(2024, 10, 7))
        assert chunk.valid_from < chunk.valid_to

    def test_rejects_inverted_validity_window(self):
        with pytest.raises(ValidationError, match="valid_from"):
            make_chunk(valid_from=date(2024, 10, 7), valid_to=date(2023, 10, 2))

    def test_accepts_point_validity_window(self):
        chunk = make_chunk(valid_from=date(2023, 10, 2), valid_to=date(2023, 10, 2))
        assert chunk.valid_from == chunk.valid_to

    def test_is_frozen(self):
        chunk = make_chunk()
        with pytest.raises(ValidationError):
            chunk.text = "tampered"

    def test_rejects_negative_position(self):
        with pytest.raises(ValidationError):
            make_chunk(position=-1)

    def test_json_round_trip_preserves_dates(self):
        original = make_chunk(valid_from=date(2023, 10, 2), valid_to=date(2024, 10, 7))
        restored = Chunk.model_validate_json(original.model_dump_json())
        assert restored == original
        assert restored.observed_at == date(2023, 10, 2)

"""P2: the supersedence resolver — rules pick current truth, older kept as history."""

from datetime import date

from groundproof.ingest import Chunk
from groundproof.retrieval import RankedChunk, resolve_supersedence, topic_key


def make_item(chunk_id: str, text: str, observed_at: date, score: float = 0.5) -> RankedChunk:
    doc_id = chunk_id.rsplit(":", 1)[0]
    chunk = Chunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        source=f"https://example.test/{doc_id}.rst.txt",
        text=text,
        position=0,
        observed_at=observed_at,
    )
    return RankedChunk(chunk=chunk, similarity=score, recency=1.0, score=score)


class TestTopicKey:
    def test_topic_is_the_normalised_first_line(self):
        item = make_item("d:0000", "  Latest   Version  \n\nPython 3.12 is out.", date(2023, 10, 2))
        assert topic_key(item.chunk) == "latest version"


class TestResolution:
    def test_later_fact_supersedes_earlier_on_same_topic(self):
        older = make_item(
            "python-whatsnew-3.12:0000", "Latest Version\n\nPython 3.12.", date(2023, 10, 2)
        )
        newer = make_item(
            "python-whatsnew-3.13:0000", "Latest Version\n\nPython 3.13.", date(2024, 10, 7)
        )
        resolved = resolve_supersedence([older, newer])

        assert [item.chunk.chunk_id for item in resolved.current] == ["python-whatsnew-3.13:0000"]
        assert len(resolved.history) == 1
        record = resolved.history[0]
        assert record.item.chunk.chunk_id == "python-whatsnew-3.12:0000"
        assert record.superseded_by == "python-whatsnew-3.13:0000"
        assert record.superseded_on == date(2024, 10, 7)

    def test_different_topics_never_supersede(self):
        removed = make_item("a:0000", "Removed\n\ndistutils is gone.", date(2023, 10, 2))
        added = make_item("b:0000", "Added\n\nzoneinfo arrived.", date(2024, 10, 7))
        resolved = resolve_supersedence([removed, added])
        assert len(resolved.current) == 2
        assert resolved.history == []

    def test_equal_dates_never_supersede(self):
        first = make_item("a:0000", "Latest Version\n\nPython 3.12.", date(2023, 10, 2))
        second = make_item("b:0000", "Latest Version\n\nAlso 3.12 news.", date(2023, 10, 2))
        resolved = resolve_supersedence([first, second])
        assert len(resolved.current) == 2
        assert resolved.history == []

    def test_chain_of_three_reads_like_a_timeline(self):
        v12 = make_item("v3.12:0000", "Latest Version\n\n3.12.", date(2023, 10, 2))
        v13 = make_item("v3.13:0000", "Latest Version\n\n3.13.", date(2024, 10, 7))
        v14 = make_item("v3.14:0000", "Latest Version\n\n3.14.", date(2025, 10, 7))
        resolved = resolve_supersedence([v12, v13, v14])

        assert [item.chunk.chunk_id for item in resolved.current] == ["v3.14:0000"]
        by_id = {record.item.chunk.chunk_id: record for record in resolved.history}
        assert by_id["v3.12:0000"].superseded_by == "v3.13:0000"
        assert by_id["v3.12:0000"].superseded_on == date(2024, 10, 7)
        assert by_id["v3.13:0000"].superseded_by == "v3.14:0000"
        assert by_id["v3.13:0000"].superseded_on == date(2025, 10, 7)

    def test_input_ranking_order_is_preserved_in_current(self):
        best = make_item("a:0000", "Removed\n\ndistutils.", date(2023, 10, 2), score=0.9)
        worse = make_item("b:0000", "Added\n\nzoneinfo.", date(2024, 10, 7), score=0.4)
        resolved = resolve_supersedence([best, worse])
        assert [item.chunk.chunk_id for item in resolved.current] == ["a:0000", "b:0000"]

    def test_empty_input_resolves_to_empty(self):
        resolved = resolve_supersedence([])
        assert resolved.current == []
        assert resolved.history == []

"""P1 Step 3: the chunker splits on reST structure and never invents time."""

from datetime import date

from groundproof.ingest import Document
from groundproof.ingest.chunker import chunk_corpus, chunk_document

REST_TEXT = """\
*************************
What's New In Python 9.9
*************************

This article explains the new features in Python 9.9.

Summary
=======

Python 9.9 is faster.

Deprecated
==========

The ``oldmod`` module is deprecated.

It will be removed in Python 10.0.

Removed
=======

``distutils`` is gone.
"""


def make_document(text: str = REST_TEXT) -> Document:
    return Document(
        doc_id="python-whatsnew-9.9",
        source="https://example.test/whatsnew/9.9.rst.txt",
        title="What's New In Python 9.9",
        text=text,
        observed_at=date(2030, 10, 1),
    )


class TestSectionSplitting:
    def test_splits_on_underlined_titles_and_prepends_title(self):
        chunks = chunk_document(make_document())
        texts = [c.text for c in chunks]
        assert any(t.startswith("Summary\n\n") for t in texts)
        assert any(t.startswith("Deprecated\n\n") for t in texts)
        assert any(t.startswith("Removed\n\n") and "distutils" in t for t in texts)

    def test_decorative_overline_does_not_leak_into_chunks(self):
        chunks = chunk_document(make_document())
        assert not any("*****" in c.text for c in chunks)

    def test_text_without_headers_still_chunks(self):
        chunks = chunk_document(make_document(text="Just one plain paragraph."))
        assert len(chunks) == 1
        assert chunks[0].text == "Just one plain paragraph."


class TestTemporalInheritance:
    def test_every_chunk_inherits_document_date_and_provenance(self):
        document = make_document()
        for chunk in chunk_document(document):
            assert chunk.observed_at == document.observed_at
            assert chunk.doc_id == document.doc_id
            assert chunk.source == document.source


class TestDeterminism:
    def test_positions_are_sequential_and_ids_derived(self):
        chunks = chunk_document(make_document())
        assert [c.position for c in chunks] == list(range(len(chunks)))
        assert [c.chunk_id for c in chunks] == [
            f"python-whatsnew-9.9:{i:04d}" for i in range(len(chunks))
        ]

    def test_same_input_gives_identical_chunks(self):
        assert chunk_document(make_document()) == chunk_document(make_document())


class TestSizeBudget:
    def test_oversized_section_splits_at_paragraphs_within_budget(self):
        paragraphs = "\n\n".join(f"Paragraph {i}: " + "x" * 80 for i in range(10))
        text = f"Big Section\n===========\n\n{paragraphs}\n"
        chunks = chunk_document(make_document(text=text), max_chars=300)
        assert len(chunks) > 1
        for chunk in chunks:
            body = chunk.text.removeprefix("Big Section\n\n")
            assert len(body) <= 300
        rejoined = "\n\n".join(c.text.removeprefix("Big Section\n\n") for c in chunks)
        assert rejoined == paragraphs

    def test_single_paragraph_over_budget_is_kept_whole(self):
        huge = "y" * 500
        text = f"Big Section\n===========\n\n{huge}\n"
        chunks = chunk_document(make_document(text=text), max_chars=300)
        assert len(chunks) == 1
        assert huge in chunks[0].text


class TestCorpus:
    def test_chunk_corpus_preserves_document_order(self):
        older = make_document()
        newer = Document(
            doc_id="python-whatsnew-10.0",
            source="https://example.test/whatsnew/10.0.rst.txt",
            title="What's New In Python 10.0",
            text="Summary\n=======\n\nPython 10.0 is out.\n",
            observed_at=date(2031, 10, 1),
        )
        chunks = chunk_corpus([older, newer])
        doc_ids = [c.doc_id for c in chunks]
        assert doc_ids == sorted(doc_ids, key=doc_ids.index)
        assert doc_ids[0] == "python-whatsnew-9.9"
        assert doc_ids[-1] == "python-whatsnew-10.0"

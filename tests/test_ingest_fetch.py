"""P1 Step 2: fetcher/parser tested fully offline via httpx.MockTransport."""

from datetime import date

import httpx
import pytest

from groundproof.errors import FetchFailure, GroundProofError
from groundproof.ingest import Document, load_corpus, save_corpus
from groundproof.ingest.fetch import (
    RELEASE_INDEX_URL,
    fetch_corpus,
    fetch_release_cycles,
    fetch_whatsnew_text,
)

RELEASE_INDEX = [
    {"cycle": "3.13", "releaseDate": "2024-10-07", "latest": "3.13.1", "eol": "2029-10-31"},
    {"cycle": "3.12", "releaseDate": "2023-10-02", "latest": "3.12.8", "eol": "2028-10-31"},
    {"cycle": "3.7", "releaseDate": "2018-06-27", "latest": "3.7.17", "eol": "2023-06-27"},
    {"cycle": "2.7", "releaseDate": "2010-07-03", "latest": "2.7.18", "eol": "2020-01-01"},
]


def happy_handler(request: httpx.Request) -> httpx.Response:
    if str(request.url) == RELEASE_INDEX_URL:
        return httpx.Response(200, json=RELEASE_INDEX)
    if "/whatsnew/" in request.url.path:
        cycle = request.url.path.rsplit("/", 1)[-1].removesuffix(".rst.txt")
        return httpx.Response(200, text=f"What's New In Python {cycle}\n\nLots of changes.")
    return httpx.Response(404)


def client_with(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


class TestFetchReleaseCycles:
    def test_keeps_only_supported_cycles_sorted_oldest_first(self):
        with client_with(happy_handler) as client:
            cycles = fetch_release_cycles(client)
        assert [c.cycle for c in cycles] == ["3.12", "3.13"]
        assert cycles[0].release_date == date(2023, 10, 2)

    def test_http_error_status_raises_fetch_failure(self):
        with client_with(lambda request: httpx.Response(500)) as client:
            with pytest.raises(FetchFailure, match="HTTP 500"):
                fetch_release_cycles(client)

    def test_network_error_raises_fetch_failure(self):
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("boom", request=request)

        with client_with(handler) as client:
            with pytest.raises(FetchFailure, match="failed"):
                fetch_release_cycles(client)

    def test_malformed_payload_raises_fetch_failure(self):
        with client_with(lambda request: httpx.Response(200, text="not json")) as client:
            with pytest.raises(FetchFailure, match="malformed"):
                fetch_release_cycles(client)

    def test_fetch_failure_is_a_groundproof_error(self):
        assert issubclass(FetchFailure, GroundProofError)


class TestFetchWhatsnew:
    def test_returns_page_text(self):
        with client_with(happy_handler) as client:
            text = fetch_whatsnew_text(client, "3.12")
        assert "What's New In Python 3.12" in text

    def test_missing_page_raises_fetch_failure(self):
        with client_with(lambda request: httpx.Response(404)) as client:
            with pytest.raises(FetchFailure, match="HTTP 404"):
                fetch_whatsnew_text(client, "9.9")


class TestFetchCorpus:
    def test_builds_one_dated_document_per_cycle(self):
        with client_with(happy_handler) as client:
            documents = fetch_corpus(client)

        assert [d.doc_id for d in documents] == [
            "python-whatsnew-3.12",
            "python-whatsnew-3.13",
        ]
        doc = documents[0]
        assert doc.observed_at == date(2023, 10, 2)
        assert doc.metadata == {"cycle": "3.12"}
        assert "What's New In Python 3.12" in doc.text
        assert doc.source.endswith("whatsnew/3.12.rst.txt")


class TestCorpusIO:
    def test_save_then_load_round_trips_sorted_by_date(self, tmp_path):
        with client_with(happy_handler) as client:
            documents = fetch_corpus(client)

        paths = save_corpus(documents, tmp_path / "corpus")
        assert [p.name for p in paths] == [
            "python-whatsnew-3.12.json",
            "python-whatsnew-3.13.json",
        ]

        loaded = load_corpus(tmp_path / "corpus")
        assert loaded == sorted(documents, key=lambda d: (d.observed_at, d.doc_id))
        assert all(isinstance(d, Document) for d in loaded)

    def test_load_ignores_non_json_files(self, tmp_path):
        corpus_dir = tmp_path / "corpus"
        corpus_dir.mkdir()
        (corpus_dir / "README.txt").write_text("not a document", encoding="utf-8")
        assert load_corpus(corpus_dir) == []

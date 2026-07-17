"""Fetch CPython release metadata and "What's New" texts into dated Documents.

Two live sources:

- endoflife.date's Python API — one JSON list of every release cycle with its
  release date. That date becomes each document's ``observed_at``: temporal
  metadata is *fetched from the record*, never guessed.
- docs.python.org's Sphinx source files (``_sources/whatsnew/3.X.rst.txt``) —
  the raw reStructuredText behind each "What's New in Python 3.X" page. Clean
  prose with no HTML parsing required.

The HTTP client is injected (any ``httpx.Client``), so the test suite runs
offline via ``httpx.MockTransport``; the one live network run lives in
``python -m groundproof.ingest``.
"""

from datetime import date

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from groundproof.errors import FetchFailure
from groundproof.ingest.models import Document

RELEASE_INDEX_URL = "https://endoflife.date/api/python.json"
WHATSNEW_URL_TEMPLATE = "https://docs.python.org/3/_sources/whatsnew/{cycle}.rst.txt"

# Document-level scope discipline: we ingest the cycles with a "What's New" page
# in the current docs build and a clean release date on record.
_MIN_SUPPORTED = (3, 8)


class ReleaseCycle(BaseModel):
    """One Python release line as reported by the release index."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    cycle: str = Field(min_length=1)
    release_date: date = Field(alias="releaseDate")


def _get(client: httpx.Client, url: str) -> httpx.Response:
    try:
        response = client.get(url)
    except httpx.HTTPError as exc:
        raise FetchFailure(f"GET {url} failed: {exc}") from exc
    if response.status_code != 200:
        raise FetchFailure(f"GET {url} returned HTTP {response.status_code}")
    return response


def _is_supported(cycle: str) -> bool:
    try:
        major, minor = (int(part) for part in cycle.split("."))
    except ValueError:
        return False
    return major == 3 and (major, minor) >= _MIN_SUPPORTED


def fetch_release_cycles(client: httpx.Client) -> list[ReleaseCycle]:
    """Fetch every supported 3.x release cycle, oldest release first."""
    response = _get(client, RELEASE_INDEX_URL)
    try:
        entries = response.json()
        cycles = [ReleaseCycle.model_validate(entry) for entry in entries]
    except (ValueError, ValidationError) as exc:
        raise FetchFailure(f"release index at {RELEASE_INDEX_URL} is malformed: {exc}") from exc
    supported = [cycle for cycle in cycles if _is_supported(cycle.cycle)]
    return sorted(supported, key=lambda cycle: cycle.release_date)


def fetch_whatsnew_text(client: httpx.Client, cycle: str) -> str:
    """Fetch the raw reStructuredText of one "What's New in Python <cycle>" page."""
    return _get(client, WHATSNEW_URL_TEMPLATE.format(cycle=cycle)).text


def build_document(cycle: ReleaseCycle, text: str) -> Document:
    """Assemble one validated, dated Document from a release cycle and its prose."""
    return Document(
        doc_id=f"python-whatsnew-{cycle.cycle}",
        source=WHATSNEW_URL_TEMPLATE.format(cycle=cycle.cycle),
        title=f"What's New In Python {cycle.cycle}",
        text=text,
        observed_at=cycle.release_date,
        metadata={"cycle": cycle.cycle},
    )


def fetch_corpus(client: httpx.Client) -> list[Document]:
    """Fetch the full changelog corpus: one dated Document per supported release cycle."""
    return [
        build_document(cycle, fetch_whatsnew_text(client, cycle.cycle))
        for cycle in fetch_release_cycles(client)
    ]

"""The temporal data model: every piece of text in GroundProof is born carrying a date.

``Document`` is what the fetcher/parser produces (one changelog page, one release note);
``Chunk`` is the retrieval unit the chunker cuts it into. Both are frozen Pydantic v2
models — data at the pipeline boundary is validated once, then immutable. Temporal
metadata is document-level first (``observed_at``); fact-level validity windows
(``valid_from``/``valid_to``) are optional and only set when genuinely derivable.
"""

from datetime import date

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Document(BaseModel):
    """One dated source document, as parsed from the corpus.

    ``observed_at`` is the document's own date (e.g. a release date) — the moment the
    world could first have read this text. It is the anchor for all as-of filtering.
    """

    model_config = ConfigDict(frozen=True)

    doc_id: str = Field(min_length=1)
    source: str = Field(min_length=1, description="URL or path this document came from")
    title: str = Field(min_length=1)
    text: str = Field(min_length=1)
    observed_at: date
    metadata: dict[str, str] = Field(default_factory=dict)


class Chunk(BaseModel):
    """One retrieval unit cut from a document, inheriting its temporal anchor.

    ``observed_at`` is copied from the parent document, never invented per-chunk.
    ``valid_from``/``valid_to`` describe the window a *fact* holds true (e.g. a Python
    version is "latest" from its release until the next one) — optional, filled only
    when derivable by rule, and always a superset question for the resolver, not the LLM.
    """

    model_config = ConfigDict(frozen=True)

    chunk_id: str = Field(min_length=1)
    doc_id: str = Field(min_length=1)
    source: str = Field(min_length=1)
    text: str = Field(min_length=1)
    position: int = Field(ge=0, description="0-based order of this chunk within its document")
    observed_at: date
    valid_from: date | None = None
    valid_to: date | None = None

    @model_validator(mode="after")
    def _validity_window_is_chronological(self) -> "Chunk":
        if (
            self.valid_from is not None
            and self.valid_to is not None
            and self.valid_from > self.valid_to
        ):
            raise ValueError(
                f"valid_from ({self.valid_from}) must not be after valid_to ({self.valid_to})"
            )
        return self

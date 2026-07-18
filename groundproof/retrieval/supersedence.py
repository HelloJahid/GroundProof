"""Deterministic supersedence: when retrieved facts conflict, rules pick the truth.

Freshness is a gate check, not a judgment call — the LLM is never asked to
track which fact is current. The rule is small and total:

- Topic = a chunk's normalised section title (its first line — the chunker
  put it there in P1 precisely so provenance like this stays attached).
- Within one topic, a strictly later ``observed_at`` supersedes an earlier one.
- Superseded chunks are NOT discarded: each becomes a history record naming
  what replaced it and when — "this was true until X" is an answerable
  question, and dated citations need exactly that.
- Equal dates never supersede each other: with no chronology there is no rule,
  and guessing is the one thing this module must never do.
"""

from datetime import date

from pydantic import BaseModel, ConfigDict

from groundproof.ingest.models import Chunk
from groundproof.retrieval.temporal import RankedChunk


class SupersededRecord(BaseModel):
    """An older fact kept as history: what replaced it, and from when."""

    model_config = ConfigDict(frozen=True)

    item: RankedChunk
    superseded_by: str
    superseded_on: date


class ResolvedEvidence(BaseModel):
    """The resolver's verdict: current truth up front, history preserved behind it."""

    model_config = ConfigDict(frozen=True)

    current: list[RankedChunk]
    history: list[SupersededRecord]


def topic_key(chunk: Chunk) -> str:
    """A chunk's topic: its first line (the section title), whitespace-normalised."""
    first_line = chunk.text.splitlines()[0] if chunk.text else ""
    return " ".join(first_line.lower().split())


def resolve_supersedence(ranked: list[RankedChunk]) -> ResolvedEvidence:
    """Split ranked evidence into current truth and dated history, by rule alone.

    Input ranking order is preserved within both lists. Each superseded chunk
    points at its *immediate* successor (the earliest strictly-later chunk on
    the same topic), so a chain of three yields two history records that read
    like a timeline, not two pointers at the newest.
    """
    groups: dict[str, list[RankedChunk]] = {}
    for item in ranked:
        groups.setdefault(topic_key(item.chunk), []).append(item)

    superseded: dict[str, tuple[str, date]] = {}
    for group in groups.values():
        if len(group) < 2:
            continue
        chronological = sorted(
            group, key=lambda item: (item.chunk.observed_at, item.chunk.chunk_id)
        )
        for item in chronological:
            successor = next(
                (
                    later
                    for later in chronological
                    if later.chunk.observed_at > item.chunk.observed_at
                ),
                None,
            )
            if successor is not None:
                superseded[item.chunk.chunk_id] = (
                    successor.chunk.chunk_id,
                    successor.chunk.observed_at,
                )

    return ResolvedEvidence(
        current=[item for item in ranked if item.chunk.chunk_id not in superseded],
        history=[
            SupersededRecord(
                item=item,
                superseded_by=superseded[item.chunk.chunk_id][0],
                superseded_on=superseded[item.chunk.chunk_id][1],
            )
            for item in ranked
            if item.chunk.chunk_id in superseded
        ],
    )

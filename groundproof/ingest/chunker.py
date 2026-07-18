"""Cut dated Documents into retrieval-sized Chunks along reST section boundaries.

The corpus is raw reStructuredText, which carries its own structure: a section
title is a line "underlined" by a run of punctuation at least as long as the
title itself. We split there — chunks stay topically coherent ("Deprecated",
"PEP 594: ...") instead of arbitrary character windows — then pack oversized
sections paragraph-by-paragraph into a size budget. The section title is
prepended to every chunk cut from it, so each chunk names its own topic.

Every chunk inherits its parent document's ``observed_at``: chunking never
invents time.
"""

import re

from groundproof.ingest.models import Chunk, Document

# A reST section underline: one punctuation character repeated to (at least)
# the title's length, e.g. "=====", "-----", "~~~~~".
_UNDERLINE = re.compile(r"^([=\-~^\"'`+#*.:_])\1{2,}\s*$")

DEFAULT_MAX_CHARS = 2000


def _is_underline_for(candidate: str, title: str) -> bool:
    return bool(_UNDERLINE.match(candidate)) and len(candidate.rstrip()) >= len(title.rstrip())


def _split_sections(text: str) -> list[tuple[str, str]]:
    """Split reST into (title, body) pairs; text before the first title gets title ''."""
    lines = text.splitlines()
    sections: list[tuple[str, str]] = []
    title = ""
    body_lines: list[str] = []

    i = 0
    while i < len(lines):
        line = lines[i]
        nxt = lines[i + 1] if i + 1 < len(lines) else None
        starts_section = (
            nxt is not None
            and line.strip()
            and not _UNDERLINE.match(line)
            and _is_underline_for(nxt, line)
        )
        if starts_section:
            # A decorative overline (title framed above and below) would have been
            # swept into the previous body — drop it there.
            if body_lines and _is_underline_for(body_lines[-1], line):
                body_lines.pop()
            sections.append((title, "\n".join(body_lines)))
            title = line.strip()
            body_lines = []
            i += 2
            continue
        body_lines.append(line)
        i += 1

    sections.append((title, "\n".join(body_lines)))
    return [(t, body.strip()) for t, body in sections if body.strip()]


def _pack_paragraphs(body: str, max_chars: int) -> list[str]:
    """Greedily pack blank-line-separated paragraphs into pieces of <= max_chars.

    A single paragraph longer than the budget is kept whole — we never cut
    mid-paragraph, because a truncated sentence is worse retrieval evidence
    than an oversized chunk.
    """
    paragraphs = [p for p in re.split(r"\n\s*\n", body) if p.strip()]
    pieces: list[str] = []
    current = ""
    for paragraph in paragraphs:
        candidate = f"{current}\n\n{paragraph}" if current else paragraph
        if len(candidate) <= max_chars or not current:
            current = candidate
        else:
            pieces.append(current)
            current = paragraph
    if current:
        pieces.append(current)
    return pieces


def chunk_document(document: Document, max_chars: int = DEFAULT_MAX_CHARS) -> list[Chunk]:
    """Cut one document into ordered, dated, deterministically-ID'd chunks."""
    chunks: list[Chunk] = []
    for title, body in _split_sections(document.text):
        for piece in _pack_paragraphs(body, max_chars):
            position = len(chunks)
            chunks.append(
                Chunk(
                    chunk_id=f"{document.doc_id}:{position:04d}",
                    doc_id=document.doc_id,
                    source=document.source,
                    text=f"{title}\n\n{piece}" if title else piece,
                    position=position,
                    observed_at=document.observed_at,
                )
            )
    return chunks


def chunk_corpus(documents: list[Document], max_chars: int = DEFAULT_MAX_CHARS) -> list[Chunk]:
    """Chunk every document, preserving corpus order."""
    return [chunk for document in documents for chunk in chunk_document(document, max_chars)]

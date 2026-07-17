"""Persist the parsed corpus as one JSON file per document, and load it back.

The corpus on disk is the reproducibility contract: fetched once, committed,
and every demo/eval loads the same dated documents without touching the network.
"""

from pathlib import Path

from groundproof.ingest.models import Document


def save_corpus(documents: list[Document], corpus_dir: Path) -> list[Path]:
    """Write each document to ``<corpus_dir>/<doc_id>.json``; returns the paths written."""
    corpus_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for document in documents:
        path = corpus_dir / f"{document.doc_id}.json"
        path.write_text(document.model_dump_json(indent=2), encoding="utf-8")
        paths.append(path)
    return paths


def load_corpus(corpus_dir: Path) -> list[Document]:
    """Load every ``*.json`` document, oldest ``observed_at`` first (ties by doc_id)."""
    documents = [
        Document.model_validate_json(path.read_text(encoding="utf-8"))
        for path in sorted(corpus_dir.glob("*.json"))
    ]
    return sorted(documents, key=lambda document: (document.observed_at, document.doc_id))

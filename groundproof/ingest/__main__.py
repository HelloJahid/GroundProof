"""Live corpus ingestion: ``python -m groundproof.ingest``.

The ONLY entry point that touches the real network. Fetches release metadata
and "What's New" texts, then writes ``corpus/*.json`` for committing.
"""

from pathlib import Path

import httpx

from groundproof.ingest.corpus_io import save_corpus
from groundproof.ingest.fetch import fetch_corpus


def main() -> None:
    corpus_dir = Path("corpus")
    with httpx.Client(timeout=30.0, follow_redirects=True) as client:
        documents = fetch_corpus(client)
    save_corpus(documents, corpus_dir)
    for document in documents:
        print(f"  {document.observed_at}  {document.doc_id}  ({len(document.text):,} chars)")
    print(f"{len(documents)} documents written to {corpus_dir}/")


if __name__ == "__main__":
    main()

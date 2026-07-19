"""The eval gate: ``python -m groundproof.evals [corpus_dir]``.

One command, CI's last word: index the committed corpus (deterministic mock
embedder — no keys, no network), run the temporal golden pairs through the
pipeline, run the A/B compression harness, print both scorecards, and exit
non-zero if any golden case broke or compression degraded retention.
"""

import sys
from pathlib import Path

from agentproof.evals.ci_gate import run_gate

from groundproof.compress import QueryAwarePruner
from groundproof.evals.ab import load_ab_cases, render_ab_scorecard, run_ab
from groundproof.evals.temporal import load_temporal_cases, run_temporal_suite
from groundproof.ingest import load_corpus
from groundproof.retrieval import (
    InMemoryVectorStore,
    MockEmbeddingClient,
    TemporalRetriever,
    index_documents,
)


def main() -> int:
    corpus_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("corpus")
    trace_dir = Path("runs/evals")

    embedder = MockEmbeddingClient()
    store = InMemoryVectorStore()
    chunk_count = index_documents(load_corpus(corpus_dir), embedder, store)
    print(f"indexed {chunk_count} chunks from {corpus_dir}/\n")
    retriever = TemporalRetriever(store, embedder)

    suite = run_temporal_suite(
        load_temporal_cases(Path("datasets/temporal_golden.jsonl")),
        retriever=retriever,
        embedder=embedder,
        trace_dir=trace_dir,
    )
    exit_code = run_gate(suite, scorecard_path=trace_dir / "scorecard.json")

    report = run_ab(
        load_ab_cases(Path("datasets/ab_compression.jsonl")),
        retriever=retriever,
        embedder=embedder,
        pruner=QueryAwarePruner(embedder),
    )
    print()
    print(render_ab_scorecard(report))
    if not report.retention_intact:
        print("A/B gate: FAIL -- compression degraded retention")
        exit_code = exit_code or 1
    else:
        print("A/B gate: PASS")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

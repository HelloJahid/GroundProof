"""Pure helpers behind the Streamlit app — zero streamlit imports, fully testable.

The UI file (``demo/app.py``) is presentation only; everything it needs to
*do* lives here, assembled from the same pieces the CLI uses. This split is
what lets the test suite cover the app's behavior without ever importing
streamlit.
"""

import uuid
from dataclasses import dataclass
from datetime import date
from os import environ
from pathlib import Path

from agentproof.trace.recorder import TraceRecorder

from demo.ask import make_model, make_web_executor
from groundproof.compress import QueryAwarePruner
from groundproof.evals.ab import ABReport, load_ab_cases, run_ab
from groundproof.ingest import load_corpus
from groundproof.retrieval import (
    InMemoryVectorStore,
    MockEmbeddingClient,
    TemporalRetriever,
    index_documents,
)
from groundproof.steps import GroundState, build_pipeline


def key_status() -> dict[str, bool]:
    """Which live-mode keys are present (reporting only — make_model decides)."""
    return {
        "ANTHROPIC_API_KEY": bool(environ.get("ANTHROPIC_API_KEY")),
        "TAVILY_API_KEY": bool(environ.get("TAVILY_API_KEY")),
    }


@dataclass(frozen=True)
class Resources:
    """Everything expensive to build, cached once per app session."""

    embedder: MockEmbeddingClient
    store: InMemoryVectorStore
    retriever: TemporalRetriever
    chunk_count: int


def build_resources(corpus_dir: str = "corpus") -> Resources:
    embedder = MockEmbeddingClient()
    store = InMemoryVectorStore()
    chunk_count = index_documents(load_corpus(Path(corpus_dir)), embedder, store)
    return Resources(
        embedder=embedder,
        store=store,
        retriever=TemporalRetriever(store, embedder),
        chunk_count=chunk_count,
    )


def run_question(
    question: str,
    as_of: date | None,
    *,
    live: bool,
    compress: bool,
    resources: Resources,
    top_k: int = 5,
    trace_dir: Path = Path("runs"),
) -> tuple[GroundState, Path]:
    """One full pipeline run, traced — the same assembly as the CLI."""
    machine = build_pipeline(
        retriever=resources.retriever,
        model=make_model(live),
        embedder=resources.embedder,
        web_executor=make_web_executor(live),
        pruner=QueryAwarePruner(resources.embedder) if compress else None,
        top_k=top_k,
    )
    trace_dir.mkdir(parents=True, exist_ok=True)
    trace_path = trace_dir / f"app-{uuid.uuid4().hex[:8]}.trace.jsonl"
    state = GroundState(query=question, as_of=as_of)
    with TraceRecorder(trace_path) as recorder:
        machine.run(state, recorder=recorder)
    return state, trace_path


def build_ab_report(resources: Resources) -> ABReport:
    """The A/B harness on the committed golden cases — deterministic, mock-only."""
    return run_ab(
        load_ab_cases(Path("datasets/ab_compression.jsonl")),
        retriever=resources.retriever,
        embedder=resources.embedder,
        pruner=QueryAwarePruner(resources.embedder),
    )


def ab_rows(report: ABReport) -> list[dict]:
    """ABReport flattened into table rows for the UI."""
    return [
        {
            "case": result.case_id,
            "tokens (off)": result.tokens_off,
            "tokens (on)": result.tokens_on,
            "saved": result.savings,
            "retention off": result.retention_off,
            "retention on": result.retention_on,
        }
        for result in report.results
    ]

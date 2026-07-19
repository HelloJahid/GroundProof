"""The GroundProof CLI: ask a question as of a moment in time.

    python -m demo.ask "is the distutils package removed" --as-of 2024-06
    python -m demo.ask "latest stable python highlights" --as-of today --no-compress
    python -m demo.ask "..." --live     # AnthropicClient + Tavily (needs .env keys)

Every run leaves a trace; the cockpit replays it:

    python -m groundproof.cockpit runs/ask-<id>.trace.jsonl
"""

import argparse
import os
import sys
import uuid
from datetime import date
from pathlib import Path

from agentproof import AnthropicClient, ModelClient
from agentproof.tools.executor import ToolExecutor
from agentproof.tools.registry import ToolRegistry
from agentproof.tools.transports import MockTransport, TavilySearchTransport
from agentproof.trace.recorder import TraceRecorder

from demo.env import load_env
from demo.offline_model import OfflineSynthesizer
from groundproof.cockpit import render_replay
from groundproof.compress import QueryAwarePruner
from groundproof.evals.temporal import load_ground_replay
from groundproof.ingest import load_corpus
from groundproof.retrieval import (
    InMemoryVectorStore,
    MockEmbeddingClient,
    TemporalRetriever,
    index_documents,
)
from groundproof.steps import GroundState, build_pipeline


def parse_as_of(raw: str | None) -> date | None:
    if raw is None or raw == "today":
        return date.today() if raw == "today" else None
    parts = raw.split("-")
    if len(parts) == 2:  # YYYY-MM -> first of the month
        return date(int(parts[0]), int(parts[1]), 1)
    return date.fromisoformat(raw)


def make_model(live: bool) -> ModelClient:
    if live and os.environ.get("ANTHROPIC_API_KEY"):
        return AnthropicClient()
    if live:
        print("(--live requested but ANTHROPIC_API_KEY not set — using offline synthesizer)")
    return OfflineSynthesizer()


def make_web_executor(live: bool) -> ToolExecutor:
    from groundproof.steps import make_web_search_tool

    registry = ToolRegistry([make_web_search_tool()])
    tavily_key = os.environ.get("TAVILY_API_KEY", "")
    if live and tavily_key:
        return ToolExecutor(registry, TavilySearchTransport(tavily_key))
    return ToolExecutor(registry, MockTransport({"web_search": []}))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ask", description="GroundProof: time-aware RAG")
    parser.add_argument("question")
    parser.add_argument("--as-of", default=None, help="YYYY-MM, YYYY-MM-DD, or 'today'")
    parser.add_argument("--no-compress", action="store_true", help="disable Hook B")
    parser.add_argument("--live", action="store_true", help="real model + web search from .env")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--corpus", default="corpus", help="corpus directory")
    parser.add_argument("--trace-dir", default="runs", help="where the trace file goes")
    args = parser.parse_args(argv)

    load_env()
    embedder = MockEmbeddingClient()  # embeddings provider port: mock for now
    store = InMemoryVectorStore()
    count = index_documents(load_corpus(Path(args.corpus)), embedder, store)

    machine = build_pipeline(
        retriever=TemporalRetriever(store, embedder),
        model=make_model(args.live),
        embedder=embedder,
        web_executor=make_web_executor(args.live),
        pruner=None if args.no_compress else QueryAwarePruner(embedder),
        top_k=args.top_k,
    )

    trace_path = Path(args.trace_dir) / f"ask-{uuid.uuid4().hex[:8]}.trace.jsonl"
    state = GroundState(query=args.question, as_of=parse_as_of(args.as_of))
    print(f"indexed {count} chunks; asking as of {state.as_of or 'latest knowledge'}...\n")
    with TraceRecorder(trace_path) as recorder:
        machine.run(state, recorder=recorder)

    print(render_replay(load_ground_replay(trace_path)))
    print(f"\ntrace: {trace_path}")
    print(f"replay it: python -m groundproof.cockpit {trace_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

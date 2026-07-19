"""P5: temporal golden pairs, GroundState replay, stale-fact tripwire, judge."""

import json
from datetime import date
from pathlib import Path

import pytest
from agentproof import MockModelClient, ModelResponse
from agentproof.evals.ci_gate import run_gate
from agentproof.evals.scorecard import render_scorecard
from agentproof.trace.replay import ReplayStep, RunReplay

from groundproof.evals import (
    ExpectedSourceCheck,
    GroundednessJudge,
    PhrasesCheck,
    TemporalCase,
    TemporalIntegrityCheck,
    load_temporal_cases,
    run_temporal_suite,
)
from groundproof.ingest import Chunk
from groundproof.retrieval import (
    InMemoryVectorStore,
    MockEmbeddingClient,
    RankedChunk,
    TemporalRetriever,
)
from groundproof.steps import GroundState

EMBEDDER = MockEmbeddingClient()


def make_chunk(chunk_id: str, text: str, observed_at: date) -> Chunk:
    doc_id = chunk_id.rsplit(":", 1)[0]
    return Chunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        source=f"https://example.test/{doc_id}.rst.txt",
        text=text,
        position=0,
        observed_at=observed_at,
    )


V312 = make_chunk(
    "python-whatsnew-3.12:0000",
    "Summary\n\nPython 3.12 is the latest stable release. The distutils package was removed.",
    date(2023, 10, 2),
)
V314 = make_chunk(
    "python-whatsnew-3.14:0000",
    "Summary\n\nPython 3.14 is the latest stable release of Python.",
    date(2025, 10, 7),
)


def make_retriever(chunks: list[Chunk]) -> TemporalRetriever:
    store = InMemoryVectorStore()
    store.index(chunks, EMBEDDER.embed_texts([chunk.text for chunk in chunks]))
    return TemporalRetriever(store, EMBEDDER)


def replay_with(state: GroundState) -> RunReplay:
    return RunReplay(
        run_id="r1",
        query=state.query,
        instructions="",
        steps=[ReplayStep(name="synthesize", duration_ms=1.0, state=state)],
        outcome="finished",
        final_answer=state.final_answer,
    )


def ranked(chunk: Chunk) -> RankedChunk:
    return RankedChunk(chunk=chunk, similarity=0.5, recency=1.0, score=0.5)


CASE = TemporalCase(
    case_id="latest-2024",
    question="latest stable python",
    as_of=date(2024, 6, 1),
    expected_doc_id="python-whatsnew-3.12",
    expected_phrases=["3.12"],
)


class TestChecks:
    def test_integrity_passes_when_no_future_evidence(self):
        state = GroundState(query="q", evidence=[ranked(V312)])
        result = TemporalIntegrityCheck().evaluate(CASE, replay_with(state))
        assert result.passed

    def test_integrity_fails_and_names_leaking_chunks(self):
        state = GroundState(query="q", evidence=[ranked(V314)])
        result = TemporalIntegrityCheck().evaluate(CASE, replay_with(state))
        assert not result.passed
        assert "python-whatsnew-3.14:0000" in result.reason

    def test_expected_source_names_both_docs_on_failure(self):
        state = GroundState(query="q", evidence=[ranked(V314)])
        result = ExpectedSourceCheck().evaluate(CASE, replay_with(state))
        assert not result.passed
        assert "python-whatsnew-3.12" in result.reason
        assert "python-whatsnew-3.14" in result.reason

    def test_phrases_check_reports_missing(self):
        state = GroundState(query="q", evidence=[ranked(V314)])
        result = PhrasesCheck().evaluate(CASE, replay_with(state))
        assert not result.passed
        assert "3.12" in result.reason

    def test_phrases_check_not_applicable_without_phrases(self):
        bare = CASE.model_copy(update={"expected_phrases": []})
        result = PhrasesCheck().evaluate(bare, replay_with(GroundState(query="q")))
        assert not result.applicable


class TestSuite:
    def golden_pair(self) -> list[TemporalCase]:
        return [
            TemporalCase(
                case_id="latest-2024",
                pair_id="latest",
                question="latest stable python release",
                as_of=date(2024, 6, 1),
                expected_doc_id="python-whatsnew-3.12",
                expected_phrases=["3.12"],
            ),
            TemporalCase(
                case_id="latest-2026",
                pair_id="latest",
                question="latest stable python release",
                as_of=date(2026, 1, 1),
                expected_doc_id="python-whatsnew-3.14",
                expected_phrases=["3.14"],
            ),
        ]

    def test_time_travel_pair_passes(self, tmp_path):
        suite = run_temporal_suite(
            self.golden_pair(),
            retriever=make_retriever([V312, V314]),
            embedder=EMBEDDER,
            trace_dir=tmp_path,
        )
        assert suite.all_passed
        assert render_scorecard(suite)  # renders without blowing up
        assert run_gate(suite) == 0

    def test_traces_are_left_as_evidence(self, tmp_path):
        run_temporal_suite(
            self.golden_pair(),
            retriever=make_retriever([V312, V314]),
            embedder=EMBEDDER,
            trace_dir=tmp_path,
        )
        trace = tmp_path / "latest-2024.trace.jsonl"
        assert trace.exists()
        kinds = [json.loads(line)["kind"] for line in trace.read_text("utf-8").splitlines()]
        assert kinds[0] == "run_started" and kinds[-1] == "run_finished"

    def test_stale_fact_goes_red_and_names_the_case(self, tmp_path):
        """Demo moment #2: a superseding doc flips yesterday's answer; the
        gate goes red naming the case and both documents."""
        # Dated BEFORE the 2026 case's as-of (a doc from after it would be
        # invisible to the case — the as-of filter itself protects the golden).
        v315 = make_chunk(
            "python-whatsnew-3.15:0000",
            "Summary\n\nPython 3.15 is the latest stable release of Python.",
            date(2025, 12, 1),
        )
        suite = run_temporal_suite(
            self.golden_pair(),
            retriever=make_retriever([V312, V314, v315]),
            embedder=EMBEDDER,
            trace_dir=tmp_path,
        )
        # As-of 2024 half is untouched by the future doc; the 2026 half breaks.
        by_id = {result.case_id: result for result in suite.results}
        assert by_id["latest-2024"].passed
        assert not by_id["latest-2026"].passed
        failing = next(
            check for check in by_id["latest-2026"].checks if check.check == "expected_source"
        )
        assert "python-whatsnew-3.14" in failing.reason
        assert "python-whatsnew-3.15" in failing.reason
        assert run_gate(suite) == 1


class TestGroundednessJudge:
    def test_judge_sees_dated_evidence_and_returns_verdict(self):
        state = GroundState(
            query="latest stable python",
            evidence=[ranked(V312)],
            final_answer="Python 3.12 is latest (2023-10-02, python-whatsnew-3.12).",
        )
        client = MockModelClient(
            [ModelResponse(text='{"passed": true, "score": 0.9, "reason": "cited and dated"}')]
        )
        judge = GroundednessJudge(client)
        from agentproof.evals.datasets import EvalCase

        result = judge.evaluate(
            EvalCase(id="c", query=state.query), replay_with(state)
        )
        assert result.passed and result.score == pytest.approx(0.9)
        prompt = client.calls[0]["messages"][0].content
        assert "(2023-10-02, python-whatsnew-3.12)" in prompt
        assert "dated citation" in prompt  # the adapted rubric is on the table


class TestGoldenDataset:
    def test_committed_dataset_loads_with_pairs(self):
        cases = load_temporal_cases(Path("datasets/temporal_golden.jsonl"))
        assert len(cases) == 5
        pair_ids = [case.pair_id for case in cases if case.pair_id]
        assert pair_ids.count("latest-version") == 2
        assert pair_ids.count("distutils") == 2

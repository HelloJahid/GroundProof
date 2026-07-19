"""Temporal golden pairs: time-travel and stale-fact detection as eval cases.

A golden PAIR is the same question asked at two as-of moments, each half
expecting evidence from a different document — proving the system actually
travels in time. The checks judge the TRACE (never the live run), and they
grade retrieval evidence rather than model prose, so the whole gate runs
deterministic and fully mocked:

- ``TemporalIntegrityCheck`` (system): no evidence may postdate the as-of.
- ``ExpectedSourceCheck`` (task_completion): the top evidence must come from
  the expected document. This is the stale-fact tripwire — add a superseding
  document to the corpus and this check goes red, naming the case and both
  doc ids.
- ``PhrasesCheck`` (quality): the facts the answer depends on are in evidence.

One seam vs stock AgentProof: its replay loader re-validates snapshots as the
base ``AgentState`` (extra="forbid"), which rejects GroundState's fields — so
``load_ground_replay`` rebuilds snapshots as GroundState and hands back a
normal ``RunReplay``. Everything downstream (SuiteResult, scorecard,
run_gate, the judge) is stock AgentProof.
"""

from datetime import date
from pathlib import Path

from agentproof import AgentProofError, MockModelClient, ModelResponse
from agentproof.errors import ReplayError
from agentproof.evals.datasets import EvalCase
from agentproof.evals.harness import CaseResult, SuiteResult
from agentproof.evals.judges import JudgeEvaluator
from agentproof.evals.results import CheckResult
from agentproof.tools.executor import ToolExecutor
from agentproof.tools.registry import ToolRegistry
from agentproof.tools.transports import MockTransport
from agentproof.trace.recorder import TraceRecorder
from agentproof.trace.records import RunFailed, RunFinished, RunStarted, StepCompleted, parse_event
from agentproof.trace.replay import ReplayStep, RunReplay
from pydantic import BaseModel, ConfigDict, Field

from groundproof.compress.pruner import QueryAwarePruner
from groundproof.retrieval.embeddings import EmbeddingClient
from groundproof.retrieval.temporal import TemporalRetriever
from groundproof.steps.pipeline import build_pipeline
from groundproof.steps.state import GroundState
from groundproof.steps.web_search import make_web_search_tool

DATED_GROUNDEDNESS_RUBRIC = (
    "PASS only if ALL of the following hold:\n"
    "1. Every factual claim in the answer is supported by the dated evidence.\n"
    "2. Every claim carries a dated citation (YYYY-MM-DD, source-id) whose "
    "date and source both appear in the evidence.\n"
    "3. Nothing material is invented beyond the evidence.\n"
    "If the answer declines because no reliable information was found, PASS "
    "only when the evidence indeed could not answer the question."
)


class TemporalCase(BaseModel):
    """One half of a golden pair: a question, a moment, and the expected truth."""

    model_config = ConfigDict(frozen=True)

    case_id: str
    question: str
    as_of: date
    expected_doc_id: str
    expected_phrases: list[str] = Field(default_factory=list)
    pair_id: str | None = None


def load_temporal_cases(path: Path) -> list[TemporalCase]:
    cases = [
        TemporalCase.model_validate_json(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    ids = [case.case_id for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate case ids in {path}")
    return cases


def load_ground_replay(path: Path) -> RunReplay:
    """A RunReplay whose snapshots are typed GroundState, not base AgentState."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ReplayError("trace file is empty")
    events = [parse_event(line) for line in lines]
    head = events[0]
    if not isinstance(head, RunStarted):
        raise ReplayError(f"first event must be run_started, got {head.kind!r}")
    if len({event.run_id for event in events}) != 1:
        raise ReplayError("trace mixes events from more than one run")
    if [event.seq for event in events] != list(range(len(events))):
        raise ReplayError("sequence numbers have gaps or are out of order")

    replay = RunReplay(
        run_id=head.run_id,
        query=head.query,
        instructions=head.instructions,
        steps=[
            ReplayStep(
                name=event.step,
                duration_ms=event.duration_ms,
                state=GroundState.model_validate(event.state),
            )
            for event in events
            if isinstance(event, StepCompleted)
        ],
        outcome="truncated",
    )
    for event in events:
        if isinstance(event, RunFinished):
            return replay.model_copy(
                update={
                    "outcome": "finished",
                    "final_answer": event.final_answer,
                    "total_duration_ms": event.duration_ms,
                }
            )
        if isinstance(event, RunFailed):
            return replay.model_copy(
                update={
                    "outcome": "failed",
                    "error_type": event.error_type,
                    "error_message": event.error_message,
                }
            )
    return replay


def _final_ground_state(replay: RunReplay) -> GroundState | None:
    state = replay.final_state
    return state if isinstance(state, GroundState) else None


class TemporalIntegrityCheck:
    """No retrieved evidence may come from after the as-of moment. Ever."""

    name = "temporal_integrity"

    def evaluate(self, case: TemporalCase, replay: RunReplay) -> CheckResult:
        state = _final_ground_state(replay)
        if state is None or not state.evidence:
            return CheckResult(
                check=self.name,
                dimension="system",
                passed=False,
                reason="no evidence in the final state",
                applicable=state is not None,
            )
        leaks = [
            item.chunk.chunk_id for item in state.evidence if item.chunk.observed_at > case.as_of
        ]
        if leaks:
            return CheckResult(
                check=self.name,
                dimension="system",
                passed=False,
                reason=f"evidence from after {case.as_of}: {leaks}",
            )
        return CheckResult(
            check=self.name,
            dimension="system",
            passed=True,
            reason=f"all {len(state.evidence)} evidence chunks observed on or before {case.as_of}",
        )


class ExpectedSourceCheck:
    """Top evidence must come from the expected document — the stale-fact tripwire."""

    name = "expected_source"

    def evaluate(self, case: TemporalCase, replay: RunReplay) -> CheckResult:
        state = _final_ground_state(replay)
        if state is None or not state.evidence:
            return CheckResult(
                check=self.name,
                dimension="task_completion",
                passed=False,
                reason="no evidence to check",
            )
        actual = state.evidence[0].chunk.doc_id
        passed = actual == case.expected_doc_id
        reason = (
            f"top evidence from {actual} (as expected)"
            if passed
            else f"expected top evidence from {case.expected_doc_id}, got {actual} — "
            "the corpus has changed what this question retrieves"
        )
        return CheckResult(
            check=self.name, dimension="task_completion", passed=passed, reason=reason
        )


class PhrasesCheck:
    """The facts the answer depends on must be present in the evidence."""

    name = "expected_phrases"

    def evaluate(self, case: TemporalCase, replay: RunReplay) -> CheckResult:
        if not case.expected_phrases:
            return CheckResult(
                check=self.name,
                dimension="quality",
                passed=True,
                reason="case declares no required phrases",
                applicable=False,
            )
        state = _final_ground_state(replay)
        joined = (
            " ".join(item.chunk.text.lower() for item in state.evidence) if state else ""
        )
        missing = [p for p in case.expected_phrases if p.lower() not in joined]
        if missing:
            return CheckResult(
                check=self.name,
                dimension="quality",
                passed=False,
                reason=f"evidence is missing required phrases: {missing}",
            )
        return CheckResult(
            check=self.name,
            dimension="quality",
            passed=True,
            reason=f"all {len(case.expected_phrases)} required phrases present in evidence",
        )


class GroundednessJudge(JudgeEvaluator):
    """AgentProof's judge (fixed rubric, gate-checked verdict, retry-with-
    feedback), with GroundProof's dated evidence laid on the table instead of
    tool observations."""

    def __init__(self, client, name: str = "groundedness_judge") -> None:
        super().__init__(client, rubric=DATED_GROUNDEDNESS_RUBRIC, name=name)

    def _build_prompt(self, case: EvalCase, replay: RunReplay) -> str:
        state = _final_ground_state(replay)
        lines: list[str] = []
        if state is not None:
            lines.extend(
                f"({item.chunk.observed_at}, {item.chunk.doc_id}) {item.chunk.text}"
                for item in state.evidence
            )
            lines.extend(
                f"(superseded {record.superseded_on} by {record.superseded_by}) "
                f"({record.item.chunk.observed_at}, {record.item.chunk.doc_id}) "
                f"{record.item.chunk.text}"
                for record in state.evidence_history
            )
            if state.web_evidence:
                lines.append(f"(live web search) {state.web_evidence}")
        evidence = "\n\n".join(lines) or "(no evidence was collected in this run)"
        return (
            f"RUBRIC:\n{self._rubric}\n\n"
            f"QUESTION:\n{replay.query}\n\n"
            f"EVIDENCE (dated):\n{evidence}\n\n"
            f"ANSWER:\n{replay.final_answer or '(no answer was produced)'}"
        )


DEFAULT_CHECKS = (TemporalIntegrityCheck(), ExpectedSourceCheck(), PhrasesCheck())


def run_temporal_suite(
    cases: list[TemporalCase],
    *,
    retriever: TemporalRetriever,
    embedder: EmbeddingClient,
    trace_dir: Path,
    pruner: QueryAwarePruner | None = None,
    judge: GroundednessJudge | None = None,
) -> SuiteResult:
    """Run each case through a fresh pipeline; judge the trace, never the run."""
    results: list[CaseResult] = []
    for case in cases:
        machine = build_pipeline(
            retriever=retriever,
            model=MockModelClient([ModelResponse(text="(scripted synthesis for eval run)")]),
            embedder=embedder,
            web_executor=ToolExecutor(
                ToolRegistry([make_web_search_tool()]), MockTransport({"web_search": []})
            ),
            pruner=pruner,
        )
        trace_path = trace_dir / f"{case.case_id}.trace.jsonl"
        state = GroundState(query=case.question, as_of=case.as_of)
        with TraceRecorder(trace_path) as recorder:
            try:
                machine.run(state, recorder=recorder)
            except AgentProofError:
                pass  # the recorder captured it; the replay will show it

        replay = load_ground_replay(trace_path)
        checks = [check.evaluate(case, replay) for check in DEFAULT_CHECKS]
        if judge is not None:
            shim = EvalCase(id=case.case_id, query=case.question)
            checks.append(judge.evaluate(shim, replay))
        results.append(
            CaseResult(
                case_id=case.case_id,
                outcome=replay.outcome,
                path=replay.path,
                final_answer=replay.final_answer,
                checks=checks,
                passed=all(check.passed for check in checks if check.applicable),
                trace_path=str(trace_path),
            )
        )
    return SuiteResult(results=results)

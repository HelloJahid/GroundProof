"""P3: the corrective spine end-to-end — including the DoD trajectory test.

Everything mocked: MockEmbeddingClient vectors, InMemoryVectorStore evidence,
MockModelClient synthesis, MockTransport web search. The trajectories asserted
here are the phase's deliverable.
"""

import json
from datetime import date

import pytest
from agentproof import MockModelClient, ModelResponse, TransportError
from agentproof.tools.executor import ToolExecutor
from agentproof.tools.registry import ToolRegistry
from agentproof.tools.transports import MockTransport
from agentproof.trace.recorder import TraceRecorder

from groundproof.ingest import Chunk
from groundproof.retrieval import (
    InMemoryVectorStore,
    MockEmbeddingClient,
    TemporalRetriever,
)
from groundproof.steps import (
    HONEST_FAILURE,
    AdaptiveRouterStep,
    GroundState,
    build_pipeline,
    make_web_search_tool,
)

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


CORPUS = [
    make_chunk(
        "python-whatsnew-3.12:0000",
        "Removed\n\nThe distutils package was removed in Python 3.12.",
        date(2023, 10, 2),
    ),
    make_chunk(
        "python-whatsnew-3.9:0000",
        "Added\n\nThe zoneinfo module was added in Python 3.9.",
        date(2020, 10, 5),
    ),
]


def make_retriever(chunks: list[Chunk] = CORPUS) -> TemporalRetriever:
    store = InMemoryVectorStore()
    store.index(chunks, EMBEDDER.embed_texts([chunk.text for chunk in chunks]))
    return TemporalRetriever(store, EMBEDDER)


def make_web_executor(script: list) -> ToolExecutor:
    registry = ToolRegistry([make_web_search_tool()])
    return ToolExecutor(registry, MockTransport({"web_search": script}))


def trajectory(state: GroundState) -> list[str]:
    return [record.step for record in state.history]


class TestAdaptiveRouterStep:
    @pytest.mark.parametrize(
        ("query", "expected"),
        [
            ("hello there!", "none"),
            ("thanks a lot", "none"),
            ("is distutils removed?", "one_shot"),
            ("what is the latest python? and when was distutils removed?", "multi_hop"),
        ],
    )
    def test_classification(self, query, expected):
        state = AdaptiveRouterStep().run(GroundState(query=query))
        assert state.route == expected

    def test_multi_hop_decomposes_into_sub_questions(self):
        state = AdaptiveRouterStep().run(
            GroundState(query="what is the latest python? when was distutils removed?")
        )
        assert state.sub_questions == [
            "what is the latest python?",
            "when was distutils removed?",
        ]


class TestStrongEvidencePath:
    def test_route_retrieve_grade_synthesize(self):
        model = MockModelClient(
            [
                ModelResponse(
                    text="Distutils was removed (2023-10-02, python-whatsnew-3.12).",
                    input_tokens=120,
                    output_tokens=25,
                )
            ]
        )
        machine = build_pipeline(
            retriever=make_retriever(), model=model, web_executor=make_web_executor([])
        )
        state = machine.run(GroundState(query="is the distutils package removed"))

        assert trajectory(state) == ["route", "retrieve", "grade", "synthesize"]
        assert state.grade is not None and state.grade.verdict == "strong"
        assert state.final_answer is not None and "2023-10-02" in state.final_answer
        assert state.input_tokens == 120 and state.output_tokens == 25
        # Synthesis saw dated evidence, not bare text.
        prompt = model.calls[0]["messages"][-1].content
        assert "(2023-10-02, python-whatsnew-3.12)" in prompt

    def test_no_retrieval_route_goes_straight_to_synthesis(self):
        model = MockModelClient([ModelResponse(text="Hello! Ask me about Python releases.")])
        machine = build_pipeline(
            retriever=make_retriever(), model=model, web_executor=make_web_executor([])
        )
        state = machine.run(GroundState(query="hello there!"))
        assert trajectory(state) == ["route", "synthesize"]
        assert state.retrieval_attempts == 0

    def test_multi_hop_merges_evidence_from_both_sub_questions(self):
        model = MockModelClient([ModelResponse(text="Both answered.")])
        machine = build_pipeline(
            retriever=make_retriever(), model=model, web_executor=make_web_executor([])
        )
        state = machine.run(
            GroundState(query="when was the distutils package removed? when was zoneinfo added?")
        )
        assert state.route == "multi_hop"
        doc_ids = {item.chunk.doc_id for item in state.evidence}
        assert doc_ids == {"python-whatsnew-3.12", "python-whatsnew-3.9"}


class TestCorrectiveFallback:
    """The Phase 3 Definition of Done: weak evidence provably triggers
    reformulation, then web fallback — trajectory visible in the trace."""

    WEAK_QUERY = "airspeed velocity of an unladen swallow"

    def test_weak_evidence_triggers_reformulate_then_web_search(self):
        web_payload = {
            "results": [
                {
                    "title": "Swallow airspeed",
                    "url": "https://example.test/swallow",
                    "content": "An unladen European swallow flies at roughly 11 m/s.",
                }
            ]
        }
        model = MockModelClient(
            [ModelResponse(text="Roughly 11 m/s (live web search).", input_tokens=90)]
        )
        machine = build_pipeline(
            retriever=make_retriever(),
            model=model,
            web_executor=make_web_executor([web_payload]),
        )
        state = machine.run(GroundState(query=self.WEAK_QUERY))

        assert trajectory(state) == [
            "route",
            "retrieve",
            "grade",
            "reformulate",
            "retrieve",
            "grade",
            "web_search",
            "synthesize",
        ]
        assert state.active_query is not None and state.active_query != self.WEAK_QUERY
        assert state.retrieval_attempts == 2
        assert state.web_evidence is not None and "11 m/s" in state.web_evidence
        assert state.final_answer == "Roughly 11 m/s (live web search)."

    def test_trajectory_is_visible_in_the_trace(self, tmp_path):
        web_payload = {"results": [{"title": "t", "url": "u", "content": "11 m/s."}]}
        model = MockModelClient([ModelResponse(text="Roughly 11 m/s.")])
        machine = build_pipeline(
            retriever=make_retriever(),
            model=model,
            web_executor=make_web_executor([web_payload]),
        )
        trace_path = tmp_path / "run.jsonl"
        with TraceRecorder(trace_path) as recorder:
            machine.run(GroundState(query=self.WEAK_QUERY), recorder=recorder)

        events = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
        steps = [event["step"] for event in events if event["kind"] == "step_completed"]
        assert steps == [
            "route",
            "retrieve",
            "grade",
            "reformulate",
            "retrieve",
            "grade",
            "web_search",
            "synthesize",
        ]
        # The state snapshots carry the corrective details for every step.
        final_snapshot = events[-2]["state"]
        assert final_snapshot["retrieval_attempts"] == 2
        assert final_snapshot["grade"]["verdict"] == "weak"
        assert events[-1]["kind"] == "run_finished"

    def test_web_failure_ends_in_honest_refusal_without_a_model_call(self):
        model = MockModelClient([])  # any model call would raise: none may happen
        machine = build_pipeline(
            retriever=make_retriever(),
            model=model,
            web_executor=make_web_executor(
                [TransportError("down"), TransportError("down"), TransportError("down")]
            ),
        )
        state = machine.run(GroundState(query=self.WEAK_QUERY))

        assert trajectory(state)[-2:] == ["web_search", "synthesize"]
        assert state.web_evidence is None
        assert state.tool_results[-1].is_error
        assert state.final_answer == HONEST_FAILURE

    def test_reformulation_cannot_game_the_grader(self):
        """Regression (found on the real corpus): the anchor terms appended by
        reformulation overlap the corpus by construction — if grading ran
        against the reformulated query, weak evidence would be waved through
        and the web fallback never reached."""
        anchor_bait = make_chunk(
            "python-whatsnew-3.14:0000",
            "Python Version Release\n\nEvery python version release changelog "
            "notes entry is dated.",
            date(2025, 10, 7),
        )
        model = MockModelClient([ModelResponse(text="From the web.")])
        machine = build_pipeline(
            retriever=make_retriever([*CORPUS, anchor_bait]),
            model=model,
            embedder=EMBEDDER,
            web_executor=make_web_executor(
                [{"results": [{"title": "t", "url": "u", "content": "c"}]}]
            ),
        )
        state = machine.run(GroundState(query="who won the world cup final"))

        assert state.grade is not None and state.grade.verdict == "weak"
        assert "web_search" in trajectory(state)
        assert state.final_answer == "From the web."

    def test_as_of_travels_through_the_pipeline(self):
        model = MockModelClient([ModelResponse(text="As of mid-2021: zoneinfo exists.")])
        machine = build_pipeline(
            retriever=make_retriever(), model=model, web_executor=make_web_executor([])
        )
        state = machine.run(
            GroundState(query="when was the zoneinfo module added", as_of=date(2021, 6, 1))
        )
        assert state.grade is not None and state.grade.verdict == "strong"
        assert all(item.chunk.observed_at <= date(2021, 6, 1) for item in state.evidence)
        prompt = model.calls[0]["messages"][-1].content
        assert "as of 2021-06-01" in prompt

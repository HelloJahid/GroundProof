"""The corrective spine, wired: six steps and one routing policy.

The router is the whole corrective-RAG algorithm in ~15 lines — every step
does one job and never decides what runs next (AgentProof's separation), so
the trajectory logic lives in exactly one, trivially testable place:

    route ──(none)──────────────────────────────► synthesize
      └──(else)─► retrieve ─► grade ─┬─(strong)─► compress ─► synthesize
                      ▲              ├─(weak, 1st)─► reformulate ─┐
                      └──────────────┼────────────────────────────┘
                                     └─(weak, again)─► web_search ─► synthesize
"""

from agentproof import ModelClient, StateMachine
from agentproof.tools.executor import ToolExecutor

from groundproof.compress.pruner import QueryAwarePruner
from groundproof.grading.grader import EvidenceGrader
from groundproof.retrieval.embeddings import EmbeddingClient
from groundproof.retrieval.temporal import TemporalRetriever
from groundproof.steps.compress import CompressStep
from groundproof.steps.grade import GradeStep
from groundproof.steps.reformulate import ReformulateStep
from groundproof.steps.retrieve import RetrieveStep
from groundproof.steps.route import AdaptiveRouterStep
from groundproof.steps.state import GroundState
from groundproof.steps.synthesize import SynthesizeStep
from groundproof.steps.web_search import WebSearchStep


def corrective_router(state: GroundState, current: str) -> str | None:
    if current == "route":
        return "synthesize" if state.route == "none" else "retrieve"
    if current == "retrieve":
        return "grade"
    if current == "grade":
        if state.grade is not None and state.grade.verdict == "strong":
            return "compress"
        if state.retrieval_attempts < 2:
            return "reformulate"
        return "web_search"
    if current == "reformulate":
        return "retrieve"
    if current == "compress":
        return "synthesize"
    if current == "web_search":
        return "synthesize"
    return None


def build_pipeline(
    *,
    retriever: TemporalRetriever,
    model: ModelClient,
    web_executor: ToolExecutor,
    grader: EvidenceGrader | None = None,
    embedder: EmbeddingClient | None = None,
    pruner: QueryAwarePruner | None = None,
    top_k: int = 5,
    max_steps: int = 12,
) -> StateMachine:
    """Assemble the corrective RAG machine. All collaborators arrive as ports.

    Pass ``embedder`` (when no custom ``grader`` is given) so the gate scores
    similarity against the original question, immune to reformulation inflation.
    ``pruner`` switches Hook B on; without it the compress step is a recorded
    no-op — the exact toggle the A/B harness flips.
    """
    return StateMachine(
        steps=[
            AdaptiveRouterStep(),
            RetrieveStep(retriever, top_k=top_k),
            GradeStep(grader or EvidenceGrader(embedder=embedder)),
            ReformulateStep(),
            CompressStep(pruner),
            WebSearchStep(web_executor),
            SynthesizeStep(model),
        ],
        router=corrective_router,
        start="route",
        max_steps=max_steps,
    )

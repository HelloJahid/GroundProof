"""GroundState: AgentProof's working memory, extended with retrieval fields.

The runtime (StateMachine, TraceRecorder) only knows ``AgentState``; every
GroundProof-specific fact a step needs to hand the next one — the route, the
as-of moment, evidence, the grade — lives in typed fields on this subclass.
Because the trace snapshots the state after every step, all of it lands in
the flight recorder for free.
"""

from datetime import date
from typing import Literal

from agentproof import AgentState
from pydantic import Field

from groundproof.compress.pruner import CompressedEvidence
from groundproof.grading.grader import EvidenceGrade
from groundproof.retrieval.supersedence import SupersededRecord
from groundproof.retrieval.temporal import RankedChunk

Route = Literal["none", "one_shot", "multi_hop"]


class GroundState(AgentState):
    """One question's journey through the corrective RAG pipeline."""

    as_of: date | None = None
    route: Route | None = None
    sub_questions: list[str] = Field(default_factory=list)
    # Set by reformulation; retrieval and grading prefer it over the original.
    active_query: str | None = None
    evidence: list[RankedChunk] = Field(default_factory=list)
    evidence_history: list[SupersededRecord] = Field(default_factory=list)
    grade: EvidenceGrade | None = None
    retrieval_attempts: int = 0
    web_evidence: str | None = None
    compressed: CompressedEvidence | None = None

    @property
    def effective_query(self) -> str:
        return self.active_query or self.query

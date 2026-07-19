"""The A/B compression harness: the "~60% fewer tokens" claim as a receipt.

Each golden case runs through the SAME pipeline twice — compressor off, then
on — and the harness measures what actually changed:

- ``tokens``: the deterministic estimate of the synthesis prompt (what the
  model would be billed for), NOT mock-reported usage;
- ``retention``: the fraction of the case's required phrases (the facts the
  answer depends on) still present in the synthesis prompt. Compression that
  saves tokens by dropping the answer is a failure, and this column shows it.

Fully mocked, fully deterministic: the scorecard diff renders identically on
every machine, which is what makes it a claim instead of marketing.
"""

from datetime import date
from pathlib import Path

from agentproof import MockModelClient, ModelResponse
from agentproof.tools.executor import ToolExecutor
from agentproof.tools.registry import ToolRegistry
from agentproof.tools.transports import MockTransport
from pydantic import BaseModel, ConfigDict, Field

from groundproof.compress.pruner import QueryAwarePruner, estimate_tokens
from groundproof.retrieval.embeddings import EmbeddingClient
from groundproof.retrieval.temporal import TemporalRetriever
from groundproof.steps.pipeline import build_pipeline
from groundproof.steps.state import GroundState
from groundproof.steps.web_search import make_web_search_tool


class ABCase(BaseModel):
    """One golden question plus the phrases its answer cannot live without."""

    model_config = ConfigDict(frozen=True)

    case_id: str
    question: str
    as_of: date | None = None
    required_phrases: list[str] = Field(min_length=1)


class ABResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    case_id: str
    tokens_off: int
    tokens_on: int
    retention_off: float
    retention_on: float

    @property
    def savings(self) -> float:
        if self.tokens_off == 0:
            return 0.0
        return 1.0 - self.tokens_on / self.tokens_off


class ABReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    token_budget: int
    results: list[ABResult]

    @property
    def mean_savings(self) -> float:
        if not self.results:
            return 0.0
        return sum(result.savings for result in self.results) / len(self.results)

    @property
    def retention_intact(self) -> bool:
        return all(result.retention_on >= result.retention_off for result in self.results)


def load_ab_cases(path: Path) -> list[ABCase]:
    """One JSON case per line — same convention as every AgentProof dataset."""
    return [
        ABCase.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _synthesis_prompt(state: GroundState) -> str:
    user_messages = [message for message in state.messages if message.role == "user"]
    return user_messages[-1].content if user_messages else ""


def _retention(prompt: str, phrases: list[str]) -> float:
    lowered = prompt.lower()
    return sum(1 for phrase in phrases if phrase.lower() in lowered) / len(phrases)


def run_ab(
    cases: list[ABCase],
    *,
    retriever: TemporalRetriever,
    embedder: EmbeddingClient,
    pruner: QueryAwarePruner,
) -> ABReport:
    """Run every case compressed and uncompressed; measure prompts, not vibes."""
    results: list[ABResult] = []
    for case in cases:
        prompts: dict[bool, str] = {}
        for compressed in (False, True):
            machine = build_pipeline(
                retriever=retriever,
                model=MockModelClient([ModelResponse(text="ok")]),
                embedder=embedder,
                web_executor=ToolExecutor(
                    ToolRegistry([make_web_search_tool()]), MockTransport({"web_search": []})
                ),
                pruner=pruner if compressed else None,
            )
            state = machine.run(GroundState(query=case.question, as_of=case.as_of))
            prompts[compressed] = _synthesis_prompt(state)
        results.append(
            ABResult(
                case_id=case.case_id,
                tokens_off=estimate_tokens(prompts[False]),
                tokens_on=estimate_tokens(prompts[True]),
                retention_off=_retention(prompts[False], case.required_phrases),
                retention_on=_retention(prompts[True], case.required_phrases),
            )
        )
    return ABReport(token_budget=pruner.token_budget, results=results)


def render_ab_scorecard(report: ABReport) -> str:
    """The diff scorecard: compressed vs uncompressed, side by side."""
    header = f"A/B compression scorecard (budget: {report.token_budget} tokens)"
    rule = "-" * 78
    lines = [
        header,
        rule,
        f"{'case':<28} {'tokens off->on':>16} {'saved':>7} {'retention off->on':>20}",
        rule,
    ]
    for result in report.results:
        lines.append(
            f"{result.case_id:<28} {result.tokens_off:>7} -> {result.tokens_on:<6}"
            f" {result.savings:>6.0%} {result.retention_off:>11.2f} -> {result.retention_on:.2f}"
        )
    lines.append(rule)
    verdict = "intact" if report.retention_intact else "DEGRADED (see rows above)"
    lines.append(f"mean savings: {report.mean_savings:.0%}   retention: {verdict}")
    return "\n".join(lines)

"""The web-search fallback step: the last resort behind AgentProof's airlock.

When the corpus cannot answer even after reformulation, the pipeline falls
back to live search — but never touches the network directly. The call goes
through AgentProof's full tool machinery: registry validates the arguments,
the transport executes (Tavily live, MockTransport in tests), the observation
gate validates the payload, and fault attribution retries transient failures.
A failed search comes back as a structured error result, not an exception —
the pipeline then answers honestly instead of crashing.
"""

from agentproof import ToolCall
from agentproof.tools.executor import ToolExecutor
from agentproof.tools.registry import Tool
from pydantic import BaseModel, Field

from groundproof.steps.state import GroundState


class SearchArgs(BaseModel):
    query: str
    max_results: int = 5


class SearchResult(BaseModel):
    title: str = ""
    url: str = ""
    content: str = ""


class SearchResults(BaseModel):
    results: list[SearchResult] = Field(default_factory=list)


def make_web_search_tool() -> Tool:
    """The web_search declaration: argument gate in, observation gate out."""
    return Tool(
        name="web_search",
        description="Live web search for facts the corpus cannot answer.",
        input_model=SearchArgs,
        output_model=SearchResults,
    )


class WebSearchStep:
    name = "web_search"

    def __init__(self, executor: ToolExecutor) -> None:
        self._executor = executor

    def run(self, state: GroundState) -> GroundState:
        call = ToolCall(
            id=f"web-{state.retrieval_attempts}",
            name="web_search",
            arguments={"query": state.effective_query},
        )
        result = self._executor.execute_call(call)
        state.tool_results.append(result)
        state.web_evidence = None if result.is_error else result.output
        return state

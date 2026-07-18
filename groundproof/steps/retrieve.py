"""The retrieval step: temporal retrieval + supersedence, one machine stage.

Runs the P2 machinery for the effective query (reformulated if the corrective
loop came back around) — or once per sub-question on the multi-hop route,
merging by chunk_id and keeping each chunk's best score. The supersedence
resolver then splits the merged evidence into current truth and dated history
before anything downstream sees it.
"""

from groundproof.retrieval.supersedence import resolve_supersedence
from groundproof.retrieval.temporal import RankedChunk, TemporalRetriever
from groundproof.steps.state import GroundState


class RetrieveStep:
    name = "retrieve"

    def __init__(self, retriever: TemporalRetriever, top_k: int = 5) -> None:
        self._retriever = retriever
        self._top_k = top_k

    def run(self, state: GroundState) -> GroundState:
        queries = (
            state.sub_questions if state.route == "multi_hop" else [state.effective_query]
        )
        merged: dict[str, RankedChunk] = {}
        for query in queries:
            for item in self._retriever.retrieve(query, as_of=state.as_of, top_k=self._top_k):
                held = merged.get(item.chunk.chunk_id)
                if held is None or item.score > held.score:
                    merged[item.chunk.chunk_id] = item
        ranked = sorted(merged.values(), key=lambda item: (-item.score, item.chunk.chunk_id))

        resolved = resolve_supersedence(ranked)
        state.evidence = resolved.current
        state.evidence_history = resolved.history
        state.retrieval_attempts += 1
        return state

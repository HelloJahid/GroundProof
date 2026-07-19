"""The compression step: prune evidence to budget before synthesis sees it.

Sits between the grader gate and synthesis — only strong evidence is worth
compressing. Scores against ``state.query``, the user's ORIGINAL question
(the same principle the grader learned the hard way: derived queries must
never decide what evidence deserves). With no pruner injected, the step is a
recorded no-op — that switch is exactly what the A/B harness flips.
"""

from groundproof.compress.pruner import QueryAwarePruner
from groundproof.steps.state import GroundState


class CompressStep:
    name = "compress"

    def __init__(self, pruner: QueryAwarePruner | None = None) -> None:
        self._pruner = pruner

    def run(self, state: GroundState) -> GroundState:
        if self._pruner is not None and state.evidence:
            state.compressed = self._pruner.compress(state.query, state.evidence)
        return state

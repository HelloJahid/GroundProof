"""The grader gate as a machine step: writes the verdict the router acts on.

Grades against ``state.query`` — the user's ORIGINAL question — never the
reformulated retrieval query. Reformulation exists to find evidence, not to
lower the bar that evidence must clear.
"""

from groundproof.grading.grader import EvidenceGrader
from groundproof.steps.state import GroundState


class GradeStep:
    name = "grade"

    def __init__(self, grader: EvidenceGrader | None = None) -> None:
        self._grader = grader or EvidenceGrader()

    def run(self, state: GroundState) -> GroundState:
        state.grade = self._grader.grade(state.query, state.evidence)
        return state

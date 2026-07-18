"""The reformulation step: one deterministic second chance before the web.

Weak evidence often means vocabulary mismatch, not missing knowledge — the
user says "is X gone", the changelog says "X was removed". The corrective
move is to re-anchor the query in the corpus's own vocabulary. Deterministic
by design: the same weak query reformulates the same way in every run, so the
corrective trajectory is reproducible and traceable.
"""

from groundproof.steps.state import GroundState

# The corpus speaks changelog: anchor weak queries to its vocabulary.
ANCHOR_TERMS = "python version release changelog notes"


class ReformulateStep:
    name = "reformulate"

    def run(self, state: GroundState) -> GroundState:
        state.active_query = f"{state.query} {ANCHOR_TERMS}"
        return state

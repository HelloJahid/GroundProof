"""The adaptive router step: how hard is this question, retrieval-wise?

Classification is deterministic rules, not a model call — the same routing
pattern AgentProof's ``react_router`` uses, applied to retrieval difficulty:

- ``none``: smalltalk/greetings — retrieval would be noise; go straight to
  synthesis.
- ``multi_hop``: several questions in one (multiple '?' segments) — decompose
  into sub-questions and retrieve for each.
- ``one_shot``: everything else — a single retrieval pass.
"""

import re

from groundproof.steps.state import GroundState

_WORD = re.compile(r"[a-z']+")
_GREETINGS = frozenset({"hi", "hello", "hey", "yo", "greetings", "thanks", "thank"})


class AdaptiveRouterStep:
    name = "route"

    def run(self, state: GroundState) -> GroundState:
        words = _WORD.findall(state.query.lower())
        parts = [part.strip() for part in state.query.split("?") if part.strip()]

        if words and words[0] in _GREETINGS and len(words) <= 4:
            state.route = "none"
        elif len(parts) >= 2:
            state.route = "multi_hop"
            state.sub_questions = [f"{part}?" for part in parts]
        else:
            state.route = "one_shot"
        return state

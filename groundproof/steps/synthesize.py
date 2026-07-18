"""The synthesis step: dated evidence in, temporally-footed answer out.

Three honest modes, chosen from what the state actually holds:

- direct (route "none"): no evidence needed, the question goes to the model;
- evidence: every chunk arrives labelled with its observed_at date and doc_id,
  superseded history arrives labelled "true until X" — and the instructions
  REQUIRE dated citations, so the answer discloses its temporal footing;
- honest failure: weak evidence and no usable web fallback → a fixed refusal,
  produced WITHOUT a model call. Never ask a model to answer from evidence
  the gate already rejected.
"""

from agentproof import ModelClient

from groundproof.steps.state import GroundState

INSTRUCTIONS = (
    "You are GroundProof, a retrieval agent whose answers are temporally footed. "
    "Answer ONLY from the evidence provided. Every factual claim must carry a dated "
    "citation in the form (YYYY-MM-DD, source-id). If the evidence includes superseded "
    "history, state what changed and when. If the evidence cannot answer the question, "
    "say so plainly."
)

HONEST_FAILURE = (
    "No reliable information found: retrieval produced weak evidence, reformulation "
    "did not improve it, and live web search returned nothing usable. Rather than "
    "guess, I am declining to answer."
)


def _evidence_prompt(state: GroundState) -> str:
    as_of = state.as_of.isoformat() if state.as_of else "latest available knowledge"
    lines = [f"Question (answer as of {as_of}): {state.query}", "", "Evidence (dated):"]
    lines.extend(
        f"[{index}] ({item.chunk.observed_at}, {item.chunk.doc_id}) {item.chunk.text}"
        for index, item in enumerate(state.evidence, start=1)
    )
    if state.evidence_history:
        lines.append("")
        lines.append("Superseded history (each item was true until the date shown):")
        lines.extend(
            f"[H{index}] (true until {record.superseded_on}, superseded by "
            f"{record.superseded_by}) ({record.item.chunk.observed_at}, "
            f"{record.item.chunk.doc_id}) {record.item.chunk.text}"
            for index, record in enumerate(state.evidence_history, start=1)
        )
    return "\n".join(lines)


def _web_prompt(state: GroundState) -> str:
    return (
        f"Question: {state.query}\n\n"
        "The internal corpus could not answer this. Live web search results (retrieved "
        "moments ago; cite as 'live web search' with today's context):\n"
        f"{state.web_evidence}"
    )


class SynthesizeStep:
    name = "synthesize"

    def __init__(self, model: ModelClient) -> None:
        self._model = model

    def run(self, state: GroundState) -> GroundState:
        if state.route == "none":
            prompt = state.query
        elif state.web_evidence:
            prompt = _web_prompt(state)
        elif state.grade is not None and state.grade.verdict == "weak":
            state.final_answer = HONEST_FAILURE
            return state
        elif state.evidence:
            prompt = _evidence_prompt(state)
        else:
            state.final_answer = HONEST_FAILURE
            return state

        state.add_message("user", prompt)
        response = self._model.complete(INSTRUCTIONS, state.messages)
        state.input_tokens += response.input_tokens
        state.output_tokens += response.output_tokens
        state.final_answer = response.text or ""
        state.add_message("assistant", state.final_answer)
        return state

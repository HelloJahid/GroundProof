"""The cockpit: a GroundProof-aware trace viewer.

``python -m groundproof.cockpit <trace.jsonl>``

AgentProof's stock viewer shows generic agent state; the cockpit renders what
THIS pipeline is doing to evidence — chunks being dated, graded, superseded,
and pruned — step by step, from the trace file alone (never the live run).
"""

import sys
from pathlib import Path

from agentproof.trace.replay import RunReplay

from groundproof.evals.temporal import load_ground_replay
from groundproof.steps.state import GroundState


def _first_line(text: str, width: int = 48) -> str:
    line = text.splitlines()[0] if text else ""
    return line[:width]


def render_replay(replay: RunReplay) -> str:
    lines: list[str] = [f"=== GroundProof cockpit — run {replay.run_id} ==="]
    lines.append(f'question: "{replay.query}"')

    for index, step in enumerate(replay.steps, start=1):
        state = step.state
        if not isinstance(state, GroundState):
            lines.append(f"[{index}] {step.name}")
            continue
        head = f"[{index}] {step.name} ({step.duration_ms:.1f} ms)"
        if step.name == "route":
            as_of = state.as_of.isoformat() if state.as_of else "latest knowledge"
            lines.append(f"{head}: {state.route}   (as-of: {as_of})")
            if state.sub_questions:
                lines.extend(f"      sub-question: {q}" for q in state.sub_questions)
        elif step.name == "retrieve":
            lines.append(
                f"{head}: attempt {state.retrieval_attempts}, "
                f"{len(state.evidence)} current + {len(state.evidence_history)} superseded"
            )
            lines.extend(
                f"      {item.score:.3f} (sim {item.similarity:.2f} rec {item.recency:.2f})"
                f"  {item.chunk.observed_at}  {item.chunk.doc_id:<24} "
                f"{_first_line(item.chunk.text)}"
                for item in state.evidence[:5]
            )
            lines.extend(
                f"      history: {record.item.chunk.doc_id} was current until "
                f"{record.superseded_on} (superseded by {record.superseded_by})"
                for record in state.evidence_history[:3]
            )
        elif step.name == "grade" and state.grade is not None:
            grade = state.grade
            lines.append(
                f"{head}: {grade.verdict.upper()}  strength={grade.strength:.2f}"
                f"  (similarity {grade.top_similarity:.2f}, overlap {grade.keyword_overlap:.2f})"
            )
        elif step.name == "reformulate":
            lines.append(f'{head}: retry with "{state.active_query}"')
        elif step.name == "compress":
            if state.compressed is None:
                lines.append(f"{head}: off (pass-through)")
            else:
                c = state.compressed
                lines.append(
                    f"{head}: {c.tokens_before} -> {c.tokens_after} tokens "
                    f"({c.savings:.0%} saved), kept {c.sentences_kept}/{c.sentences_total} "
                    "sentences, attribution preserved"
                )
        elif step.name == "web_search":
            result = state.tool_results[-1] if state.tool_results else None
            outcome = "error" if (result and result.is_error) else "ok"
            lines.append(f"{head}: {outcome}")
        elif step.name == "synthesize":
            lines.append(head)
        else:
            lines.append(head)

    lines.append("")
    lines.append(f"outcome: {replay.outcome}")
    lines.append(f"answer:\n{replay.final_answer or '(none)'}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    if len(argv) != 1:
        print("usage: python -m groundproof.cockpit <trace.jsonl>")
        return 2
    print(render_replay(load_ground_replay(Path(argv[0]))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

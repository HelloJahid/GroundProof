"""GroundProof interactive demo: ``streamlit run demo/app.py``.

Presentation only — all behavior lives in demo/app_runners.py (tested,
streamlit-free). Three tabs: Ask (one question, full pipeline visible),
Time travel (same question, two as-of moments side by side), and A/B
compression (the token receipts). Keyless by default; the sidebar Live
toggle switches to the real model + web search using keys from .env.
"""

import sys
from pathlib import Path

# `streamlit run demo/app.py` puts demo/ (the script's dir) on sys.path, not the
# repo root — so the `demo` package itself is not importable without this.
_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import streamlit as st  # noqa: E402

from demo.app_runners import (  # noqa: E402
    ab_rows,
    build_ab_report,
    build_resources,
    key_status,
    run_question,
)
from demo.ask import parse_as_of  # noqa: E402
from demo.env import load_env  # noqa: E402
from groundproof.cockpit import render_replay  # noqa: E402
from groundproof.evals.temporal import load_ground_replay  # noqa: E402
from groundproof.steps import GroundState  # noqa: E402

st.set_page_config(page_title="GroundProof", page_icon="⏳", layout="wide")
load_env()

cached_resources = st.cache_resource(build_resources)
resources = cached_resources()


@st.cache_resource
def cached_ab_report(_resources):
    # Leading underscore: streamlit must not try to hash the (unhashable)
    # Resources object — it is a cache_resource singleton already.
    return build_ab_report(_resources)

# ---------------------------------------------------------------- sidebar ----

with st.sidebar:
    st.title("⏳ GroundProof")
    st.markdown("**RAG that knows when facts expire** — and pays only for the context it needs.")
    live = st.toggle("Live mode (Anthropic + Tavily)", value=False)
    keys = key_status()
    if live and not keys["ANTHROPIC_API_KEY"]:
        st.warning("ANTHROPIC_API_KEY not set — answers fall back to the offline synthesizer.")
    if live and not keys["TAVILY_API_KEY"]:
        st.warning("TAVILY_API_KEY not set — web fallback stays mocked.")
    live_model = live and keys["ANTHROPIC_API_KEY"]
    mode_badge = "🟢 Live (Anthropic)" if live_model else "⚪ Offline (extractive, keyless)"
    st.markdown(f"**Mode:** {mode_badge}")
    st.caption(
        f"{resources.chunk_count} chunks indexed (CPython 3.8–3.14 changelogs). "
        "Embeddings are the deterministic mock in both modes — the live embedding "
        "provider is a port, not yet wired — so retrieval is bag-of-words either way."
    )


def render_run(state: GroundState, trace_path, *, show_trace: bool = True) -> None:
    """The full pipeline story for one run, top to bottom."""
    # Trajectory
    st.markdown(" → ".join(f"`{record.step}`" for record in state.history))
    route_line = f"route: **{state.route}** · retrieval attempts: {state.retrieval_attempts}"
    if state.as_of:
        route_line += f" · as-of: **{state.as_of}**"
    st.caption(route_line)
    if state.sub_questions:
        st.caption("sub-questions: " + " | ".join(state.sub_questions))
    if state.active_query:
        st.caption(f'reformulated retrieval query: "{state.active_query}"')

    # Grade
    if state.grade is not None:
        grade = state.grade
        verdict = "STRONG ✅" if grade.verdict == "strong" else "WEAK ⚠️"
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Grader verdict", verdict)
        col2.metric("Strength", f"{grade.strength:.2f}")
        col3.metric("Top similarity", f"{grade.top_similarity:.2f}")
        col4.metric("Keyword overlap", f"{grade.keyword_overlap:.2f}")
        st.progress(min(grade.strength, 1.0))
    if state.web_evidence:
        st.info("Corpus evidence was weak — the web-search fallback fired.")

    # Evidence
    if state.evidence:
        st.subheader("Evidence (dated)")
        for index, item in enumerate(state.evidence, start=1):
            with st.container(border=True):
                st.markdown(f"**{index}.** `{item.chunk.doc_id}` · 📅 {item.chunk.observed_at}")
                st.caption(
                    f"score {item.score:.3f} — similarity {item.similarity:.3f}, "
                    f"recency {item.recency:.3f}"
                )
                with st.expander("chunk text"):
                    st.text(item.chunk.text)
    if state.evidence_history:
        with st.expander(f"Superseded history ({len(state.evidence_history)})"):
            for record in state.evidence_history:
                st.markdown(
                    f"- `{record.item.chunk.doc_id}` was current until "
                    f"**{record.superseded_on}** (superseded by `{record.superseded_by}`)"
                )

    # Compression
    if state.compressed is not None:
        compressed = state.compressed
        st.subheader("Compression (Hook B)")
        col1, col2, col3 = st.columns(3)
        col1.metric("Prompt evidence tokens", compressed.tokens_after, f"-{compressed.savings:.0%}")
        col2.metric("Before", compressed.tokens_before)
        col3.metric("Sentences kept", f"{compressed.sentences_kept}/{compressed.sentences_total}")
        with st.expander("compressed evidence block (attribution preserved)"):
            st.code(compressed.text)
    elif state.evidence:
        st.caption("compression disabled for this run")

    # Answer
    st.subheader("Answer")
    with st.container(border=True):
        st.caption(mode_badge)
        st.markdown(state.final_answer or "_(no answer)_")
    st.caption(f"model tokens: {state.input_tokens} in / {state.output_tokens} out")

    # Trace
    if show_trace:
        with st.expander("Flight recorder (full trace)"):
            st.code(render_replay(load_ground_replay(trace_path)))
            st.download_button(
                "Download trace",
                trace_path.read_bytes(),
                file_name=trace_path.name,
                key=f"dl-{trace_path.name}",
            )


tab_ask, tab_travel, tab_ab = st.tabs(["Ask", "Time travel", "A/B compression"])

# --------------------------------------------------------------- tab: ask ----

with tab_ask:
    with st.form("ask"):
        question = st.text_input("Question", value="is the distutils package removed")
        col1, col2 = st.columns([1, 1])
        as_of_raw = col1.text_input(
            "As of", value="2024-06", help="YYYY-MM, YYYY-MM-DD, 'today', or blank = latest"
        )
        compress = col2.checkbox("Query-aware compression (Hook B)", value=True)
        submitted = st.form_submit_button("Ask")
    if submitted:
        try:
            as_of = parse_as_of(as_of_raw.strip() or None)
        except ValueError as exc:
            st.error(f"Bad as-of date: {exc}")
        else:
            with st.spinner("Running the corrective pipeline..."):
                # Key must differ from the form's widget key ("ask") — widget
                # keys own their session_state slots.
                st.session_state["ask_result"] = run_question(
                    question, as_of, live=live, compress=compress, resources=resources
                )
    if "ask_result" in st.session_state:
        state, trace_path = st.session_state["ask_result"]
        render_run(state, trace_path)

# ------------------------------------------------------- tab: time travel ----

with tab_travel:
    st.caption(
        "The same question, asked at two moments. The as-of filter hides everything "
        "published after each moment — two different, correctly-dated answers."
    )
    with st.form("travel"):
        travel_question = st.text_input(
            "Question", value="summary release highlights latest stable python"
        )
        col1, col2 = st.columns(2)
        as_of_a = col1.text_input("As of (left)", value="2024-06")
        as_of_b = col2.text_input("As of (right)", value="2026-01")
        travel_submitted = st.form_submit_button("Compare")
    if travel_submitted:
        try:
            parsed_a, parsed_b = parse_as_of(as_of_a.strip()), parse_as_of(as_of_b.strip())
        except ValueError as exc:
            st.error(f"Bad as-of date: {exc}")
        else:
            with st.spinner("Running the pipeline twice..."):
                st.session_state["travel_result"] = [
                    (parsed, *run_question(
                        travel_question, parsed, live=live, compress=True, resources=resources
                    ))
                    for parsed in (parsed_a, parsed_b)
                ]
    if "travel_result" in st.session_state:
        runs = st.session_state["travel_result"]
        columns = st.columns(2)
        for column, (as_of, state, _trace) in zip(columns, runs, strict=True):
            with column:
                st.subheader(f"as of {as_of}")
                for item in state.evidence[:3]:
                    st.markdown(
                        f"- `{item.chunk.doc_id}` · 📅 {item.chunk.observed_at} "
                        f"· score {item.score:.3f}"
                    )
                with st.container(border=True):
                    st.markdown(state.final_answer or "_(no answer)_")
                if state.grade:
                    st.caption(f"grade: {state.grade.verdict} ({state.grade.strength:.2f})")
        docs_a = {item.chunk.doc_id for item in runs[0][1].evidence}
        docs_b = {item.chunk.doc_id for item in runs[1][1].evidence}
        only_a, only_b = sorted(docs_a - docs_b), sorted(docs_b - docs_a)
        if only_a or only_b:
            st.caption(
                f"📅 evidence only at {runs[0][0]}: {', '.join(only_a) or '—'} · "
                f"only at {runs[1][0]}: {', '.join(only_b) or '—'}"
            )

# ------------------------------------------------------------- tab: a/b ------

with tab_ab:
    st.caption(
        "Every golden case runs through the same pipeline twice — compressor off, then on. "
        "Fully mocked and deterministic; independent of the Live toggle."
    )
    if st.button("Run A/B harness"):
        with st.spinner("Running golden cases, compressed and uncompressed..."):
            st.session_state["ab_result"] = cached_ab_report(resources)
    if "ab_result" in st.session_state:
        report = st.session_state["ab_result"]
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Mean savings", f"{report.mean_savings:.0%}")
        col2.metric("Retention", "intact ✅" if report.retention_intact else "DEGRADED ❌")
        col3.metric("Token budget", report.token_budget)
        col4.metric("Golden cases", len(report.results))
        st.dataframe(
            ab_rows(report),
            use_container_width=True,
            column_config={
                "saved": st.column_config.ProgressColumn(
                    "saved", format="percent", min_value=0.0, max_value=1.0
                ),
            },
        )

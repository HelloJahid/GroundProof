# Changelog

All notable changes to GroundProof are documented here.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning: [SemVer](https://semver.org/).

## [Unreleased]

### Added

- Streamlit demo app (`streamlit run demo/app.py`): Ask / Time travel / A/B compression
  tabs with the evidence pipeline visible — offline by default, sidebar Live toggle
  using `.env` keys.

## [0.1.0] - 2026-07-19

First public release — the complete stage-3 build.

### Added

- **Time-aware retrieval (Hook A):** every chunk carries a required `observed_at`;
  queries execute as of a moment (`--as-of`) with in-store filtering and a tunable
  recency prior; conflicting facts resolved by deterministic supersedence rules —
  later date wins, older kept as dated history.
- **Corrective spine:** adaptive router (none / one-shot / multi-hop), rule-based
  retrieval grader gate (always grading the user's original question), deterministic
  reformulation, and web-search fallback through AgentProof's tool airlock — with an
  honest refusal when nothing reliable is found.
- **Query-aware compression (Hook B):** sentence-level scoring and token-budget
  knapsack packing with per-sentence source attribution preserved; A/B harness
  receipts on the committed corpus: ~62% mean prompt-token savings, evidence
  retention unchanged.
- **Eval CI gate:** temporal golden pairs (same question, two as-of dates, different
  expected documents), trace-judged rule checks, stale-fact tripwire that goes red
  naming the case when a superseding document changes an answer; wired into CI as
  lint → tests → eval gate.
- **Demo CLI + cockpit:** `python -m demo.ask "…" --as-of 2024-06` runs keyless from
  a clone (offline extractive synthesizer; `--live` swaps in the real model + search);
  `python -m groundproof.cockpit <trace>` replays any run from its flight-recorder
  trace — chunks dated, graded, superseded, pruned.
- **Committed corpus:** CPython "What's New" documents (3.8–3.14), dates sourced from
  a structured release record, chunked along reST section structure.

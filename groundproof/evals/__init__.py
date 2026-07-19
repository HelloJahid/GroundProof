"""Temporal golden pairs, A/B compression comparison, and CI gate wiring (P4-P5)."""

from groundproof.evals.ab import (
    ABCase,
    ABReport,
    ABResult,
    load_ab_cases,
    render_ab_scorecard,
    run_ab,
)
from groundproof.evals.temporal import (
    DATED_GROUNDEDNESS_RUBRIC,
    ExpectedSourceCheck,
    GroundednessJudge,
    PhrasesCheck,
    TemporalCase,
    TemporalIntegrityCheck,
    load_ground_replay,
    load_temporal_cases,
    run_temporal_suite,
)

__all__ = [
    "DATED_GROUNDEDNESS_RUBRIC",
    "ABCase",
    "ABReport",
    "ABResult",
    "ExpectedSourceCheck",
    "GroundednessJudge",
    "PhrasesCheck",
    "TemporalCase",
    "TemporalIntegrityCheck",
    "load_ab_cases",
    "load_ground_replay",
    "load_temporal_cases",
    "render_ab_scorecard",
    "run_ab",
    "run_temporal_suite",
]

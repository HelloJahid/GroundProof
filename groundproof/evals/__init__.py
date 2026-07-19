"""Temporal golden pairs, A/B compression comparison, and CI gate wiring (P4-P5)."""

from groundproof.evals.ab import (
    ABCase,
    ABReport,
    ABResult,
    load_ab_cases,
    render_ab_scorecard,
    run_ab,
)

__all__ = [
    "ABCase",
    "ABReport",
    "ABResult",
    "load_ab_cases",
    "render_ab_scorecard",
    "run_ab",
]

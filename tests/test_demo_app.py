"""The Streamlit app's behavior layer — tested keyless, without streamlit."""

import ast
from datetime import date
from pathlib import Path

import pytest

from demo.app_runners import (
    ab_rows,
    build_ab_report,
    build_resources,
    key_status,
    run_question,
)


def test_app_runners_never_imports_streamlit():
    """Static guard: the behavior layer must stay streamlit-free. Checked via
    the AST (not sys.modules, which other tests legitimately pollute)."""
    tree = ast.parse(Path("demo/app_runners.py").read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert "streamlit" not in imported


class TestKeyStatus:
    def test_reports_presence_and_absence(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        monkeypatch.delenv("TAVILY_API_KEY", raising=False)
        status = key_status()
        assert status == {"ANTHROPIC_API_KEY": True, "TAVILY_API_KEY": False}

    def test_empty_value_counts_as_missing(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "")
        assert key_status()["ANTHROPIC_API_KEY"] is False


@pytest.fixture(scope="module")
def resources():
    return build_resources()


class TestRunQuestion:
    def test_offline_run_on_real_corpus(self, resources, tmp_path):
        state, trace_path = run_question(
            "is the distutils package removed",
            date(2024, 6, 1),
            live=False,
            compress=True,
            resources=resources,
            trace_dir=tmp_path,
        )
        assert state.grade is not None and state.grade.verdict == "strong"
        assert state.final_answer is not None
        assert "(2023-10-02, python-whatsnew-3.12)" in state.final_answer
        assert state.compressed is not None and state.compressed.savings > 0
        assert trace_path.exists() and trace_path.name.startswith("app-")

    def test_compression_off_leaves_compressed_none(self, resources, tmp_path):
        state, _trace = run_question(
            "is the distutils package removed",
            date(2024, 6, 1),
            live=False,
            compress=False,
            resources=resources,
            trace_dir=tmp_path,
        )
        assert state.compressed is None
        assert state.final_answer is not None


class TestABRows:
    def test_rows_mirror_the_report(self, resources):
        report = build_ab_report(resources)
        rows = ab_rows(report)
        assert len(rows) == len(report.results)
        first, result = rows[0], report.results[0]
        assert first["case"] == result.case_id
        assert first["tokens (off)"] == result.tokens_off
        assert first["saved"] == pytest.approx(result.savings)
        assert set(first) == {
            "case",
            "tokens (off)",
            "tokens (on)",
            "saved",
            "retention off",
            "retention on",
        }

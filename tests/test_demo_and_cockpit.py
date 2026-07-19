"""P6: the demo CLI runs keyless from a clone; the cockpit renders the story."""

from datetime import date

from agentproof import Message

from demo.ask import main as ask_main
from demo.ask import parse_as_of
from demo.offline_model import OfflineSynthesizer
from groundproof.cockpit import render_replay
from groundproof.evals.temporal import load_ground_replay


class TestParseAsOf:
    def test_month_precision_becomes_first_of_month(self):
        assert parse_as_of("2024-06") == date(2024, 6, 1)

    def test_full_date_and_none(self):
        assert parse_as_of("2023-10-02") == date(2023, 10, 2)
        assert parse_as_of(None) is None


class TestOfflineSynthesizer:
    def test_extracts_dated_evidence_lines(self):
        prompt = (
            "Question (answer as of 2024-06-01): is distutils removed\n\n"
            "Evidence (dated):\n"
            "(2023-10-02, python-whatsnew-3.12):\n"
            "[1] (2023-10-02, python-whatsnew-3.12) The distutils package was removed.\n"
        )
        response = OfflineSynthesizer().complete("", [Message(role="user", content=prompt)])
        assert response.text is not None
        assert "(2023-10-02, python-whatsnew-3.12)" in response.text
        assert response.input_tokens > 0

    def test_chitchat_prompt_gets_greeting(self):
        response = OfflineSynthesizer().complete("", [Message(role="user", content="hello!")])
        assert response.text is not None and "offline" in response.text


class TestAskEndToEnd:
    def test_offline_ask_from_real_corpus(self, tmp_path, capsys):
        code = ask_main(
            [
                "is the distutils package removed",
                "--as-of",
                "2024-06",
                "--trace-dir",
                str(tmp_path),
            ]
        )
        out = capsys.readouterr().out
        assert code == 0
        assert "GroundProof cockpit" in out
        assert "route" in out and "grade" in out and "compress" in out
        assert "(2023-10-02, python-whatsnew-3.12)" in out  # dated citation in answer
        traces = list(tmp_path.glob("ask-*.trace.jsonl"))
        assert len(traces) == 1

        # The cockpit re-renders the same story from the trace alone.
        rendered = render_replay(load_ground_replay(traces[0]))
        assert "STRONG" in rendered
        assert "tokens" in rendered and "saved" in rendered

    def test_time_travel_gives_different_answers(self, tmp_path, capsys):
        for as_of in ("2024-06", "2026-01"):
            ask_main(
                [
                    "summary release highlights latest stable python",
                    "--as-of",
                    as_of,
                    "--trace-dir",
                    str(tmp_path),
                ]
            )
        out = capsys.readouterr().out
        assert "python-whatsnew-3.12" in out  # the 2024 answer's evidence
        assert "python-whatsnew-3.14" in out  # the 2026 answer's evidence

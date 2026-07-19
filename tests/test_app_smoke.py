"""Smoke-execute the Streamlit app script itself.

Skipped wherever streamlit isn't installed (CI installs only [dev]), so the
suite stays streamlit-free there — but locally, with the [demo] extra, this
actually RUNS demo/app.py the way `streamlit run` does. It exists because an
HTTP-200 boot check once passed while the script itself crashed on import:
streamlit only executes the script when a session connects.
"""

import pytest

streamlit_testing = pytest.importorskip("streamlit.testing.v1")


def test_app_script_executes_without_exceptions():
    app = streamlit_testing.AppTest.from_file("demo/app.py", default_timeout=60)
    app.run()
    assert not app.exception, f"app raised: {app.exception}"
    # The three tabs and the live toggle exist on first render.
    assert len(app.tabs) == 3
    assert len(app.toggle) == 1
    assert app.toggle[0].value is False  # offline by default


def test_ab_button_runs_the_harness():
    """Click the A/B tab's button — catches cache-hashing errors on the
    Resources argument that only fire when the harness actually runs."""
    app = streamlit_testing.AppTest.from_file("demo/app.py", default_timeout=120)
    app.run()
    app.button[2].click()  # buttons in creation order: ask submit, travel submit, A/B
    app.run()
    assert not app.exception, f"app raised: {app.exception}"
    assert "ab_result" in app.session_state
    assert app.session_state["ab_result"].mean_savings > 0.4


def test_ask_form_submits_and_renders_a_run():
    """Submit the Ask form for real — catches widget/session-state collisions
    that first-render smoke checks can't see."""
    app = streamlit_testing.AppTest.from_file("demo/app.py", default_timeout=120)
    app.run()
    app.button[0].click()  # the Ask form's submit button
    app.run()
    assert not app.exception, f"app raised: {app.exception}"
    assert "ask_result" in app.session_state
    state, _trace = app.session_state["ask_result"]
    assert state.final_answer is not None
    assert "(2023-10-02, python-whatsnew-3.12)" in state.final_answer

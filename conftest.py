"""Root conftest: makes the repo root importable during test collection.

The ``demo`` package is deliberately NOT installed (only ``groundproof``
ships), so tests import it from the repo root. A root-level conftest is the
standard pytest mechanism that puts this directory on ``sys.path`` regardless
of how pytest is invoked (``pytest`` on CI vs ``python -m pytest`` locally —
only the latter adds the CWD on its own, which is how this difference hid).
"""

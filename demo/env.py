"""Tiny .env loader (the AgentProof demo pattern): KEY=VALUE lines into os.environ.

Only the demo edge reads keys; the library and test suite never do.
"""

import os
from pathlib import Path


def load_env(path: Path | str = ".env") -> None:
    path = Path(path)
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))

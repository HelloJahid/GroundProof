# Contributing to GroundProof

GroundProof is stage 3 of the Proof series — a portfolio project built to accompany a
published article. It is maintained, but not aiming to grow a large contributor base.
Issues (bugs, questions, corrections) and small focused PRs are welcome.

## Dev setup

```bash
git clone https://github.com/HelloJahid/GroundProof.git
cd GroundProof
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

Python 3.11+ required. Everything runs offline — no API keys, no network: the corpus
is committed and every external boundary (embeddings, vector store, model, web search)
sits behind an injectable port with a first-class mock.

## The three checks

Every change must keep all three green (CI runs exactly these):

```bash
ruff check .                   # lint
pytest                         # full mocked test suite
python -m groundproof.evals    # the eval gate: temporal golden pairs + A/B receipts
```

If your change alters what a golden question retrieves, the eval gate will go red and
name the case — that is the system working. Update `datasets/*.jsonl` only when the
new behavior is genuinely the intended truth, and say so in the PR.

## Ground rules

- `agentproof` is intentionally a **pinned git dependency** — never vendor its code;
  fixes to the runtime belong in [AgentProof](https://github.com/HelloJahid/AgentProof).
- No agent/RAG frameworks (LangChain, LlamaIndex, etc.) in core machinery.
- Tests never touch the network and never need keys. New external boundaries get a
  port + mock, and mocks are held to the live implementation by contract tests.
- Typed errors only, extending `GroundProofError`.

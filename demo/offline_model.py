"""OfflineSynthesizer: a keyless, deterministic ModelClient for the demo.

Not a language model and not pretending to be one: it EXTRACTS the top dated
evidence lines from the synthesis prompt and assembles a labelled, citation-
carrying answer. The point of the demo is the pipeline (routing, time travel,
grading, compression, traces) — this stand-in lets all of it run from a fresh
clone with no API key, while `--live` swaps in the real AnthropicClient
through the same ModelClient port.
"""

import re

from agentproof import Message, ModelResponse

from groundproof.compress.pruner import estimate_tokens
from groundproof.grading.grader import content_words

_CITED_LINE = re.compile(r"\(\d{4}-\d{2}-\d{2}, [^)]+\)")
_QUESTION_LINE = re.compile(r"^Question[^:]*:\s*(.+)$", re.MULTILINE)


class OfflineSynthesizer:
    """ModelClient port, fulfilled without a network or a key."""

    def complete(
        self,
        instructions: str,
        messages: list[Message],
        tools: list[dict] | None = None,
    ) -> ModelResponse:
        prompt = messages[-1].content if messages else ""
        cited: list[str] = []
        # Compressed evidence puts a citation header on its own line with the
        # sentences below it; uncompressed puts citation and text on one line.
        # Attribute every sentence to its nearest header so ranking can pick
        # the best fact, not just the first line.
        current_header: str | None = None
        for raw in prompt.splitlines():
            line = raw.strip()
            if not line:
                continue
            if _CITED_LINE.search(line):
                if line.endswith("):"):
                    current_header = line
                else:
                    cited.append(line)
                continue
            if current_header is not None:
                cited.append(f"{current_header} {line}")
        if cited:
            # Surface the snippets that overlap the question most (stable order
            # on ties), so the extractive answer leads with the relevant fact.
            match = _QUESTION_LINE.search(prompt)
            keywords = content_words(match.group(1)) if match else []
            cited.sort(
                key=lambda s: -sum(1 for word in keywords if word in s.lower())
            )
            top = "\n".join(f"  {line[:220]}" for line in cited[:3])
            text = (
                "[offline extractive answer — run with --live and an API key for "
                "model synthesis]\nMost relevant dated evidence:\n" + top
            )
        elif "Live web evidence" in prompt or "live web search" in prompt:
            text = "[offline] The corpus could not answer; web fallback evidence was found."
        else:
            text = (
                "[offline] Hello! Ask me about Python releases, e.g. "
                '"is the distutils package removed" --as-of 2024-06.'
            )
        return ModelResponse(
            text=text,
            input_tokens=estimate_tokens(prompt),
            output_tokens=estimate_tokens(text),
        )

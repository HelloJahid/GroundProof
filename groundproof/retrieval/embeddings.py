"""The embedding port: text -> vector, behind an injectable seam.

``EmbeddingClient`` is the protocol every consumer depends on; *which* provider
fulfils it is a wiring decision made at the edge, not in the pipeline.
``MockEmbeddingClient`` is the first-class test implementation: deterministic
hashed bag-of-words vectors — same text, same vector, in every process, with no
keys and no network — yet word overlap still moves cosine similarity, so
retrieval tests can assert *real rankings*, not just plumbing.
"""

import hashlib
import math
import re
from typing import Protocol, runtime_checkable


@runtime_checkable
class EmbeddingClient(Protocol):
    """Anything that can turn a batch of texts into equal-length float vectors."""

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed each text; ``result[i]`` corresponds to ``texts[i]``."""
        ...


# Dots survive tokenisation on purpose: "3.12" must stay one token so version
# numbers are first-class retrieval vocabulary in this corpus.
_TOKEN = re.compile(r"[a-z0-9_.]+")


class MockEmbeddingClient:
    """Deterministic, offline embeddings: hashed bag-of-words, L2-normalised.

    Each token is hashed into one of ``dim`` buckets via sha256 — stable across
    processes, unlike Python's builtin ``hash``, which is salted per run — and
    the bucket counts, normalised, form the vector. Texts sharing words land
    near each other in cosine space: exactly enough structure for ranking tests.
    """

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        for token in _TOKEN.findall(text.lower()):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:8], "big") % self.dim
            vector[bucket] += 1.0
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0.0:
            return vector
        return [value / norm for value in vector]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity of two equal-length vectors (0.0 if either is all-zero)."""
    if len(a) != len(b):
        raise ValueError(f"vector lengths differ: {len(a)} != {len(b)}")
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)

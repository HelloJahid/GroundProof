"""The retrieval grader gate: is this evidence strong enough to answer from?

Before the model may synthesise from retrieved chunks, a grader scores them
against the question — with RULES, not another model call. Two cheap signals:

- ``top_similarity``: the best embedding similarity among the evidence (how
  close the closest chunk is in vector space);
- ``keyword_overlap``: the fraction of the question's content words that
  appear anywhere in the evidence text (vector-blind, catches the case where
  embeddings are confidently wrong).

The blend must clear a tunable threshold or the verdict is "weak" — and weak
evidence triggers the corrective fallback instead of a hallucinated answer.
A cheap model-graded second opinion could slot in behind the same interface
later; the rule signals stay the gate either way.

Hard-won rule (caught on the real corpus): the gate always grades against the
user's ORIGINAL question. Reformulation rewrites the *retrieval* query with
corpus vocabulary — grade against that and the reformulation games its own
gate, because the anchor terms trivially overlap the corpus. For the same
reason, pass an ``embedder`` so similarity is recomputed against the graded
question instead of trusting scores computed against the reformulated one.
"""

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict

from groundproof.retrieval.embeddings import EmbeddingClient, cosine_similarity
from groundproof.retrieval.temporal import RankedChunk

_WORD = re.compile(r"[a-z0-9_.]+")
_STOPWORDS = frozenset(
    """a an the is are was were be been being am do does did has have had what when which
    who whom whose how why where in on of off for to from as at by with about into over
    under and or not no nor it its this that these those there here i you he she we they
    my your his her our their me him us them will would can could shall should may might
    must still than then so if""".split()
)

DEFAULT_THRESHOLD = 0.35
DEFAULT_SIMILARITY_WEIGHT = 0.5


def content_words(text: str) -> list[str]:
    """The words that carry meaning: lowercased, stopwords dropped."""
    return [word for word in _WORD.findall(text.lower()) if word not in _STOPWORDS]


class EvidenceGrade(BaseModel):
    """The gate's verdict, with the signals that produced it kept visible."""

    model_config = ConfigDict(frozen=True)

    strength: float
    verdict: Literal["strong", "weak"]
    top_similarity: float
    keyword_overlap: float


class EvidenceGrader:
    """Rule-based evidence grading with tunable threshold and signal blend.

    With an ``embedder``, similarity is recomputed between the graded question
    and each chunk's text; without one, the retrieval-time similarity is used
    (fine when the retrieval query IS the graded question).
    """

    def __init__(
        self,
        embedder: EmbeddingClient | None = None,
        threshold: float = DEFAULT_THRESHOLD,
        similarity_weight: float = DEFAULT_SIMILARITY_WEIGHT,
    ) -> None:
        if not 0.0 <= similarity_weight <= 1.0:
            raise ValueError(f"similarity_weight must be in [0, 1], got {similarity_weight}")
        self._embedder = embedder
        self._threshold = threshold
        self._similarity_weight = similarity_weight

    def grade(self, question: str, evidence: list[RankedChunk]) -> EvidenceGrade:
        if not evidence:
            return EvidenceGrade(
                strength=0.0, verdict="weak", top_similarity=0.0, keyword_overlap=0.0
            )
        if self._embedder is not None:
            question_vector = self._embedder.embed_texts([question])[0]
            chunk_vectors = self._embedder.embed_texts([item.chunk.text for item in evidence])
            top_similarity = max(
                cosine_similarity(question_vector, vector) for vector in chunk_vectors
            )
        else:
            top_similarity = max(item.similarity for item in evidence)
        keywords = content_words(question)
        joined = " ".join(item.chunk.text.lower() for item in evidence)
        overlap = (
            sum(1 for word in keywords if word in joined) / len(keywords) if keywords else 0.0
        )
        strength = (
            self._similarity_weight * top_similarity + (1.0 - self._similarity_weight) * overlap
        )
        verdict: Literal["strong", "weak"] = "strong" if strength >= self._threshold else "weak"
        return EvidenceGrade(
            strength=strength,
            verdict=verdict,
            top_similarity=top_similarity,
            keyword_overlap=overlap,
        )

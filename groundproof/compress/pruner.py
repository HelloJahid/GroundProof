"""The query-aware pruner: context must earn its place in the prompt (Hook B).

Between retrieval and synthesis, surviving chunks are split into sentences and
each sentence is scored against the question — embedding similarity + keyword
overlap + a position prior — then the best are packed into a fixed token
budget (a greedy knapsack). Two invariants:

- **Attribution survives compression.** Every sentence carries its chunk_id,
  doc_id, and observed_at from birth, and the rendered block re-attaches the
  dated source header to each group — so dated citations still work on
  compressed evidence.
- **Determinism.** Scores are pure functions, ties break on (chunk_id, order),
  and token counts are a fixed estimate (chars/4, no tokenizer dependency) —
  the same evidence compresses identically in every run, which is what lets
  the A/B scorecard be a reproducible receipt instead of a vibe.
"""

import re
from datetime import date

from pydantic import BaseModel, ConfigDict

from groundproof.grading.grader import content_words
from groundproof.ingest.models import Chunk
from groundproof.retrieval.embeddings import EmbeddingClient, cosine_similarity
from groundproof.retrieval.temporal import RankedChunk

DEFAULT_TOKEN_BUDGET = 600

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")


def estimate_tokens(text: str) -> int:
    """Deterministic token estimate: ~4 chars per token, floor 1 for non-empty."""
    return max(1, round(len(text) / 4)) if text else 0


class SourcedSentence(BaseModel):
    """One sentence that still knows exactly where (and when) it came from."""

    model_config = ConfigDict(frozen=True)

    text: str
    chunk_id: str
    doc_id: str
    observed_at: date
    order: int  # position of this sentence within its chunk


class CompressedEvidence(BaseModel):
    """The pruner's output: the packed block plus the receipts."""

    model_config = ConfigDict(frozen=True)

    text: str
    sentences: list[SourcedSentence]
    tokens_before: int
    tokens_after: int
    sentences_total: int
    sentences_kept: int

    @property
    def savings(self) -> float:
        """Fraction of tokens removed (0.0 when there was nothing to remove)."""
        if self.tokens_before == 0:
            return 0.0
        return 1.0 - self.tokens_after / self.tokens_before


def split_chunk(chunk: Chunk) -> list[SourcedSentence]:
    """Split one chunk into attributed sentences (hard-wrapped lines unwrapped)."""
    sentences: list[SourcedSentence] = []
    order = 0
    for paragraph in _PARAGRAPH_BREAK.split(chunk.text):
        unwrapped = " ".join(paragraph.split())
        if not unwrapped:
            continue
        for text in _SENTENCE_END.split(unwrapped):
            text = text.strip()
            if text:
                sentences.append(
                    SourcedSentence(
                        text=text,
                        chunk_id=chunk.chunk_id,
                        doc_id=chunk.doc_id,
                        observed_at=chunk.observed_at,
                        order=order,
                    )
                )
                order += 1
    return sentences


class QueryAwarePruner:
    """Score sentences against the question, pack the best into the budget."""

    def __init__(
        self,
        embedder: EmbeddingClient,
        token_budget: int = DEFAULT_TOKEN_BUDGET,
        similarity_weight: float = 0.5,
        keyword_weight: float = 0.3,
        position_weight: float = 0.2,
    ) -> None:
        if token_budget <= 0:
            raise ValueError(f"token_budget must be positive, got {token_budget}")
        self._embedder = embedder
        self._token_budget = token_budget
        self._weights = (similarity_weight, keyword_weight, position_weight)

    @property
    def token_budget(self) -> int:
        return self._token_budget

    def compress(self, question: str, evidence: list[RankedChunk]) -> CompressedEvidence:
        sentences = [
            sentence for item in evidence for sentence in split_chunk(item.chunk)
        ]
        tokens_before = sum(estimate_tokens(item.chunk.text) for item in evidence)
        if not sentences:
            return CompressedEvidence(
                text="",
                sentences=[],
                tokens_before=tokens_before,
                tokens_after=0,
                sentences_total=0,
                sentences_kept=0,
            )

        scored = self._score(question, sentences)
        # Greedy knapsack: best score first, deterministic tie-break; a sentence
        # that does not fit is skipped, smaller ones after it may still fit.
        # The budget bounds what SHIPS: a chunk's attribution header is charged
        # the first time that chunk contributes a sentence.
        scored.sort(key=lambda pair: (-pair[0], pair[1].chunk_id, pair[1].order))
        budget_left = self._token_budget
        selected: list[tuple[float, SourcedSentence]] = []
        headered: set[str] = set()
        for score, sentence in scored:
            cost = estimate_tokens(sentence.text)
            if sentence.chunk_id not in headered:
                cost += estimate_tokens(_header(sentence))
            if cost <= budget_left:
                selected.append((score, sentence))
                budget_left -= cost
                headered.add(sentence.chunk_id)

        # Render in evidence-ranking order (then sentence order), attribution
        # header re-attached per chunk group. Per-part token estimates round
        # independently of the joined text, so verify the final block and trim
        # the lowest-scoring sentences if rounding pushed it over.
        chunk_rank = {item.chunk.chunk_id: rank for rank, item in enumerate(evidence)}

        def rendered(pairs: list[tuple[float, SourcedSentence]]) -> str:
            ordered = sorted(pairs, key=lambda pair: (chunk_rank[pair[1].chunk_id], pair[1].order))
            return _render([sentence for _score, sentence in ordered])

        text = rendered(selected)
        while selected and estimate_tokens(text) > self._token_budget:
            selected.remove(min(selected, key=lambda pair: (pair[0], pair[1].chunk_id)))
            text = rendered(selected)
        selected_sentences = [
            sentence
            for _score, sentence in sorted(
                selected, key=lambda pair: (chunk_rank[pair[1].chunk_id], pair[1].order)
            )
        ]
        return CompressedEvidence(
            text=text,
            sentences=selected_sentences,
            tokens_before=tokens_before,
            tokens_after=estimate_tokens(text),
            sentences_total=len(sentences),
            sentences_kept=len(selected_sentences),
        )

    def _score(
        self, question: str, sentences: list[SourcedSentence]
    ) -> list[tuple[float, SourcedSentence]]:
        similarity_weight, keyword_weight, position_weight = self._weights
        question_vector = self._embedder.embed_texts([question])[0]
        sentence_vectors = self._embedder.embed_texts([s.text for s in sentences])
        keywords = content_words(question)
        scored: list[tuple[float, SourcedSentence]] = []
        for sentence, vector in zip(sentences, sentence_vectors, strict=True):
            similarity = cosine_similarity(question_vector, vector)
            lowered = sentence.text.lower()
            overlap = (
                sum(1 for word in keywords if word in lowered) / len(keywords)
                if keywords
                else 0.0
            )
            position = 1.0 / (1.0 + sentence.order)
            score = (
                similarity_weight * similarity
                + keyword_weight * overlap
                + position_weight * position
            )
            scored.append((score, sentence))
        return scored


def _header(sentence: SourcedSentence) -> str:
    return f"({sentence.observed_at}, {sentence.doc_id}):"


def _render(selected: list[SourcedSentence]) -> str:
    """The compressed block: dated source header per group, sentences below it."""
    lines: list[str] = []
    current_chunk: str | None = None
    for sentence in selected:
        if sentence.chunk_id != current_chunk:
            lines.append(_header(sentence))
            current_chunk = sentence.chunk_id
        lines.append(sentence.text)
    return "\n".join(lines)

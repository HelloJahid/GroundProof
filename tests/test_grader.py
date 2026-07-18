"""P3: the retrieval grader gate — rules decide whether evidence may be used."""

from datetime import date

from groundproof.grading import EvidenceGrader, content_words
from groundproof.ingest import Chunk
from groundproof.retrieval import RankedChunk


def make_item(text: str, similarity: float) -> RankedChunk:
    chunk = Chunk(
        chunk_id="python-whatsnew-3.12:0000",
        doc_id="python-whatsnew-3.12",
        source="https://example.test/3.12.rst.txt",
        text=text,
        position=0,
        observed_at=date(2023, 10, 2),
    )
    return RankedChunk(chunk=chunk, similarity=similarity, recency=1.0, score=similarity)


class TestContentWords:
    def test_drops_stopwords_keeps_meaning(self):
        assert content_words("Is the distutils package still available?") == [
            "distutils",
            "package",
            "available",
        ]


class TestGrading:
    def test_no_evidence_is_weak_with_zero_strength(self):
        grade = EvidenceGrader().grade("anything at all", [])
        assert grade.verdict == "weak"
        assert grade.strength == 0.0

    def test_matching_evidence_is_strong(self):
        evidence = [make_item("The distutils package was removed in 3.12.", similarity=0.5)]
        grade = EvidenceGrader().grade("is the distutils package removed", evidence)
        assert grade.verdict == "strong"
        assert grade.keyword_overlap == 1.0

    def test_unrelated_evidence_is_weak(self):
        evidence = [make_item("The zoneinfo module was added.", similarity=0.05)]
        grade = EvidenceGrader().grade("airspeed velocity of an unladen swallow", evidence)
        assert grade.verdict == "weak"

    def test_threshold_is_tunable(self):
        evidence = [make_item("The distutils package was removed.", similarity=0.5)]
        question = "is the distutils package removed"
        assert EvidenceGrader(threshold=0.1).grade(question, evidence).verdict == "strong"
        assert EvidenceGrader(threshold=0.99).grade(question, evidence).verdict == "weak"

    def test_signals_are_visible_in_the_grade(self):
        evidence = [make_item("The distutils package was removed.", similarity=0.4)]
        grade = EvidenceGrader(similarity_weight=0.5).grade("distutils removed", evidence)
        assert grade.top_similarity == 0.4
        assert grade.strength == 0.5 * 0.4 + 0.5 * grade.keyword_overlap

    def test_embedder_recomputes_similarity_against_the_graded_question(self):
        from groundproof.retrieval import MockEmbeddingClient

        # Retrieval-time similarity lies (0.9, computed against some other
        # query); the embedder-backed grader must ignore it and re-score
        # against the question it is actually grading.
        evidence = [make_item("The zoneinfo module was added.", similarity=0.9)]
        grade = EvidenceGrader(embedder=MockEmbeddingClient()).grade(
            "airspeed velocity of an unladen swallow", evidence
        )
        assert grade.top_similarity < 0.2
        assert grade.verdict == "weak"

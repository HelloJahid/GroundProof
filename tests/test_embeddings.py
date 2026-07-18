"""P1 Step 4: the embedding port's mock is deterministic, offline, and rankable."""

import pytest

from groundproof.retrieval import EmbeddingClient, MockEmbeddingClient, cosine_similarity


class TestPort:
    def test_mock_satisfies_the_protocol(self):
        assert isinstance(MockEmbeddingClient(), EmbeddingClient)

    def test_batch_preserves_order_and_length(self):
        client = MockEmbeddingClient(dim=64)
        vectors = client.embed_texts(["alpha", "beta", "gamma"])
        assert len(vectors) == 3
        assert vectors[0] == client.embed_texts(["alpha"])[0]
        assert vectors[2] == client.embed_texts(["gamma"])[0]


class TestDeterminism:
    def test_same_text_same_vector_across_instances(self):
        one = MockEmbeddingClient().embed_texts(["Python 3.12 removed distutils"])[0]
        two = MockEmbeddingClient().embed_texts(["Python 3.12 removed distutils"])[0]
        assert one == two

    def test_configured_dimension_is_respected(self):
        assert len(MockEmbeddingClient(dim=32).embed_texts(["hello"])[0]) == 32

    def test_nonempty_text_gives_unit_vector(self):
        vector = MockEmbeddingClient().embed_texts(["hello world"])[0]
        assert cosine_similarity(vector, vector) == pytest.approx(1.0)

    def test_empty_text_gives_zero_vector(self):
        vector = MockEmbeddingClient().embed_texts([""])[0]
        assert all(value == 0.0 for value in vector)


class TestSimilarityStructure:
    def test_word_overlap_ranks_higher_than_unrelated_text(self):
        client = MockEmbeddingClient()
        query, related, unrelated = client.embed_texts(
            [
                "when was python 3.12 released",
                "Python 3.12 was released in October 2023.",
                "A banana smoothie needs two ripe bananas.",
            ]
        )
        assert cosine_similarity(query, related) > cosine_similarity(query, unrelated)

    def test_version_numbers_are_retrieval_vocabulary(self):
        client = MockEmbeddingClient()
        query, right_version, wrong_version = client.embed_texts(
            [
                "what changed in 3.12",
                "In 3.12 the distutils package was removed.",
                "In 3.9 the zoneinfo module was added.",
            ]
        )
        assert cosine_similarity(query, right_version) > cosine_similarity(query, wrong_version)


class TestCosine:
    def test_orthogonal_vectors_score_zero(self):
        assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0

    def test_zero_vector_scores_zero_not_nan(self):
        assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0

    def test_length_mismatch_is_an_error(self):
        with pytest.raises(ValueError, match="lengths differ"):
            cosine_similarity([1.0], [1.0, 2.0])

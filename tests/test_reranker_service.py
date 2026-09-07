import pytest

from app.services.reranker_service import RerankerService
from app.services.rrf_service import RankedItem


class FakeCrossEncoder:
    def __init__(self, scores):
        self.scores = scores
        self.received_pairs = None
        self.called = False

    def predict(
        self,
        pairs,
        batch_size=8,
        show_progress_bar=False,
    ):
        self.called = True
        self.received_pairs = pairs
        return self.scores


def make_candidate(
    chunk_id: str,
    document_id: str,
    text: str,
    score: float = 0.0,
) -> RankedItem:
    return RankedItem(
        chunk_id=chunk_id,
        document_id=document_id,
        chunk_index=0,
        text=text,
        score=score,
    )


def test_reranker_orders_candidates_by_score():
    model = FakeCrossEncoder(
        scores=[0.2, 0.95, 0.5]
    )

    service = RerankerService(model=model)

    candidates = [
        make_candidate(
            "chunk_a",
            "doc_a",
            "First passage",
        ),
        make_candidate(
            "chunk_b",
            "doc_b",
            "Second passage",
        ),
        make_candidate(
            "chunk_c",
            "doc_c",
            "Third passage",
        ),
    ]

    results = service.rerank(
        query="test query",
        candidates=candidates,
        top_k=3,
    )

    assert [
        result.chunk_id
        for result in results
    ] == [
        "chunk_b",
        "chunk_c",
        "chunk_a",
    ]


def test_reranker_returns_only_top_k():
    model = FakeCrossEncoder(
        scores=[0.1, 0.9, 0.5]
    )

    service = RerankerService(model=model)

    candidates = [
        make_candidate("a", "doc_a", "A"),
        make_candidate("b", "doc_b", "B"),
        make_candidate("c", "doc_c", "C"),
    ]

    results = service.rerank(
        query="query",
        candidates=candidates,
        top_k=2,
    )

    assert len(results) == 2
    assert results[0].chunk_id == "b"
    assert results[1].chunk_id == "c"


def test_reranker_builds_query_document_pairs():
    model = FakeCrossEncoder(
        scores=[0.8, 0.3]
    )

    service = RerankerService(model=model)

    candidates = [
        make_candidate(
            "a",
            "doc_a",
            "Document A text",
        ),
        make_candidate(
            "b",
            "doc_b",
            "Document B text",
        ),
    ]

    service.rerank(
        query="My query",
        candidates=candidates,
    )

    assert model.received_pairs == [
        ["My query", "Document A text"],
        ["My query", "Document B text"],
    ]


def test_reranker_handles_empty_candidates():
    model = FakeCrossEncoder(scores=[])

    service = RerankerService(model=model)

    results = service.rerank(
        query="query",
        candidates=[],
    )

    assert results == []
    assert model.called is False


def test_reranker_rejects_invalid_top_k():
    model = FakeCrossEncoder(scores=[0.5])

    service = RerankerService(model=model)

    candidates = [
        make_candidate(
            "a",
            "doc_a",
            "Document",
        )
    ]

    with pytest.raises(ValueError):
        service.rerank(
            query="query",
            candidates=candidates,
            top_k=0,
        )
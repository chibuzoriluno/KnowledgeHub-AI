import pytest

from app.models.search import SearchResponse, SearchResult
from app.services.reranked_hybrid_retrieval_service import (
    RerankedHybridRetrievalService,
)
from app.services.rrf_service import RankedItem


class FakeHybridService:
    def __init__(self, response):
        self.response = response
        self.called = False
        self.received_top_k = None
        self.received_max_distance = None

    def search(
        self,
        query,
        top_k,
        dense_candidate_k,
        bm25_candidate_k,
        max_distance,
        document_id=None,
    ):
        self.called = True
        self.received_top_k = top_k
        self.received_max_distance = max_distance
        return self.response


class FakeRerankerService:
    def __init__(self):
        self.called = False
        self.received_candidates = None

    def rerank(
        self,
        query,
        candidates,
        top_k,
    ):
        self.called = True
        self.received_candidates = candidates

        return list(reversed(candidates))[:top_k]


def make_result(
    chunk_id: str,
    document_id: str,
    score: float,
):
    return SearchResult(
        chunk_id=chunk_id,
        document_id=document_id,
        chunk_index=0,
        text=f"Text for {chunk_id}",
        distance=0.0,
        score=score,
    )


def test_reranked_hybrid_uses_larger_candidate_pool():
    hybrid_response = SearchResponse(
        query="test",
        result_count=3,
        results=[
            make_result("a", "doc_a", 0.03),
            make_result("b", "doc_b", 0.02),
            make_result("c", "doc_c", 0.01),
        ],
    )

    hybrid_service = FakeHybridService(
        hybrid_response
    )
    reranker_service = FakeRerankerService()

    service = RerankedHybridRetrievalService(
        hybrid_service=hybrid_service,
        reranker_service=reranker_service,
    )

    response = service.search(
        query="test",
        top_k=2,
        rerank_candidate_k=10,
    )

    assert hybrid_service.received_top_k == 10
    assert reranker_service.called is True
    assert response.result_count == 2


def test_reranked_hybrid_preserves_ood_rejection():
    hybrid_response = SearchResponse(
        query="unknown",
        result_count=0,
        results=[],
    )

    hybrid_service = FakeHybridService(
        hybrid_response
    )
    reranker_service = FakeRerankerService()

    service = RerankedHybridRetrievalService(
        hybrid_service=hybrid_service,
        reranker_service=reranker_service,
    )

    response = service.search(
        query="unknown"
    )

    assert response.result_count == 0
    assert response.results == []
    assert reranker_service.called is False


def test_reranked_hybrid_returns_reranker_order():
    hybrid_response = SearchResponse(
        query="test",
        result_count=3,
        results=[
            make_result("a", "doc_a", 0.03),
            make_result("b", "doc_b", 0.02),
            make_result("c", "doc_c", 0.01),
        ],
    )

    service = RerankedHybridRetrievalService(
        hybrid_service=FakeHybridService(
            hybrid_response
        ),
        reranker_service=FakeRerankerService(),
    )

    response = service.search(
        query="test",
        top_k=3,
        rerank_candidate_k=3,
    )

    assert [
        result.chunk_id
        for result in response.results
    ] == ["c", "b", "a"]


def test_reranked_hybrid_rejects_candidate_pool_smaller_than_top_k():
    service = RerankedHybridRetrievalService(
        hybrid_service=FakeHybridService(
            SearchResponse(
                query="test",
                result_count=0,
                results=[],
            )
        ),
        reranker_service=FakeRerankerService(),
    )

    with pytest.raises(ValueError):
        service.search(
            query="test",
            top_k=3,
            rerank_candidate_k=2,
        )


def test_reranked_hybrid_uses_default_distance_when_none():
    hybrid_response = SearchResponse(
        query="unknown",
        result_count=0,
        results=[],
    )

    hybrid_service = FakeHybridService(
        hybrid_response
    )

    service = RerankedHybridRetrievalService(
        hybrid_service=hybrid_service,
        reranker_service=FakeRerankerService(),
    )

    service.search(
        query="unknown",
        max_distance=None,
    )

    assert hybrid_service.received_max_distance == 1.3
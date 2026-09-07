from dataclasses import dataclass

from app.models.search import SearchResponse, SearchResult
from app.services.hybrid_retrieval_service import (
    HybridRetrievalService,
)


@dataclass
class FakeDenseService:
    response: SearchResponse

    def search(
        self,
        query: str,
        top_k: int,
        max_distance: float | None,
        document_id: str | None = None,
    ) -> SearchResponse:
        return self.response


@dataclass
class FakeBM25Service:
    response: SearchResponse
    called: bool = False

    def search(
        self,
        query: str,
        top_k: int,
        document_id: str | None = None,
    ) -> SearchResponse:
        self.called = True
        return self.response


class FakeRRFService:
    def __init__(self):
        self.called = False

    def fuse(
        self,
        rankings,
        top_k: int,
    ):
        self.called = True
        return rankings[0][:top_k]


def make_result(
    chunk_id: str,
    document_id: str,
) -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id,
        document_id=document_id,
        chunk_index=0,
        text=chunk_id,
        distance=0.5,
    )


def test_hybrid_search_combines_retrievers():
    dense_response = SearchResponse(
        query="test",
        result_count=2,
        results=[
            make_result("dense_1", "doc_a"),
            make_result("dense_2", "doc_a"),
        ],
    )

    bm25_response = SearchResponse(
        query="test",
        result_count=2,
        results=[
            make_result("bm25_1", "doc_b"),
            make_result("bm25_2", "doc_b"),
        ],
    )

    bm25_service = FakeBM25Service(
        response=bm25_response
    )

    rrf_service = FakeRRFService()

    service = HybridRetrievalService(
        dense_service=FakeDenseService(
            response=dense_response
        ),
        bm25_service=bm25_service,
        rrf_service=rrf_service,
    )

    response = service.search(
        query="test",
        top_k=2,
    )

    assert response.result_count == 2
    assert bm25_service.called is True
    assert rrf_service.called is True


def test_hybrid_search_rejects_when_dense_returns_no_results():
    dense_response = SearchResponse(
        query="unknown",
        result_count=0,
        results=[],
    )

    bm25_response = SearchResponse(
        query="unknown",
        result_count=1,
        results=[
            make_result(
                "bm25_1",
                "unrelated",
            )
        ],
    )

    bm25_service = FakeBM25Service(
        response=bm25_response
    )

    rrf_service = FakeRRFService()

    service = HybridRetrievalService(
        dense_service=FakeDenseService(
            response=dense_response
        ),
        bm25_service=bm25_service,
        rrf_service=rrf_service,
    )

    response = service.search(
        query="unknown",
        top_k=3,
    )

    assert response.result_count == 0
    assert response.results == []
    assert bm25_service.called is False
    assert rrf_service.called is False
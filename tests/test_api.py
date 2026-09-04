import asyncio

import httpx

from app.main import app
from app.models.search import RAGResponse


class FakeRAGService:
    async def answer(
        self,
        query: str,
        top_k: int = 3,
        max_distance: float | None = None,
        document_id: str | None = None,
    ) -> RAGResponse:
        return RAGResponse(
            query=query,
            answer="Fake test answer.",
            sources=[],
        )


class FailingRAGService:
    async def answer(
        self,
        query: str,
        top_k: int = 3,
        max_distance: float | None = None,
        document_id: str | None = None,
    ) -> RAGResponse:
        from app.services.generation_service import GenerationServiceError

        raise GenerationServiceError(
            "The local LLM service is unavailable."
        )


async def make_request(
    method: str,
    url: str,
    json: dict | None = None,
) -> httpx.Response:
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://testserver",
    ) as client:
        return await client.request(
            method,
            url,
            json=json,
        )


def test_rag_endpoint_returns_200(monkeypatch):
    monkeypatch.setattr(
        "app.api.routes.rag_service",
        FakeRAGService(),
    )

    response = asyncio.run(
        make_request(
            "POST",
            "/rag",
            {"query": "What is machine learning?"},
        )
    )

    assert response.status_code == 200
    assert response.json() == {
        "query": "What is machine learning?",
        "answer": "Fake test answer.",
        "sources": [],
    }


def test_rag_endpoint_rejects_empty_query():
    response = asyncio.run(
        make_request(
            "POST",
            "/rag",
            {"query": ""},
        )
    )

    assert response.status_code == 422


def test_rag_endpoint_rejects_invalid_top_k():
    response = asyncio.run(
        make_request(
            "POST",
            "/rag",
            {
                "query": "What is machine learning?",
                "top_k": 0,
            },
        )
    )

    assert response.status_code == 422


def test_rag_endpoint_accepts_explicit_distance(monkeypatch):
    monkeypatch.setattr(
        "app.api.routes.rag_service",
        FakeRAGService(),
    )

    response = asyncio.run(
        make_request(
            "POST",
            "/rag",
            {
                "query": "What is machine learning?",
                "max_distance": 0.4,
            },
        )
    )

    assert response.status_code == 200


def test_rag_endpoint_returns_503_on_generation_failure(monkeypatch):
    monkeypatch.setattr(
        "app.api.routes.rag_service",
        FailingRAGService(),
    )

    response = asyncio.run(
        make_request(
            "POST",
            "/rag",
            {"query": "What is machine learning?"},
        )
    )

    assert response.status_code == 503
    assert response.json() == {
        "detail": "The local LLM service is unavailable."
    }


def test_search_endpoint_accepts_document_filter(monkeypatch):
    class FakeRetrievalService:
        def search(
            self,
            query: str,
            top_k: int = 3,
            max_distance: float | None = None,
            document_id: str | None = None,
        ):
            from app.models.search import SearchResponse, SearchResult

            return SearchResponse(
                query=query,
                result_count=1,
                results=[
                    SearchResult(
                        chunk_id="test_doc_chunk_000",
                        document_id=document_id or "test_doc",
                        chunk_index=0,
                        text="Machine learning learns from data.",
                        distance=0.5,
                    )
                ],
            )

    monkeypatch.setattr(
        "app.api.routes.retrieval_service",
        FakeRetrievalService(),
    )

    response = asyncio.run(
        make_request(
            "POST",
            "/search",
            {
                "query": "What is machine learning?",
                "document_id": "test_doc",
            },
        )
    )

    assert response.status_code == 200
    assert response.json()["results"][0]["document_id"] == "test_doc"


def test_search_endpoint_works_without_document_filter(monkeypatch):
    class FakeRetrievalService:
        def search(
            self,
            query: str,
            top_k: int = 3,
            max_distance: float | None = None,
            document_id: str | None = None,
        ):
            from app.models.search import SearchResponse, SearchResult

            return SearchResponse(
                query=query,
                result_count=1,
                results=[
                    SearchResult(
                        chunk_id="test_doc_chunk_000",
                        document_id="test_doc",
                        chunk_index=0,
                        text="Machine learning learns from data.",
                        distance=0.5,
                    )
                ],
            )

    monkeypatch.setattr(
        "app.api.routes.retrieval_service",
        FakeRetrievalService(),
    )

    response = asyncio.run(
        make_request(
            "POST",
            "/search",
            {
                "query": "What is machine learning?",
            },
        )
    )

    assert response.status_code == 200
    assert response.json()["result_count"] == 1


def test_rag_endpoint_accepts_document_filter(monkeypatch):
    class FilterAwareRAGService:
        async def answer(
            self,
            query: str,
            top_k: int = 3,
            max_distance: float | None = None,
            document_id: str | None = None,
        ) -> RAGResponse:
            assert document_id == "test_doc"

            return RAGResponse(
                query=query,
                answer="Fake filtered answer.",
                sources=[],
            )

    monkeypatch.setattr(
        "app.api.routes.rag_service",
        FilterAwareRAGService(),
    )

    response = asyncio.run(
        make_request(
            "POST",
            "/rag",
            {
                "query": "What is machine learning?",
                "document_id": "test_doc",
            },
        )
    )

    assert response.status_code == 200
    assert response.json()["answer"] == "Fake filtered answer."
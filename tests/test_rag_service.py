import asyncio

from app.models.search import SearchResponse, SearchResult
from app.services.rag_service import RAGService


class FakeRetrievalService:
    def __init__(self, results: list[SearchResult]):
        self.results = results
        self.last_query = None
        self.last_top_k = None
        self.last_max_distance = None

    def search(
        self,
        query: str,
        top_k: int,
        max_distance: float | None,
    ) -> SearchResponse:
        self.last_query = query
        self.last_top_k = top_k
        self.last_max_distance = max_distance

        return SearchResponse(
            query=query,
            result_count=len(self.results),
            results=self.results,
        )


class FakeGenerationService:
    def __init__(self):
        self.called = False
        self.last_prompt = None

    async def generate(self, prompt: str) -> str:
        self.called = True
        self.last_prompt = prompt
        return "Machine learning learns patterns from data."


def make_search_result(
    chunk_id: str = "chunk_001",
    document_id: str = "document_001",
    chunk_index: int = 0,
    text: str = "Machine learning learns patterns from data.",
    distance: float = 0.5,
) -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id,
        document_id=document_id,
        chunk_index=chunk_index,
        text=text,
        distance=distance,
    )


def test_rag_returns_answer_and_sources():
    retrieval_service = FakeRetrievalService(
        [make_search_result()]
    )
    generation_service = FakeGenerationService()

    service = RAGService(
        retrieval_service=retrieval_service,
        generation_service=generation_service,
    )

    response = asyncio.run(
        service.answer("What is machine learning?")
    )

    assert response.query == "What is machine learning?"
    assert response.answer == (
        "Machine learning learns patterns from data."
    )
    assert len(response.sources) == 1
    assert response.sources[0].chunk_id == "chunk_001"
    assert generation_service.called is True


def test_rag_uses_configured_default_distance():
    retrieval_service = FakeRetrievalService(
        [make_search_result()]
    )
    generation_service = FakeGenerationService()

    service = RAGService(
        retrieval_service=retrieval_service,
        generation_service=generation_service,
    )

    asyncio.run(
        service.answer("What is machine learning?")
    )

    assert retrieval_service.last_max_distance == 1.3


def test_rag_respects_explicit_distance():
    retrieval_service = FakeRetrievalService(
        [make_search_result()]
    )
    generation_service = FakeGenerationService()

    service = RAGService(
        retrieval_service=retrieval_service,
        generation_service=generation_service,
    )

    asyncio.run(
        service.answer(
            "What is machine learning?",
            max_distance=0.4,
        )
    )

    assert retrieval_service.last_max_distance == 0.4


def test_rag_does_not_generate_without_sources():
    retrieval_service = FakeRetrievalService([])
    generation_service = FakeGenerationService()

    service = RAGService(
        retrieval_service=retrieval_service,
        generation_service=generation_service,
    )

    response = asyncio.run(
        service.answer("What is quantum entanglement?")
    )

    assert response.sources == []
    assert "don't have enough information" in response.answer.lower()
    assert generation_service.called is False
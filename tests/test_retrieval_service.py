from app.services.retrieval_service import RetrievalService


class FakeEmbeddingService:
    def embed_text(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3]


class FakeVectorService:
    def __init__(self):
        self.last_top_k = None
        self.last_max_distance = None
        self.last_document_id = None

    def search(
        self,
        query_embedding: list[float],
        top_k: int,
        max_distance: float | None,
        document_id: str | None,
    ):
        self.last_top_k = top_k
        self.last_max_distance = max_distance
        self.last_document_id = document_id

        return {
            "ids": [["chunk_001"]],
            "documents": [["Machine learning learns from data."]],
            "metadatas": [[
                {
                    "document_id": (
                        document_id
                        if document_id is not None
                        else "test_doc"
                    ),
                    "chunk_index": 0,
                }
            ]],
            "distances": [[0.5]],
        }


def test_retrieval_passes_document_filter():
    embedding_service = FakeEmbeddingService()
    vector_service = FakeVectorService()

    service = RetrievalService.__new__(RetrievalService)

    service.embedding_service = embedding_service
    service.vector_service = vector_service

    response = service.search(
        query="What is machine learning?",
        top_k=3,
        max_distance=1.3,
        document_id="test_doc",
    )

    assert vector_service.last_document_id == "test_doc"
    assert response.result_count == 1
    assert response.results[0].document_id == "test_doc"


def test_retrieval_works_without_document_filter():
    embedding_service = FakeEmbeddingService()
    vector_service = FakeVectorService()

    service = RetrievalService.__new__(RetrievalService)

    service.embedding_service = embedding_service
    service.vector_service = vector_service

    response = service.search(
        query="What is machine learning?",
        top_k=3,
        max_distance=1.3,
    )

    assert vector_service.last_document_id is None
    assert response.result_count == 1


def test_retrieval_passes_top_k_and_distance():
    embedding_service = FakeEmbeddingService()
    vector_service = FakeVectorService()

    service = RetrievalService.__new__(RetrievalService)

    service.embedding_service = embedding_service
    service.vector_service = vector_service

    service.search(
        query="What is machine learning?",
        top_k=5,
        max_distance=0.8,
        document_id="test_doc",
    )

    assert vector_service.last_top_k == 5
    assert vector_service.last_max_distance == 0.8
    assert vector_service.last_document_id == "test_doc"
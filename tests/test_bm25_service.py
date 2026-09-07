from app.models.search import SearchResponse
from app.services.bm25_service import BM25Service


class FakeCollection:
    def __init__(self, data):
        self.data = data

    def get(self, include=None):
        return self.data

    def count(self):
        return len(self.data["ids"])


class FakeVectorService:
    def __init__(self, data):
        self.collection = FakeCollection(data)


def make_data():
    return {
        "ids": [
            "chunk_1",
            "chunk_2",
            "chunk_3",
        ],
        "documents": [
            "Vector databases support semantic retrieval.",
            "Relational databases organize structured tables.",
            "FastAPI provides a web API framework.",
        ],
        "metadatas": [
            {
                "document_id": "doc_a",
                "chunk_index": 0,
            },
            {
                "document_id": "doc_b",
                "chunk_index": 0,
            },
            {
                "document_id": "doc_c",
                "chunk_index": 0,
            },
        ],
    }


def build_service(data):
    service = BM25Service.__new__(BM25Service)
    service.vector_service = FakeVectorService(data)

    service.ids = []
    service.documents = []
    service.metadatas = []
    service.bm25 = None
    service._indexed_count = -1

    service._refresh_index()

    return service


def test_bm25_returns_relevant_result():
    service = build_service(make_data())

    response = service.search(
        query="semantic vector retrieval",
        top_k=1,
    )

    assert isinstance(response, SearchResponse)
    assert response.result_count == 1
    assert response.results[0].document_id == "doc_a"


def test_bm25_distance_is_none():
    service = build_service(make_data())

    response = service.search(
        query="database",
        top_k=1,
    )

    assert response.results[0].distance is None
    assert response.results[0].score is not None


def test_bm25_document_filter():
    service = build_service(make_data())

    response = service.search(
        query="database",
        top_k=3,
        document_id="doc_b",
    )

    assert response.result_count == 1
    assert response.results[0].document_id == "doc_b"


def test_bm25_handles_empty_collection():
    data = {
        "ids": [],
        "documents": [],
        "metadatas": [],
    }

    service = build_service(data)

    response = service.search(
        query="anything",
        top_k=3,
    )

    assert response.result_count == 0
    assert response.results == []


def test_bm25_refreshes_when_collection_changes():
    data = make_data()

    service = build_service(data)

    assert service._indexed_count == 3

    service.vector_service.collection.data = {
        "ids": [
            "chunk_1",
            "chunk_2",
            "chunk_3",
            "chunk_4",
        ],
        "documents": [
            "Vector databases support semantic retrieval.",
            "Relational databases organize structured tables.",
            "FastAPI provides a web API framework.",
            "BM25 performs lexical ranking.",
        ],
        "metadatas": [
            {
                "document_id": "doc_a",
                "chunk_index": 0,
            },
            {
                "document_id": "doc_b",
                "chunk_index": 0,
            },
            {
                "document_id": "doc_c",
                "chunk_index": 0,
            },
            {
                "document_id": "doc_d",
                "chunk_index": 0,
            },
        ],
    }

    response = service.search(
        query="lexical ranking",
        top_k=1,
    )

    assert service._indexed_count == 4
    assert response.results[0].document_id == "doc_d"
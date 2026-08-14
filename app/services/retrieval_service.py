from app.services.embedding_service import EmbeddingService
from app.services.vector_service import VectorService


class RetrievalService:
    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.vector_service = VectorService()

    def search(self, query: str, top_k: int = 3):
        query_embedding = self.embedding_service.embed_text(query)

        return self.vector_service.search(
            query_embedding,
            top_k=top_k,
        )
from app.models.search import SearchResponse, SearchResult
from app.services.embedding_service import EmbeddingService
from app.services.vector_service import VectorService


class RetrievalService:
    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.vector_service = VectorService()

    def search(self, query: str, top_k: int = 3) -> SearchResponse:
        query_embedding = self.embedding_service.embed_text(query)

        results = self.vector_service.search(
            query_embedding,
            top_k=top_k,
        )

        search_results = []

        for index in range(len(results["ids"][0])):
            search_results.append(
                SearchResult(
                    chunk_id=results["ids"][0][index],
                    document_id=results["metadatas"][0][index]["document_id"],
                    chunk_index=results["metadatas"][0][index]["chunk_index"],
                    text=results["documents"][0][index],
                    distance=results["distances"][0][index],
                )
            )

        return SearchResponse(
            query=query,
            results=search_results,
        )
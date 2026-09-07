from app.models.search import SearchResponse
from app.core.config import settings
from app.services.hybrid_retrieval_service import (
    HybridRetrievalService,
)
from app.services.reranker_service import RerankerService
from app.services.rrf_service import RankedItem


class RerankedHybridRetrievalService:
    def __init__(
        self,
        hybrid_service: HybridRetrievalService | None = None,
        reranker_service: RerankerService | None = None,
    ):
        self.hybrid_service = (
            hybrid_service
            or HybridRetrievalService()
        )

        self.reranker_service = (
            reranker_service
            or RerankerService()
        )


    def search(
        self,
        query: str,
        top_k: int = settings.RETRIEVAL_FINAL_TOP_K,
        rerank_candidate_k: int = settings.RETRIEVAL_RERANK_TOP_K,
        dense_candidate_k: int = settings.RETRIEVAL_DENSE_TOP_K,
        bm25_candidate_k: int = settings.RETRIEVAL_BM25_TOP_K,
        max_distance: float | None = None,
        document_id: str | None = None,
    ) -> SearchResponse:

        if max_distance is None:
            max_distance = settings.RAG_DEFAULT_MAX_DISTANCE

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero."
            )

        if rerank_candidate_k < top_k:
            raise ValueError(
                "rerank_candidate_k must be "
                "greater than or equal to top_k."
            )

        hybrid_response = self.hybrid_service.search(
            query=query,
            top_k=rerank_candidate_k,
            dense_candidate_k=dense_candidate_k,
            bm25_candidate_k=bm25_candidate_k,
            max_distance=max_distance,
            document_id=document_id,
        )

        # Preserve the dense relevance/OOD gate.
        if not hybrid_response.results:
            return SearchResponse(
                query=query,
                result_count=0,
                results=[],
            )

        candidates = [
            RankedItem(
                chunk_id=result.chunk_id,
                document_id=result.document_id,
                chunk_index=result.chunk_index,
                text=result.text,
                score=result.score or 0.0,
                distance=result.distance,
            )
            for result in hybrid_response.results
        ]

        reranked_results = (
            self.reranker_service.rerank(
                query=query,
                candidates=candidates,
                top_k=top_k,
            )
        )

        search_results = [
            {
                "chunk_id": item.chunk_id,
                "document_id": item.document_id,
                "chunk_index": item.chunk_index,
                "text": item.text,
                "distance": item.distance,
                "score": item.score,
            }
            for item in reranked_results
        ]

        return SearchResponse(
            query=query,
            result_count=len(search_results),
            results=search_results,
        )
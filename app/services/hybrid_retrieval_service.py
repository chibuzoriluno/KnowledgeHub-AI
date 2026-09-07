from app.models.search import SearchResponse
from app.services.bm25_service import BM25Service
from app.services.retrieval_service import RetrievalService
from app.services.rrf_service import RRFService, RankedItem
from app.core.config import settings


class HybridRetrievalService:
    def __init__(
        self,
        dense_service: RetrievalService | None = None,
        bm25_service: BM25Service | None = None,
        rrf_service: RRFService | None = None,
    ):
        self.dense_service = (
            dense_service
            or RetrievalService()
        )

        self.bm25_service = (
            bm25_service
            or BM25Service()
        )

        self.rrf_service = (
            rrf_service
            or RRFService(
                k=settings.RETRIEVAL_RRF_K
            )
        )


    @staticmethod
    def _to_ranked_items(
        results,
        preserve_distance: bool,
    ) -> list[RankedItem]:
        return [
            RankedItem(
                chunk_id=result.chunk_id,
                document_id=result.document_id,
                chunk_index=result.chunk_index,
                text=result.text,
                distance=(
                    result.distance
                    if preserve_distance
                    else None
                ),
            )
            for result in results
        ]


    def search(
        self,
        query: str,
        top_k: int = settings.RETRIEVAL_FINAL_TOP_K,
        dense_candidate_k: int = settings.RETRIEVAL_DENSE_TOP_K,
        bm25_candidate_k: int = settings.RETRIEVAL_BM25_TOP_K,
        max_distance: float | None = None,
        document_id: str | None = None,
    ) -> SearchResponse:

        dense_response = self.dense_service.search(
            query=query,
            top_k=dense_candidate_k,
            max_distance=max_distance,
            document_id=document_id,
        )

        if max_distance is None:
            max_distance = settings.RAG_DEFAULT_MAX_DISTANCE

        # Dense retrieval acts as the relevance/OOD gate.
        if not dense_response.results:
            return SearchResponse(
                query=query,
                result_count=0,
                results=[],
            )

        bm25_response = self.bm25_service.search(
            query=query,
            top_k=bm25_candidate_k,
            document_id=document_id,
        )


        dense_ranking = self._to_ranked_items(
            dense_response.results,
            preserve_distance=True,
        )


        bm25_ranking = self._to_ranked_items(
            bm25_response.results,
            preserve_distance=False,
        )

        fused_results = self.rrf_service.fuse(
            rankings=[
                dense_ranking,
                bm25_ranking,
            ],
            top_k=top_k,
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
            for item in fused_results
        ]

        return SearchResponse(
            query=query,
            result_count=len(search_results),
            results=search_results,
        )
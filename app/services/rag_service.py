from app.core.config import settings
from app.models.search import RAGResponse, RAGSource
from app.services.retrieval_service import RetrievalService
from app.services.generation_service import GenerationService


class RAGService:
    def __init__(
        self,
        retrieval_service: RetrievalService | None = None,
        generation_service: GenerationService | None = None,
    ):
        self.retrieval_service = (
            retrieval_service
            if retrieval_service is not None
            else RetrievalService()
        )

        self.generation_service = (
            generation_service
            if generation_service is not None
            else GenerationService()
        )

    async def answer(
        self,
        query: str,
        top_k: int = 3,
        max_distance: float | None = None,
    ) -> RAGResponse:

        if max_distance is None:
            max_distance = settings.RAG_DEFAULT_MAX_DISTANCE

        search_response = self.retrieval_service.search(
            query=query,
            top_k=top_k,
            max_distance=max_distance,
        )

        if not search_response.results:
            return RAGResponse(
                query=query,
                answer=(
                    "I don't have enough information in the provided "
                    "documents to answer that question."
                ),
                sources=[],
            )

        context = "\n\n".join(
            result.text
            for result in search_response.results
        )

        prompt = f"""
You are a question-answering assistant.

Answer the user's question using ONLY the information contained
in the provided context.

If the context does not contain enough information to answer the
question, say that you do not have enough information in the
provided documents.

Do not use outside knowledge.
Do not invent facts.

Context:
{context}

Question:
{query}

Answer:
""".strip()

        answer = await self.generation_service.generate(prompt)

        sources = [
            RAGSource(
                chunk_id=result.chunk_id,
                document_id=result.document_id,
                chunk_index=result.chunk_index,
                distance=result.distance,
            )
            for result in search_response.results
        ]

        return RAGResponse(
            query=query,
            answer=answer,
            sources=sources,
        )
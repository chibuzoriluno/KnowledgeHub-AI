from app.services.retrieval_service import RetrievalService
from app.services.generation_service import GenerationService


class RAGService:
    def __init__(self):
        self.retrieval_service = RetrievalService()
        self.generation_service = GenerationService()

    async def answer(
        self,
        query: str,
        top_k: int = 3,
        max_distance: float | None = None,
    ) -> str:

        search_response = self.retrieval_service.search(
            query=query,
            top_k=top_k,
            max_distance=max_distance,
        )

        if not search_response.results:
            return (
                "I don't have enough information in the provided "
                "documents to answer that question."
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

        return await self.generation_service.generate(prompt)
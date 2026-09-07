import re

from rank_bm25 import BM25Okapi

from app.models.search import SearchResponse, SearchResult
from app.services.vector_service import VectorService


_TOKEN_PATTERN = re.compile(r"\b\w+\b")


def tokenize(text: str) -> list[str]:
    return _TOKEN_PATTERN.findall(text.lower())


class BM25Service:
    def __init__(self):
        self.vector_service = VectorService()

        self.ids = []
        self.documents = []
        self.metadatas = []
        self.bm25 = None
        self._indexed_count = -1

        self._refresh_index()

    def _refresh_index(self) -> None:
        data = self.vector_service.collection.get(
            include=["documents", "metadatas"],
        )

        self.ids = data["ids"]
        self.documents = data["documents"]
        self.metadatas = data["metadatas"]

        tokenized_documents = [
            tokenize(document)
            for document in self.documents
        ]

        self.bm25 = (
            BM25Okapi(tokenized_documents)
            if tokenized_documents
            else None
        )

        self._indexed_count = len(self.ids)

    def _refresh_if_needed(self) -> None:
        current_count = (
            self.vector_service.collection.count()
        )

        if current_count != self._indexed_count:
            self._refresh_index()

    def search(
        self,
        query: str,
        top_k: int = 3,
        document_id: str | None = None,
    ) -> SearchResponse:

        self._refresh_if_needed()

        if self.bm25 is None:
            return SearchResponse(
                query=query,
                result_count=0,
                results=[],
            )

        query_tokens = tokenize(query)

        scores = self.bm25.get_scores(
            query_tokens
        )

        candidate_indices = list(
            range(len(self.ids))
        )

        if document_id is not None:
            candidate_indices = [
                index
                for index in candidate_indices
                if self.metadatas[index][
                    "document_id"
                ] == document_id
            ]

        ranked_indices = sorted(
            candidate_indices,
            key=lambda index: scores[index],
            reverse=True,
        )[:top_k]

        search_results = []

        for index in ranked_indices:
            search_results.append(
                SearchResult(
                    chunk_id=self.ids[index],
                    document_id=self.metadatas[
                        index
                    ]["document_id"],
                    chunk_index=self.metadatas[
                        index
                    ]["chunk_index"],
                    text=self.documents[index],
                    distance=None,
                    score=float(scores[index]),
                )
            )

        return SearchResponse(
            query=query,
            result_count=len(search_results),
            results=search_results,
        )
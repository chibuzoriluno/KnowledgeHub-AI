from functools import lru_cache

import torch
from sentence_transformers import CrossEncoder

from app.core.config import settings
from app.services.rrf_service import RankedItem


@lru_cache(maxsize=1)
def get_reranker_model() -> CrossEncoder:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    return CrossEncoder(
        settings.RERANKER_MODEL,
        device=device,
    )


class RerankerService:
    def __init__(self, model=None):
        self._model = model

    @property
    def model(self):
        if self._model is None:
            self._model = get_reranker_model()

        return self._model

    def rerank(
        self,
        query: str,
        candidates: list[RankedItem],
        top_k: int = 3,
        batch_size: int = 8,
    ) -> list[RankedItem]:

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero."
            )

        if not candidates:
            return []

        pairs = [
            [query, candidate.text]
            for candidate in candidates
        ]

        scores = self.model.predict(
            pairs,
            batch_size=batch_size,
            show_progress_bar=False,
        )

        reranked = sorted(
            zip(candidates, scores),
            key=lambda item: float(item[1]),
            reverse=True,
        )

        results = []

        for candidate, score in reranked[:top_k]:
            results.append(
                RankedItem(
                    chunk_id=candidate.chunk_id,
                    document_id=candidate.document_id,
                    chunk_index=candidate.chunk_index,
                    text=candidate.text,
                    score=float(score),
                    distance=candidate.distance,
                )
            )

        return results
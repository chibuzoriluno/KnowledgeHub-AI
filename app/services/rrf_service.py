from dataclasses import dataclass


@dataclass
class RankedItem:
    chunk_id: str
    document_id: str
    chunk_index: int
    text: str
    score: float = 0.0
    distance: float | None = None


class RRFService:
    def __init__(self, k: int = 60):
        if k <= 0:
            raise ValueError(
                "RRF k must be greater than zero."
            )

        self.k = k

    def fuse(
        self,
        rankings: list[list[RankedItem]],
        top_k: int = 3,
    ) -> list[RankedItem]:

        fused_scores: dict[str, float] = {}
        items_by_id: dict[str, RankedItem] = {}

        for ranking in rankings:
            for rank, item in enumerate(
                ranking,
                start=1,
            ):
                fused_scores[item.chunk_id] = (
                    fused_scores.get(
                        item.chunk_id,
                        0.0,
                    )
                    + 1.0 / (self.k + rank)
                )

                existing = items_by_id.get(
                    item.chunk_id
                )

                if existing is None:
                    items_by_id[item.chunk_id] = item

                elif (
                    existing.distance is None
                    and item.distance is not None
                ):
                    # Prefer metadata from dense retrieval
                    # when the same chunk appears in both
                    # dense and BM25 rankings.
                    items_by_id[item.chunk_id] = item

        ranked_ids = sorted(
            fused_scores,
            key=fused_scores.get,
            reverse=True,
        )[:top_k]

        results = []

        for chunk_id in ranked_ids:
            item = items_by_id[chunk_id]

            results.append(
                RankedItem(
                    chunk_id=item.chunk_id,
                    document_id=item.document_id,
                    chunk_index=item.chunk_index,
                    text=item.text,
                    score=fused_scores[chunk_id],
                    distance=item.distance,
                )
            )

        return results
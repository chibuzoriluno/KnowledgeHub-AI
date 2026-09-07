from app.services.rrf_service import RankedItem, RRFService


def make_item(
    chunk_id: str,
    document_id: str,
) -> RankedItem:
    return RankedItem(
        chunk_id=chunk_id,
        document_id=document_id,
        chunk_index=0,
        text=chunk_id,
    )


def test_rrf_promotes_items_appearing_in_both_rankings():
    service = RRFService(k=60)

    ranking_a = [
        make_item("chunk_a", "doc_a"),
        make_item("chunk_b", "doc_b"),
        make_item("chunk_c", "doc_c"),
    ]

    ranking_b = [
        make_item("chunk_c", "doc_c"),
        make_item("chunk_b", "doc_b"),
        make_item("chunk_d", "doc_d"),
    ]

    results = service.fuse(
        [ranking_a, ranking_b],
        top_k=4,
    )

    assert [item.chunk_id for item in results] == [
        "chunk_c",
        "chunk_b",
        "chunk_a",
        "chunk_d",
    ]


def test_rrf_returns_top_k_results():
    service = RRFService(k=60)

    ranking = [
        make_item("chunk_a", "doc_a"),
        make_item("chunk_b", "doc_b"),
        make_item("chunk_c", "doc_c"),
        make_item("chunk_d", "doc_d"),
    ]

    results = service.fuse(
        [ranking],
        top_k=2,
    )

    assert len(results) == 2
    assert [item.chunk_id for item in results] == [
        "chunk_a",
        "chunk_b",
    ]


def test_rrf_score_uses_rank_not_original_score():
    service = RRFService(k=60)

    ranking_a = [
        make_item("chunk_a", "doc_a"),
        make_item("chunk_b", "doc_b"),
    ]

    ranking_b = [
        make_item("chunk_b", "doc_b"),
        make_item("chunk_a", "doc_a"),
    ]

    results = service.fuse(
        [ranking_a, ranking_b],
        top_k=2,
    )

    assert results[0].chunk_id == "chunk_a"
    assert results[1].chunk_id == "chunk_b"

    assert results[0].score == results[1].score


def test_rrf_rejects_invalid_k():
    try:
        RRFService(k=0)
        assert False
    except ValueError as exc:
        assert str(exc) == "RRF k must be greater than zero."
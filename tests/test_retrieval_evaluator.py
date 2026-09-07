from dataclasses import dataclass

from app.evaluation.retrieval_evaluator import (
    calculate_document_recall_at_3,
    calculate_reciprocal_rank,
    evaluate_retrieval,
)


@dataclass
class FakeResult:
    document_id: str


@dataclass
class FakeResponse:
    results: list[FakeResult]


class FakeRetrievalService:
    def __init__(
        self,
        results_by_query: dict[str, list[str]],
    ):
        self.results_by_query = results_by_query

    def search(
        self,
        query: str,
        top_k: int,
        max_distance: float,
    ) -> FakeResponse:
        document_ids = self.results_by_query.get(
            query,
            [],
        )

        return FakeResponse(
            results=[
                FakeResult(document_id=document_id)
                for document_id in document_ids[:top_k]
            ]
        )


def test_evaluation_metrics_calculate_correctly():
    evaluation_set = [
        {
            "query": "q1",
            "expected_document_ids": ["doc_a"],
        },
        {
            "query": "q2",
            "expected_document_ids": ["doc_b"],
        },
        {
            "query": "q3",
            "expected_document_ids": [],
        },
    ]

    retrieval_service = FakeRetrievalService(
        {
            "q1": ["doc_a", "doc_x"],
            "q2": ["doc_x", "doc_b"],
            "q3": [],
        }
    )

    metrics = evaluate_retrieval(
        evaluation_set,
        retrieval_service,
        top_k=3,
        max_distance=1.3,
    )

    assert metrics.in_domain_total == 2
    assert metrics.hit_at_1 == 1
    assert metrics.hit_at_3 == 2
    assert metrics.out_of_domain_total == 1
    assert metrics.out_of_domain_rejected == 1

    assert metrics.mean_reciprocal_rank == 0.75
    assert metrics.mean_document_recall_at_3 == 1.0


def test_evaluation_detects_out_of_domain_false_positive():
    evaluation_set = [
        {
            "query": "unknown",
            "expected_document_ids": [],
        }
    ]

    retrieval_service = FakeRetrievalService(
        {
            "unknown": ["doc_a"],
        }
    )

    metrics = evaluate_retrieval(
        evaluation_set,
        retrieval_service,
        top_k=3,
        max_distance=1.3,
    )

    assert metrics.out_of_domain_total == 1
    assert metrics.out_of_domain_rejected == 0


def test_reciprocal_rank():
    expected_documents = {"doc_a"}

    assert (
        calculate_reciprocal_rank(
            expected_documents,
            ["doc_a", "doc_b", "doc_c"],
        )
        == 1.0
    )

    assert (
        calculate_reciprocal_rank(
            expected_documents,
            ["doc_b", "doc_a", "doc_c"],
        )
        == 0.5
    )

    assert (
        calculate_reciprocal_rank(
            expected_documents,
            ["doc_b", "doc_c", "doc_a"],
        )
        == 1 / 3
    )


def test_document_recall_at_3():
    expected_documents = {
        "doc_a",
        "doc_b",
    }

    assert (
        calculate_document_recall_at_3(
            expected_documents,
            ["doc_a", "doc_b", "doc_c"],
        )
        == 1.0
    )

    assert (
        calculate_document_recall_at_3(
            expected_documents,
            ["doc_a", "doc_c"],
        )
        == 0.5
    )
from dataclasses import dataclass

from app.evaluation.retrieval_evaluator import (
    EvaluationMetrics,
    evaluate_retrieval,
)


@dataclass
class FakeResult:
    document_id: str


class FakeResponse:
    def __init__(self, document_ids: list[str]):
        self.results = [
            FakeResult(document_id=document_id)
            for document_id in document_ids
        ]


class FakeRetrievalService:
    def __init__(self, results_by_query: dict[str, list[str]]):
        self.results_by_query = results_by_query

    def search(
        self,
        query: str,
        top_k: int,
        max_distance: float | None,
    ) -> FakeResponse:
        return FakeResponse(
            self.results_by_query.get(query, [])
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

    assert isinstance(metrics, EvaluationMetrics)
    assert metrics.in_domain_total == 2
    assert metrics.hit_at_1 == 1
    assert metrics.hit_at_3 == 2
    assert metrics.out_of_domain_total == 1
    assert metrics.out_of_domain_rejected == 1


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
    assert metrics.out_of_domain_rejection_rate == 0.0
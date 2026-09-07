import json
import time
from dataclasses import dataclass
from pathlib import Path

from app.services.retrieval_service import RetrievalService


EVAL_FILE = Path("app/evaluation/retrieval_eval.json")
TOP_K = 3
MAX_DISTANCE = 1.3


@dataclass
class EvaluationMetrics:
    in_domain_total: int
    hit_at_1: int
    hit_at_3: int
    out_of_domain_total: int
    out_of_domain_rejected: int

    reciprocal_rank_sum: float = 0.0
    document_recall_at_3_sum: float = 0.0
    total_latency_ms: float = 0.0

    @property
    def hit_at_1_rate(self) -> float:
        if self.in_domain_total == 0:
            return 0.0

        return self.hit_at_1 / self.in_domain_total

    @property
    def hit_at_3_rate(self) -> float:
        if self.in_domain_total == 0:
            return 0.0

        return self.hit_at_3 / self.in_domain_total

    @property
    def out_of_domain_rejection_rate(self) -> float:
        if self.out_of_domain_total == 0:
            return 0.0

        return (
            self.out_of_domain_rejected
            / self.out_of_domain_total
        )

    @property
    def mean_reciprocal_rank(self) -> float:
        if self.in_domain_total == 0:
            return 0.0

        return (
            self.reciprocal_rank_sum
            / self.in_domain_total
        )

    @property
    def mean_document_recall_at_3(self) -> float:
        if self.in_domain_total == 0:
            return 0.0

        return (
            self.document_recall_at_3_sum
            / self.in_domain_total
        )

    @property
    def average_latency_ms(self) -> float:
        total_cases = (
            self.in_domain_total
            + self.out_of_domain_total
        )

        if total_cases == 0:
            return 0.0

        return self.total_latency_ms / total_cases


def load_evaluation_set():
    data = json.loads(
        EVAL_FILE.read_text(encoding="utf-8")
    )

    if not isinstance(data, dict):
        raise ValueError(
            "Evaluation file must contain a JSON object."
        )

    evaluation_cases = data.get("evaluation_cases")

    if not isinstance(evaluation_cases, list):
        raise ValueError(
            "Evaluation file must contain an "
            "'evaluation_cases' list."
        )

    return evaluation_cases


def calculate_reciprocal_rank(
    expected_documents: set[str],
    retrieved_documents: list[str],
) -> float:
    for rank, document_id in enumerate(
        retrieved_documents,
        start=1,
    ):
        if document_id in expected_documents:
            return 1.0 / rank

    return 0.0


def calculate_document_recall_at_3(
    expected_documents: set[str],
    retrieved_documents: list[str],
) -> float:
    if not expected_documents:
        return 0.0

    retrieved_at_3 = set(
        retrieved_documents[:3]
    )

    relevant_retrieved = (
        expected_documents & retrieved_at_3
    )

    return len(relevant_retrieved) / len(
        expected_documents
    )


def evaluate_retrieval(
    evaluation_set: list[dict],
    retrieval_service: RetrievalService,
    top_k: int = TOP_K,
    max_distance: float = MAX_DISTANCE,
) -> EvaluationMetrics:

    in_domain_total = 0
    hit_at_1 = 0
    hit_at_3 = 0

    out_of_domain_total = 0
    out_of_domain_rejected = 0

    reciprocal_rank_sum = 0.0
    document_recall_at_3_sum = 0.0
    total_latency_ms = 0.0

    for item in evaluation_set:
        question = item["query"]
        expected_documents = set(
            item["expected_document_ids"]
        )

        start_time = time.perf_counter()

        response = retrieval_service.search(
            query=question,
            top_k=top_k,
            max_distance=max_distance,
        )

        latency_ms = (
            time.perf_counter() - start_time
        ) * 1000

        total_latency_ms += latency_ms

        retrieved_documents = [
            result.document_id
            for result in response.results
        ]

        if expected_documents:
            in_domain_total += 1

            if expected_documents & set(
                retrieved_documents[:1]
            ):
                hit_at_1 += 1

            if expected_documents & set(
                retrieved_documents[:3]
            ):
                hit_at_3 += 1

            reciprocal_rank_sum += (
                calculate_reciprocal_rank(
                    expected_documents,
                    retrieved_documents,
                )
            )

            document_recall_at_3_sum += (
                calculate_document_recall_at_3(
                    expected_documents,
                    retrieved_documents,
                )
            )

        else:
            out_of_domain_total += 1

            if not retrieved_documents:
                out_of_domain_rejected += 1

    return EvaluationMetrics(
        in_domain_total=in_domain_total,
        hit_at_1=hit_at_1,
        hit_at_3=hit_at_3,
        out_of_domain_total=out_of_domain_total,
        out_of_domain_rejected=out_of_domain_rejected,
        reciprocal_rank_sum=reciprocal_rank_sum,
        document_recall_at_3_sum=(
            document_recall_at_3_sum
        ),
        total_latency_ms=total_latency_ms,
    )


def print_metrics(metrics: EvaluationMetrics) -> None:
    print("\n" + "=" * 70)

    print(
        f"In-domain Hit@1: "
        f"{metrics.hit_at_1}/"
        f"{metrics.in_domain_total} "
        f"({metrics.hit_at_1_rate:.1%})"
    )

    print(
        f"In-domain Hit@3: "
        f"{metrics.hit_at_3}/"
        f"{metrics.in_domain_total} "
        f"({metrics.hit_at_3_rate:.1%})"
    )

    print(
        f"Mean Reciprocal Rank: "
        f"{metrics.mean_reciprocal_rank:.4f}"
    )

    print(
        f"Mean document Recall@3: "
        f"{metrics.mean_document_recall_at_3:.1%}"
    )

    print(
        f"Out-of-domain rejection: "
        f"{metrics.out_of_domain_rejected}/"
        f"{metrics.out_of_domain_total} "
        f"({metrics.out_of_domain_rejection_rate:.1%})"
    )

    print(
        f"Average retrieval latency: "
        f"{metrics.average_latency_ms:.2f} ms"
    )


def evaluate() -> EvaluationMetrics:
    evaluation_set = load_evaluation_set()
    retrieval_service = RetrievalService()

    print("\nRetrieval Evaluation")
    print("=" * 70)
    print(f"Questions: {len(evaluation_set)}")
    print(f"Top-k: {TOP_K}")
    print(f"Max distance: {MAX_DISTANCE}")

    for item in evaluation_set:
        question = item["query"]
        expected_documents = set(
            item["expected_document_ids"]
        )

        start_time = time.perf_counter()

        response = retrieval_service.search(
            query=question,
            top_k=TOP_K,
            max_distance=MAX_DISTANCE,
        )

        latency_ms = (
            time.perf_counter() - start_time
        ) * 1000

        retrieved_documents = [
            result.document_id
            for result in response.results
        ]

        print(f"\nQuestion: {question}")
        print(f"Expected: {sorted(expected_documents)}")
        print(f"Retrieved: {retrieved_documents}")
        print(
            f"Latency: {latency_ms:.2f} ms"
        )

        for result in response.results:
            print(
                f"  - {result.chunk_id} | "
                f"{result.document_id} | "
                f"distance={result.distance:.4f}"
            )

        if expected_documents:
            hit_at_1 = bool(
                expected_documents
                & set(retrieved_documents[:1])
            )

            hit_at_3 = bool(
                expected_documents
                & set(retrieved_documents[:3])
            )

            reciprocal_rank = (
                calculate_reciprocal_rank(
                    expected_documents,
                    retrieved_documents,
                )
            )

            document_recall = (
                calculate_document_recall_at_3(
                    expected_documents,
                    retrieved_documents,
                )
            )

            print(
                f"Hit@1: "
                f"{'YES' if hit_at_1 else 'NO'}"
            )

            print(
                f"Hit@3: "
                f"{'YES' if hit_at_3 else 'NO'}"
            )

            print(
                f"Reciprocal rank: "
                f"{reciprocal_rank:.4f}"
            )

            print(
                f"Document Recall@3: "
                f"{document_recall:.1%}"
            )

        else:
            print(
                "Out-of-domain rejected: "
                f"{'YES' if not retrieved_documents else 'NO'}"
            )

    metrics = evaluate_retrieval(
        evaluation_set,
        retrieval_service,
        top_k=TOP_K,
        max_distance=MAX_DISTANCE,
    )

    print_metrics(metrics)

    return metrics


if __name__ == "__main__":
    evaluate()
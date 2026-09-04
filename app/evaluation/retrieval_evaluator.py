import json
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
            "Evaluation file must contain an 'evaluation_cases' list."
        )

    return evaluation_cases


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

    for item in evaluation_set:
        question = item["query"]
        expected_documents = set(
            item["expected_document_ids"]
        )

        response = retrieval_service.search(
            query=question,
            top_k=top_k,
            max_distance=max_distance,
        )

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
        f"Out-of-domain rejection: "
        f"{metrics.out_of_domain_rejected}/"
        f"{metrics.out_of_domain_total} "
        f"({metrics.out_of_domain_rejection_rate:.1%})"
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

        response = retrieval_service.search(
            query=question,
            top_k=TOP_K,
            max_distance=MAX_DISTANCE,
        )

        retrieved_documents = [
            result.document_id
            for result in response.results
        ]

        print(f"\nQuestion: {question}")
        print(f"Expected: {sorted(expected_documents)}")
        print(f"Retrieved: {retrieved_documents}")

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

            print(
                f"Hit@1: {'YES' if hit_at_1 else 'NO'}"
            )
            print(
                f"Hit@3: {'YES' if hit_at_3 else 'NO'}"
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
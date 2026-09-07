import time

from app.evaluation.retrieval_evaluator import (
    calculate_document_recall_at_3,
    calculate_reciprocal_rank,
    load_evaluation_set,
)
from app.services.reranked_hybrid_retrieval_service import (
    RerankedHybridRetrievalService,
)


DENSE_TOP_K = 10
BM25_TOP_K = 10
RERANK_CANDIDATE_K = 10
FINAL_TOP_K = 3
MAX_DISTANCE = 1.3


def evaluate_reranked_hybrid() -> None:
    evaluation_set = load_evaluation_set()

    print("\nLoading reranker...")
    load_start = time.perf_counter()

    service = RerankedHybridRetrievalService()

    # Force lazy reranker initialization now so model load time
    # does not contaminate per-query latency measurements.
    _ = service.reranker_service.model

    model_load_ms = (
        time.perf_counter() - load_start
    ) * 1000

    in_domain_total = 0
    hit_at_1 = 0
    hit_at_3 = 0

    out_of_domain_total = 0
    out_of_domain_rejected = 0

    reciprocal_rank_sum = 0.0
    document_recall_sum = 0.0
    total_latency_ms = 0.0

    print(
        "\nHybrid Dense + BM25 + RRF + BGE "
        "Reranker Evaluation"
    )
    print("=" * 70)
    print(f"Questions: {len(evaluation_set)}")
    print(f"Dense candidate pool: {DENSE_TOP_K}")
    print(f"BM25 candidate pool: {BM25_TOP_K}")
    print(
        f"Rerank candidate pool: "
        f"{RERANK_CANDIDATE_K}"
    )
    print(f"Final top-k: {FINAL_TOP_K}")
    print(f"Dense max distance: {MAX_DISTANCE}")
    print(
        f"Reranker load time: "
        f"{model_load_ms:.2f} ms"
    )

    for item in evaluation_set:
        query = item["query"]

        expected_documents = set(
            item["expected_document_ids"]
        )

        start_time = time.perf_counter()

        response = service.search(
            query=query,
            top_k=FINAL_TOP_K,
            rerank_candidate_k=RERANK_CANDIDATE_K,
            dense_candidate_k=DENSE_TOP_K,
            bm25_candidate_k=BM25_TOP_K,
            max_distance=MAX_DISTANCE,
        )

        latency_ms = (
            time.perf_counter() - start_time
        ) * 1000

        total_latency_ms += latency_ms

        retrieved_documents = [
            result.document_id
            for result in response.results
        ]

        print(f"\nQuestion: {query}")
        print(
            f"Expected: "
            f"{sorted(expected_documents)}"
        )
        print(
            f"Retrieved: "
            f"{retrieved_documents}"
        )
        print(
            f"Latency: "
            f"{latency_ms:.2f} ms"
        )

        for result in response.results:
            print(
                f"  - {result.chunk_id} | "
                f"{result.document_id} | "
                f"reranker_score="
                f"{result.score:.6f}"
            )

        if expected_documents:
            in_domain_total += 1

            hit1 = bool(
                expected_documents
                & set(retrieved_documents[:1])
            )

            hit3 = bool(
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

            hit_at_1 += int(hit1)
            hit_at_3 += int(hit3)

            reciprocal_rank_sum += (
                reciprocal_rank
            )
            document_recall_sum += (
                document_recall
            )

            print(
                f"Hit@1: "
                f"{'YES' if hit1 else 'NO'}"
            )

            print(
                f"Hit@3: "
                f"{'YES' if hit3 else 'NO'}"
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
            out_of_domain_total += 1

            rejected = (
                len(retrieved_documents) == 0
            )

            if rejected:
                out_of_domain_rejected += 1

            print(
                "Out-of-domain rejected: "
                f"{'YES' if rejected else 'NO'}"
            )

    total_cases = (
        in_domain_total
        + out_of_domain_total
    )

    average_latency = (
        total_latency_ms / total_cases
        if total_cases
        else 0.0
    )

    print("\n" + "=" * 70)

    print(
        f"In-domain Hit@1: "
        f"{hit_at_1}/{in_domain_total} "
        f"({hit_at_1 / in_domain_total:.1%})"
    )

    print(
        f"In-domain Hit@3: "
        f"{hit_at_3}/{in_domain_total} "
        f"({hit_at_3 / in_domain_total:.1%})"
    )

    print(
        f"Mean Reciprocal Rank: "
        f"{reciprocal_rank_sum / in_domain_total:.4f}"
    )

    print(
        f"Mean document Recall@3: "
        f"{document_recall_sum / in_domain_total:.1%}"
    )

    print(
        f"Out-of-domain rejection: "
        f"{out_of_domain_rejected}/"
        f"{out_of_domain_total} "
        f"({out_of_domain_rejected / out_of_domain_total:.1%})"
    )

    print(
        f"Average retrieval + reranking latency: "
        f"{average_latency:.2f} ms"
    )

    print(
        f"One-time reranker load time: "
        f"{model_load_ms:.2f} ms"
    )


if __name__ == "__main__":
    evaluate_reranked_hybrid()
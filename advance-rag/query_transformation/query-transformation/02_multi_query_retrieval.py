# 02_multi_query_retrieval.py

from collections import defaultdict

documents = [
    "Spark executor failed due to out of memory during shuffle stage.",
    "How to tune Spark executor memory for large jobs.",
    "Troubleshooting heap overflow in distributed data pipelines.",
    "Database timeout error while calling external API.",
    "Spark job failed because memory overhead was too low."
]


def generate_query_variants(query: str) -> list[str]:
    """
    Generate multiple alternative forms of the same user query.
    In real systems, an LLM often does this.
    Here we do it manually for clarity.
    """
    return [
        query,
        "Spark out of memory fix",
        "Spark executor memory issue",
        "Spark job memory exceeded",
        "Spark heap overflow troubleshooting"
    ]


def search_documents(query: str, docs: list[str], top_k: int = 3) -> list[tuple[str, int]]:
    """
    Very simple keyword overlap search.
    This is intentionally basic so the retrieval logic is easy to understand.
    """
    query_terms = set(query.lower().split())
    scored = []

    for doc in docs:
        doc_terms = set(doc.lower().split())
        score = len(query_terms.intersection(doc_terms))
        scored.append((doc, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


def reciprocal_rank_fusion(result_lists: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """
    Combine multiple ranked lists into one final ranking using RRF.
    """
    scores = defaultdict(float)

    for result_list in result_lists:
        for rank, doc in enumerate(result_list, start=1):
            scores[doc] += 1 / (k + rank)

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


if __name__ == "__main__":
    user_query = "How do I fix OOM in Spark?"

    print("User query:")
    print(user_query)

    variants = generate_query_variants(user_query)

    print("\nGenerated query variants:")
    for i, variant in enumerate(variants, start=1):
        print(f"{i}. {variant}")

    all_ranked_lists = []

    print("\nTop results per variant:")
    for variant in variants:
        results = search_documents(variant, documents, top_k=3)
        ranked_docs = [doc for doc, score in results]
        all_ranked_lists.append(ranked_docs)

        print(f"\nQuery: {variant}")
        for rank, (doc, score) in enumerate(results, start=1):
            print(f"{rank}. [score={score}] {doc}")

    fused = reciprocal_rank_fusion(all_ranked_lists)

    print("\nFinal fused ranking using RRF:")
    for rank, (doc, score) in enumerate(fused, start=1):
        print(f"{rank}. [RRF={score:.4f}] {doc}")

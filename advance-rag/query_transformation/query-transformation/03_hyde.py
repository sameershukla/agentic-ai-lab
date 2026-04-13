# 03_hyde.py

documents = [
    "Spark executor failed due to out of memory during shuffle stage.",
    "Increase spark.executor.memory and spark.executor.memoryOverhead to reduce OOM errors.",
    "Large joins and shuffles can create heavy memory pressure in Spark jobs.",
    "Database timeout error while calling external API.",
    "Use repartitioning carefully to avoid skew and excessive memory usage."
]


def generate_hypothetical_document(user_query: str) -> str:
    """
    In a real HyDE pipeline, an LLM writes a hypothetical answer/document.
    Here we simulate that behavior with a handcrafted example.
    """
    if "oom" in user_query.lower() and "spark" in user_query.lower():
        return (
            "Spark OOM issues usually happen when executor memory is too small, "
            "memory overhead is insufficient, or shuffle operations create high "
            "heap pressure. Common fixes include increasing executor memory, "
            "tuning partitions, and reducing large shuffles."
        )

    return f"A detailed answer about: {user_query}"


def search_documents(query: str, docs: list[str], top_k: int = 3) -> list[tuple[str, int]]:
    """
    Simple lexical search to keep the demo easy to understand.
    In a real system, this would usually be embedding-based retrieval.
    """
    query_terms = set(query.lower().split())
    scored = []

    for doc in docs:
        doc_terms = set(doc.lower().split())
        score = len(query_terms.intersection(doc_terms))
        scored.append((doc, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


if __name__ == "__main__":
    user_query = "How do I fix OOM in Spark?"

    print("User query:")
    print(user_query)

    direct_results = search_documents(user_query, documents, top_k=3)

    print("\nResults using raw query:")
    for rank, (doc, score) in enumerate(direct_results, start=1):
        print(f"{rank}. [score={score}] {doc}")

    hypothetical_doc = generate_hypothetical_document(user_query)

    print("\nGenerated hypothetical document:")
    print(hypothetical_doc)

    hyde_results = search_documents(hypothetical_doc, documents, top_k=3)

    print("\nResults using hypothetical document:")
    for rank, (doc, score) in enumerate(hyde_results, start=1):
        print(f"{rank}. [score={score}] {doc}")

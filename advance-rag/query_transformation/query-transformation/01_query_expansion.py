# 01_query_expansion.py

def expand_query(query: str) -> str:
    """
    Expand important terms in the query using a small manual synonym map.
    This is intentionally simple for learning purposes.
    """

    synonym_map = {
        "oom": [
            "out of memory",
            "memory exceeded",
            "heap overflow",
            "outofmemoryerror",
            "gc overhead"
        ],
        "spark": [
            "apache spark",
            "spark executor",
            "spark job",
            "spark application"
        ],
        "timeout": [
            "request timeout",
            "connection timeout",
            "read timeout"
        ]
    }

    words = query.lower().split()
    expanded_terms = []

    for word in words:
        expanded_terms.append(word)
        if word in synonym_map:
            expanded_terms.extend(synonym_map[word])

    # Remove duplicates while keeping order
    seen = set()
    final_terms = []
    for term in expanded_terms:
        if term not in seen:
            seen.add(term)
            final_terms.append(term)

    return " ".join(final_terms)


if __name__ == "__main__":
    original_query = "OOM error in Spark job"
    expanded_query = expand_query(original_query)

    print("Original query:")
    print(original_query)

    print("\nExpanded query:")
    print(expanded_query)

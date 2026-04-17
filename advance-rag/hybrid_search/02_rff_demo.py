vector_results = ["Fluent Python", "Python Cookbook", "Learning Python", "Automate with Python"]
keyword_results = ["Automate with Python", "Fluent Python", "Python Tricks", "Learning Python"]

ranked_lists = {
    "vector_results": vector_results,
    "keyword_results": keyword_results,
}

def rrf(ranked_lists: dict[str, list[str]], k: int = 60) -> dict[str, list[str]]:
    scores = {}
    for docs in ranked_lists.values():
        for rank, doc in enumerate(docs, start=1):
            scores[doc] = scores.get(doc, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

results = rrf(ranked_lists)

print(f"{'Rank':<6} {'Document':<25} {'RRF Score'}")
print("-" * 45)
for i, (doc, score) in enumerate(results):
    print(f"{i:<6} {doc:<25} {score:.5f}")

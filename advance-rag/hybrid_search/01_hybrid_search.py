from collections import defaultdict
import chromadb
from sentence_transformers import SentenceTransformer

# 1. Sample dataset
incidents = [
    {
        "id": "d6",
        "text": "Spark job terminated after the container ran out of memory.",
        "system": "spark",
        "severity": 1,
        "memory_gb": 11,
        "status": "failed"
    },
    {
        "id": "d7",
        "text": "The workload failed after exhausting available heap space.",
        "system": "spark",
        "severity": 4,
        "memory_gb": 10,
        "status": "failed"
    },
    {
        "id": "d8",
        "text": "Executor crashed because memory resources were depleted.",
        "system": "spark",
        "severity": 5,
        "memory_gb": 9,
        "status": "failed"
    }
]
# Query:
# semantic intent -> "OOM"
# structured conditions -> severity > 3, memory_gb < 13, status = failed
query = "OOM issue in Spark job where severity < 3 and memory_gb < 13 and status = failed"

where_filter = {
    "$and": [
        {"severity": {"$lt": 3}},      # greater than
        {"memory_gb": {"$lt": 13}},    # less than
        {"status": {"$eq": "failed"}}  # equals
    ]
}

# 2. Create ChromaDB collection
model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.Client()
collection = client.create_collection(name="hybrid_incident_demo")

documents = [item["text"] for item in incidents]
ids = [item["id"] for item in incidents]
metadatas = [
    {
        "system": item["system"],
        "severity": item["severity"],
        "memory_gb": item["memory_gb"],
        "status": item["status"]
    }
    for item in incidents
]

embeddings = model.encode(documents).tolist()

collection.add(
    ids=ids,
    documents=documents,
    metadatas=metadatas,
    embeddings=embeddings
)

# 3. Semantic search with metadata filtering
query_embedding = model.encode([query]).tolist()


semantic_response = collection.query(
    query_embeddings=query_embedding,
    n_results=5,
    where=where_filter
)

semantic_ids = semantic_response["ids"][0]
semantic_docs = semantic_response["documents"][0]

print("\nSEMANTIC SEARCH RESULTS")
print("-" * 50)
for rank, doc in enumerate(semantic_docs, start=1):
    print(f"{rank}. {doc}")

# 4. Simple keyword search with same filters
def passes_filter(item):
    return (
        item["severity"] > 3 and
        item["memory_gb"] < 13 and
        item["status"] == "failed"
    )

def keyword_search(query, items, top_k=5):
    query_terms = set(query.lower().split())
    scored = []

    for item in items:
        if not passes_filter(item):
            continue

        doc_terms = set(item["text"].lower().split())

        # simple overlap score
        score = len(query_terms.intersection(doc_terms))

        scored.append((item["id"], item["text"], score))

    scored.sort(key=lambda x: x[2], reverse=True)
    return scored[:top_k]

keyword_results = keyword_search(query, incidents, top_k=5)

keyword_ids = [row[0] for row in keyword_results]
keyword_docs = [row[1] for row in keyword_results]

print("\nKEYWORD SEARCH RESULTS")
print("-" * 50)
for rank, doc in enumerate(keyword_docs, start=1):
    print(f"{rank}. {doc}")

# 5. Reciprocal Rank Fusion (RRF)
def rrf_fusion(rank_lists, k=60):
    scores = defaultdict(float)

    for rank_list in rank_lists:
        for rank, doc_id in enumerate(rank_list, start=1):
            scores[doc_id] += 1 / (k + rank)

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

fused_results = rrf_fusion([semantic_ids, keyword_ids])

# helper map for printing
incident_map = {item["id"]: item for item in incidents}

print("\nHYBRID SEARCH RESULTS USING RRF")
print("-" * 50)
for rank, (doc_id, score) in enumerate(fused_results, start=1):
    item = incident_map[doc_id]
    print(
        f"{rank}. {item['text']} "
        f"(severity={item['severity']}, memory_gb={item['memory_gb']}, status={item['status']}) "
        f"-> RRF score: {score:.4f}"
    )

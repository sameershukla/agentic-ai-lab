"""
Hallucination Detection — Level 2: Score Threshold Monitoring
=============================================================
Every low retrieval similarity score is a hallucination risk signal.
This file shows how to:
  - Compute cosine similarity between a query and retrieved chunks
  - Flag responses whose similarity falls below a safe threshold
  - Log and monitor score trends over a batch of queries

Model : all-MiniLM-L6-v2  (fast, local, no API cost)
Threshold: 0.40  (standard for all-MiniLM-L6-v2 — tune for your corpus)

Run: python detect_l2_score_threshold.py
     (no API key required — embeddings run locally)
"""

import numpy as np
from sentence_transformers import SentenceTransformer

THRESHOLD = 0.40
MODEL_NAME = "all-MiniLM-L6-v2"
SEP = "=" * 65

print(f"\nLoading embedding model: {MODEL_NAME} ...")
model = SentenceTransformer(MODEL_NAME)
print("Model ready.\n")

# ── Knowledge base ────────────────────────────────────────────────────────────
CHUNKS = [
    {
        "id": "spark_aqe",
        "text": (
            "PySpark Adaptive Query Execution (AQE) tunes query plans at runtime. "
            "Enable with spark.conf.set('spark.sql.adaptive.enabled', 'true'). "
            "AQE coalesces shuffle partitions automatically. Requires Spark 3.0+."
        ),
    },
    {
        "id": "airflow_celery",
        "text": (
            "Airflow CeleryExecutor distributes tasks across a worker cluster. "
            "Requires a message broker such as Redis or RabbitMQ. "
            "Workers are scaled independently of the scheduler."
        ),
    },
    {
        "id": "snowflake_clone",
        "text": (
            "Snowflake zero-copy cloning creates an instant copy of a table, "
            "schema, or database without duplicating storage. "
            "Clones share the same underlying micro-partitions until data diverges."
        ),
    },
    {
        "id": "kafka_partitions",
        "text": (
            "Kafka partitions are the unit of parallelism. "
            "Each partition is an ordered, immutable sequence of records. "
            "Partition count cannot be decreased after topic creation."
        ),
    },
]

# Pre-compute chunk embeddings once
chunk_texts = [c["text"] for c in CHUNKS]
chunk_embeddings = model.encode(chunk_texts, normalize_embeddings=True)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))  # vectors are already L2-normalised


def retrieve_with_score(query: str) -> tuple[dict, float]:
    """Return the best matching chunk and its similarity score."""
    query_emb = model.encode([query], normalize_embeddings=True)[0]
    scores = [cosine_similarity(query_emb, ce) for ce in chunk_embeddings]
    best_idx = int(np.argmax(scores))
    return CHUNKS[best_idx], scores[best_idx]


def score_bar(score: float, width: int = 30) -> str:
    filled = int(score * width)
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {score:.3f}"


def evaluate_query(query: str) -> dict:
    chunk, score = retrieve_with_score(query)
    risk = score < THRESHOLD
    return {
        "query": query,
        "retrieved_chunk": chunk["id"],
        "score": score,
        "risk": risk,
    }


# ─────────────────────────────────────────────────────────────────────────────
# DEMO A — Single query, detailed output
# ─────────────────────────────────────────────────────────────────────────────
print(f"{SEP}")
print("DEMO A — Single query score inspection")
print(SEP)

query_a = "How does AQE automatically manage shuffle partitions in Spark?"
chunk_a, score_a = retrieve_with_score(query_a)

print(f"\n  Query     : {query_a}")
print(f"  Best chunk: '{chunk_a['id']}'")
print(f"  Chunk text: {chunk_a['text'][:90]}...")
print(f"\n  Similarity score: {score_bar(score_a)}")
print(f"  Threshold       : {THRESHOLD}")

if score_a >= THRESHOLD:
    print(f"\n  ✔  SAFE — score {score_a:.3f} is above threshold.")
    print("     Retrieval is strong. Grounded generation can proceed.")
else:
    print(f"\n  ⚠  RISK — score {score_a:.3f} is below threshold {THRESHOLD}.")
    print("     Retrieved chunk may not match the query.")
    print("     Answer is a hallucination risk. Consider: expand knowledge base,")
    print("     add a fallback response, or escalate to a human.")

# ─────────────────────────────────────────────────────────────────────────────
# DEMO B — Out-of-domain query (should score low → flag as risk)
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("DEMO B — Out-of-domain query (expect: low score, flagged as risk)")
print(SEP)

query_b = "What is the capital of France and what is its population?"
chunk_b, score_b = retrieve_with_score(query_b)

print(f"\n  Query     : {query_b}")
print(f"  Best chunk: '{chunk_b['id']}'  ← clearly wrong domain")
print(f"\n  Similarity score: {score_bar(score_b)}")
print(f"  Threshold       : {THRESHOLD}")

if score_b < THRESHOLD:
    print(f"\n  ⚠  RISK — score {score_b:.3f} is below threshold.")
    print("     The query is outside the indexed knowledge base.")
    print("     A model answering from parametric memory here will hallucinate.")
    print("     Correct action: return 'I don't have information on this topic.'")

# ─────────────────────────────────────────────────────────────────────────────
# DEMO C — Batch monitoring across multiple queries
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("DEMO C — Batch score monitoring  (simulates production logging)")
print(SEP)

TEST_QUERIES = [
    "How do I enable Adaptive Query Execution in PySpark 3?",
    "What broker does CeleryExecutor require?",
    "Explain zero-copy cloning in Snowflake.",
    "How are Kafka partitions ordered?",
    "What is the best way to train a neural network?",       # out of domain
    "How do I configure TLS in Nginx?",                     # out of domain
    "Can AQE be used with Spark 2.x?",
]

results = [evaluate_query(q) for q in TEST_QUERIES]
risk_results = [r for r in results if r["risk"]]
avg_score = np.mean([r["score"] for r in results])

print(f"\n  {'QUERY':<48} {'CHUNK':<22} {'SCORE':>6}  {'STATUS'}")
print(f"  {'─'*48} {'─'*22} {'─'*6}  {'─'*10}")
for r in results:
    status = "⚠  RISK" if r["risk"] else "✔  safe"
    print(f"  {r['query'][:47]:<48} {r['retrieved_chunk']:<22} {r['score']:>6.3f}  {status}")

print(f"\n  Average score      : {avg_score:.3f}")
print(f"  Below threshold    : {len(risk_results)} / {len(results)} queries")
print(f"  Risk rate          : {len(risk_results)/len(results)*100:.0f}%")

ALERT_THRESHOLD = 0.30  # alert if >30% of queries are at risk
if len(risk_results) / len(results) > ALERT_THRESHOLD:
    print(f"\n  ⚠  ALERT: Risk rate exceeds {ALERT_THRESHOLD*100:.0f}%.")
    print("     Knowledge base may not cover current query patterns.")
    print("     Actions: index new documents, expand corpus, or add fallback handling.")
else:
    print(f"\n  ✔  Risk rate within acceptable range.")

print(f"\n{SEP}")
print("KEY TAKEAWAY")
print(SEP)
print(f"""
  Log the retrieval score alongside every answer in production.
  Flag anything below {THRESHOLD} as a hallucination risk.

  Monitoring patterns to watch:
    - Average score dropping over time  → knowledge base becoming stale
    - Risk rate rising                  → query patterns drifting from indexed content
    - Sudden spike in low scores        → a new topic entering user queries

  Score thresholds by model (approximate):
    all-MiniLM-L6-v2    : 0.40
    all-mpnet-base-v2   : 0.45
    text-embedding-3-*  : 0.35

  Tune your threshold on a labelled eval set, not on defaults alone.
""")

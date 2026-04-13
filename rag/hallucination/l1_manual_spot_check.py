"""
Hallucination Detection — Level 1: Manual Spot-Check
=====================================================
The cheapest and most effective debugging habit in RAG development.
Print the retrieved context alongside every answer. If the answer
contains a claim that is not in the retrieved chunks, that is a
hallucination. This file shows both a passing and a failing case.

Run: ANTHROPIC_API_KEY=<your-key> python detect_l1_spot_check.py
"""

import os
import anthropic

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
MODEL = "claude-haiku-4-5-20251001"

SEP  = "=" * 65
DASH = "-" * 65

# ── Tiny in-memory knowledge base ────────────────────────────────────────────
# In production this would be a vector store. Here we keep it plain-text
# so the retrieval logic stays out of the way of the detection lesson.

KNOWLEDGE_BASE = {
    "airflow_executors": """
[Airflow Docs — Executors]
Available executors: SequentialExecutor, LocalExecutor, CeleryExecutor,
KubernetesExecutor.
LocalExecutor runs tasks in parallel using local processes.
CeleryExecutor distributes tasks across a worker cluster.
Default in Airflow 2.x: SequentialExecutor (single-process, no parallelism).
""",
    "spark_adaptive": """
[PySpark 3.5 Docs — Adaptive Query Execution]
Enable AQE: spark.conf.set("spark.sql.adaptive.enabled", "true")
AQE automatically coalesces shuffle partitions after each stage.
Default shuffle partitions: 200.
AQE requires Spark 3.0 or later.
""",
    "snowflake_clustering": """
[Snowflake Docs — Clustering Keys]
Clustering keys improve performance for large tables queried on specific columns.
Define with: ALTER TABLE t1 CLUSTER BY (col1, col2);
Automatic clustering is available on Enterprise edition and above.
Clustering is billed separately from storage and compute.
""",
}


def retrieve(query: str) -> tuple[str, str]:
    """
    Naive keyword retrieval — returns (chunk_id, chunk_text).
    Replace with a vector store query in production.
    """
    query_lower = query.lower()
    scores = {
        k: sum(word in query_lower for word in v.lower().split())
        for k, v in KNOWLEDGE_BASE.items()
    }
    best_key = max(scores, key=scores.get)
    return best_key, KNOWLEDGE_BASE[best_key]


def generate_answer(question: str, context: str) -> str:
    response = client.messages.create(
        model=MODEL,
        max_tokens=200,
        system="You are a data engineering assistant. Answer concisely.",
        messages=[{"role": "user", "content":
            f"Context:\n{context}\n\nQuestion: {question}\n\n"
            "Answer using ONLY the context above."
        }]
    )
    return response.content[0].text.strip()


def generate_answer_no_context(question: str) -> str:
    """Ungrounded call — hallucination risk."""
    response = client.messages.create(
        model=MODEL,
        max_tokens=200,
        system="You are a data engineering assistant. Answer concisely and specifically.",
        messages=[{"role": "user", "content": question}]
    )
    return response.content[0].text.strip()


def spot_check(question: str, context: str, answer: str) -> None:
    """
    Level 1 detection: print retrieved context and answer side by side.
    Flag any sentence in the answer whose keywords don't appear in the context.
    This is the thirty-second habit that catches most hallucinations in dev.
    """
    print(f"\n{'RETRIEVED CONTEXT':─^65}")
    print(context.strip())
    print(f"\n{'GENERATED ANSWER':─^65}")
    print(answer)
    print(f"\n{'SPOT-CHECK RESULTS':─^65}")

    context_words = set(context.lower().split())
    sentences = [s.strip() for s in answer.replace("\n", " ").split(".") if len(s.strip()) > 10]
    all_pass = True

    for sentence in sentences:
        sentence_words = set(sentence.lower().split())
        overlap = sentence_words & context_words
        # Flag if fewer than 3 content words overlap with the context
        if len(overlap) < 3:
            print(f"  ⚠  UNSUPPORTED: '{sentence[:80]}...'")
            print(f"     → Only {len(overlap)} words overlap with retrieved context.")
            all_pass = False
        else:
            print(f"  ✔  Supported: '{sentence[:80]}'")

    if all_pass:
        print("\n  VERDICT: Answer appears grounded in the retrieved context.")
    else:
        print("\n  VERDICT: Answer contains claims not traceable to retrieved context.")
        print("  ACTION : Review flagged sentences or tighten the system prompt.")


# ─────────────────────────────────────────────────────────────────────────────
# CASE 1 — GROUNDED  (answer should pass spot-check)
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("CASE 1 — Grounded answer  (expect: spot-check passes)")
print(SEP)

q1 = "What are the available Airflow executors and which is the default in 2.x?"
chunk_id, context1 = retrieve(q1)
print(f"\n  Question : {q1}")
print(f"  Retrieved: chunk '{chunk_id}'")

answer1 = generate_answer(q1, context1)
spot_check(q1, context1, answer1)

# ─────────────────────────────────────────────────────────────────────────────
# CASE 2 — UNGROUNDED  (answer likely fails spot-check)
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("CASE 2 — Ungrounded answer  (expect: spot-check flags hallucinations)")
print(SEP)

q2 = "What are the available Airflow executors and which is the default in 2.x?"
# We still retrieve context but do NOT pass it to the model — simulating a
# broken RAG pipeline where retrieval succeeds but context is dropped.
chunk_id, context2 = retrieve(q2)
print(f"\n  Question : {q2}")
print(f"  Retrieved: chunk '{chunk_id}'  (context NOT passed to model)")

answer2 = generate_answer_no_context(q2)
spot_check(q2, context2, answer2)

print(f"\n{SEP}")
print("KEY TAKEAWAY")
print(SEP)
print("""
  Always print retrieved context alongside every answer during development.
  If the spot-check flags a sentence, the pipeline has one of three problems:

    1. Wrong chunk retrieved — retrieval needs tuning
    2. Context dropped — a bug in the prompt assembly
    3. Model ignored context — tighten the system prompt instruction

  This single habit takes thirty seconds and catches most hallucination
  issues before they reach production.
""")

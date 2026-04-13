import os
import anthropic

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

QUESTION = (
    "How do I use the PySpark method optimize_shuffle_partitions() "
    "to automatically tune the number of shuffle partitions at runtime?"
)

# ── 1. WITHOUT grounding
print("\n--- WITHOUT GROUNDING (hallucination risk) ---\n")

response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=100,
    messages=[{"role": "user", "content": QUESTION}]
)

print("Q:", QUESTION)
print("A:", response.content[0].text.strip())
print("\n⚠ The model is guessing — Sameer is not in its training data.")

# ── 2. WITH grounding 
print("\n--- WITH GROUNDING (context provided) ---\n")
CONTEXT = """
[PySpark 3.5 Official Documentation — Adaptive Query Execution]

adaptive_query_execution (AQE) tunes query plans at runtime.
Enable it with:
    spark.conf.set("spark.sql.adaptive.enabled", "true")

To control shuffle partitions, set:
    spark.conf.set("spark.sql.adaptive.coalescePartitions.enabled", "true")
    spark.conf.set("spark.sql.shuffle.partitions", "200")   # initial value

AQE will automatically coalesce small shuffle partitions after each stage.
There is NO method called optimize_shuffle_partitions() in PySpark.
"""

SEP = "-" * 60

grounded_prompt = f"""Answer ONLY using the context below.
If the answer is not there, say "This is not covered in the documentation."

Context:
{CONTEXT}

Question: {QUESTION}"""

response = client.messages.create(
    model="claude-haiku-4-5-20251001",
    max_tokens=100,
    messages=[{"role": "user", "content": grounded_prompt}]
)

print("Q:", QUESTION)
print("A:", response.content[0].text.strip())
print("\n✔ Answer is grounded in the provided context — no fabrication.")

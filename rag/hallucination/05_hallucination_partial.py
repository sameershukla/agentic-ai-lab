import os
import anthropic

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
MODEL = "claude-haiku-4-5-20251001"

SEP = "=" * 65

def ask(system: str, user: str) -> str:
    r = client.messages.create(
        model=MODEL, max_tokens=180,
        system=system,
        messages=[{"role": "user", "content": user}]
    )
    return r.content[0].text.strip()


print(f"\n{SEP}")
print("EXAMPLE 4 — PARTIAL  (mixes correct facts with invented ones)")
print(SEP)

CONTEXT_4 = """
[PySpark 3.5 — DataFrame.write API]
Supported write modes: append, overwrite, ignore, error (default).
Supported formats: parquet, delta, csv, json, orc, avro.
partitionBy(*cols) partitions output by the specified columns.
"""

QUESTION_4 = (
    "Explain PySpark DataFrame write modes, supported formats, "
    "and how to use dynamic partition pruning with write."
)

print(f"\nContext contains: write modes, formats, partitionBy")
print(f"Context does NOT contain: dynamic partition pruning\n")

print("--- Without grounding (partial hallucination risk) ---")
print(ask(
    "You are a PySpark expert. Be specific and include code snippets.",
    QUESTION_4
))
print(
    "\n  PARTIAL: Write modes and formats may be correct (in training data)."
    "\n   But 'dynamic partition pruning with write' is a read-time optimization"
    "\n   misapplied here — any details about it are partially or fully invented."
)

print("\n--- With grounding ---")
print(ask(
    "Answer ONLY from the context. For anything not in the context, say "
    "'Not covered in the documentation.'",
    f"Context:\n{CONTEXT_4}\n\nQuestion: {QUESTION_4}"
))
print("\n  Grounded: model answers from the doc and flags dynamic partition")
print("   pruning as not covered rather than inventing an explanation.")

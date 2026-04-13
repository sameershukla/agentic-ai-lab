import os
import anthropic

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
MODEL = "claude-sonnet-4-20250514"

SEP = "=" * 65

def ask(system: str, user: str) -> str:
    r = client.messages.create(
        model=MODEL, max_tokens=180,
        system=system,
        messages=[{"role": "user", "content": user}]
    )
    return r.content[0].text.strip()


print(f"\n{SEP}")
print("EXAMPLE 1 — FABRICATION  (no source, pure invention)")
print(SEP)

QUESTION_1 = (
    "Show me how to use the Snowflake built-in function "
    "SEMANTIC_CLUSTER_MATCH() to group similar customer records."
)

print(f"\nQuestion: {QUESTION_1}\n")
print("--- Without grounding ---")
print(ask(
    "You are a Snowflake SQL expert. Answer concisely with a code example.",
    QUESTION_1
))
print(
    "\n   SEMANTIC_CLUSTER_MATCH() does not exist in Snowflake."
    "\n   The model invented the function name, parameters, and usage — fabrication."
)

print("\n--- With grounding ---")
CONTEXT_1 = """
[Snowflake Documentation — Available AI/ML Functions]
Snowflake does not have a function called SEMANTIC_CLUSTER_MATCH().
For grouping similar records, use:
  - VECTOR_COSINE_SIMILARITY(v1, v2)  : cosine similarity between two vectors
  - SOUNDEX(str)                       : phonetic similarity for name matching
  - EDITDISTANCE(str1, str2)           : Levenshtein distance between strings
Entity resolution workflows typically combine these with a JOIN and a
similarity threshold (e.g., EDITDISTANCE < 3).
"""
print(ask(
    "Answer ONLY from the context. If not present, say 'Not in documentation.'",
    f"Context:\n{CONTEXT_1}\n\nQuestion: {QUESTION_1}"
))
print("\n  Grounded: model correctly says the function doesn't exist and suggests real alternatives.")

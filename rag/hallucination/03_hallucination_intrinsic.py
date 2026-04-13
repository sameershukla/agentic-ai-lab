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
print("EXAMPLE 2 — INTRINSIC  (contradicts the provided context)")
print(SEP)

CONTEXT_2 = """
[Release Notes — Apache Airflow 2.9]
The default executor in Apache Airflow 2.9 was changed from
SequentialExecutor to LocalExecutor.
SequentialExecutor is now only recommended for development and testing.
LocalExecutor supports parallel task execution using multiple processes.
"""

QUESTION_2 = "What is the default executor in Apache Airflow 2.9?"

print(f"\nContext says: default executor = LocalExecutor")
print(f"Question: {QUESTION_2}\n")

print("--- Without grounding (intrinsic hallucination risk) ---")
print(ask(
    "You are an Airflow expert. Answer confidently and specifically.",
    QUESTION_2
))
print(
    "\n  INTRINSIC: If the model answers 'SequentialExecutor' it directly"
    "\n   contradicts the provided context, which clearly states LocalExecutor."
)

print("\n--- With grounding ---")
print(ask(
    "Answer ONLY from the context below. Do not add outside knowledge.",
    f"Context:\n{CONTEXT_2}\n\nQuestion: {QUESTION_2}"
))
print("\n  Grounded: model correctly returns LocalExecutor as stated in the context.")

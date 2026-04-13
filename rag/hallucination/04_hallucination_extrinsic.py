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
print("EXAMPLE 3 — EXTRINSIC  (adds facts not present in the context)")
print(SEP)

CONTEXT_3 = """
[Internal Architecture Doc — DataPipeline v3]
The ingestion layer uses Apache Kafka with 12 partitions.
Average message throughput is 80,000 events per second.
"""

QUESTION_3 = (
    "Describe the DataPipeline v3 architecture: "
    "Kafka setup, throughput, retention policy, and consumer group design."
)
print(f"\nContext contains: partitions=12, throughput=80k/sec")
print(f"Context does NOT contain: retention policy, consumer group design\n")

print("--- Without grounding (extrinsic hallucination risk) ---")
print(ask(
    "You are a data architect. Give a detailed, specific answer.",
    QUESTION_3
))
print(
    "\n   EXTRINSIC: Retention policy and consumer group details are not in"
    "\n   the context. Any specific claims about them are invented additions."
)

print("\n--- With grounding ---")
print(ask(
    "Answer ONLY from the context. For anything not covered, explicitly say "
    "'Not specified in the document.'",
    f"Context:\n{CONTEXT_3}\n\nQuestion: {QUESTION_3}"
))
print("\n  Grounded: model answers what's in the doc and flags what's missing.")

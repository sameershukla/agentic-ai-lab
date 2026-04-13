"""
Hallucination Detection — Level 3: LLM-as-Judge
================================================
After generating an answer, a second LLM call asks:
  "Given only this context, is this answer fully supported?"

This is the most reliable automated detection technique available.
The tradeoff is cost — it roughly doubles your API spend per query.
Use it during evaluation and testing, not on every production request.

This file runs three cases:
  Case 1 — Grounded answer   : judge should PASS
  Case 2 — Hallucinated answer: judge should FAIL with specific claims listed
  Case 3 — Partial answer    : judge should PARTIAL with mixed verdict

Run: ANTHROPIC_API_KEY=<your-key> python detect_l3_llm_as_judge.py
"""

import os
import json
import anthropic

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
MODEL = "claude-haiku-4-5-20251001"
SEP   = "=" * 65

# ── Shared context ────────────────────────────────────────────────────────────

CONTEXT = """
[Snowflake Documentation — Zero-Copy Cloning]
Zero-copy cloning creates an instant snapshot of a table, schema, or database.
Clones share the underlying micro-partitions with the source until data diverges.
Storage is only billed for data written after the clone is created.
Cloning is available on all Snowflake editions.
Syntax: CREATE TABLE t_clone CLONE t_source;
Zero-copy cloning does NOT copy data streams or tasks.
"""


def generate_answer(question: str, context: str, grounded: bool = True) -> str:
    """Generate an answer — grounded or ungrounded depending on the flag."""
    if grounded:
        prompt = (
            f"Answer using ONLY the context below.\n\n"
            f"Context:\n{context}\n\nQuestion: {question}"
        )
        system = "You are a Snowflake expert. Be specific and concise."
    else:
        prompt = question
        system = (
            "You are a Snowflake expert. Give a detailed, specific answer "
            "with concrete facts, edition requirements, and limitations."
        )
    r = client.messages.create(
        model=MODEL, max_tokens=220, system=system,
        messages=[{"role": "user", "content": prompt}]
    )
    return r.content[0].text.strip()


def llm_as_judge(context: str, answer: str) -> dict:
    """
    Second LLM call that acts as a faithfulness judge.
    Returns a structured verdict: supported / unsupported claims.
    """
    judge_system = """You are a strict hallucination auditor.
Given a CONTEXT and an ANSWER, evaluate whether every factual claim in the
answer is directly supported by the context.

Return ONLY a JSON object with this exact structure — no prose, no fences:
{
  "verdict": "PASS" | "FAIL" | "PARTIAL",
  "supported_claims": ["..."],
  "unsupported_claims": ["..."],
  "reasoning": "one sentence explanation"
}

Rules:
- PASS    : every claim in the answer is supported by the context
- FAIL    : one or more claims directly contradict or are absent from context
- PARTIAL : some claims are supported, others are not
- Be specific: quote the exact phrase from the answer that is unsupported."""

    judge_prompt = f"CONTEXT:\n{context}\n\nANSWER:\n{answer}"

    r = client.messages.create(
        model=MODEL, max_tokens=400, system=judge_system,
        messages=[{"role": "user", "content": judge_prompt}]
    )
    raw = r.content[0].text.strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Fallback: return raw text if model didn't produce clean JSON
        return {"verdict": "PARSE_ERROR", "raw": raw}


def print_verdict(result: dict) -> None:
    verdict = result.get("verdict", "UNKNOWN")
    icon = {"PASS": "✔", "FAIL": "⚠", "PARTIAL": "◑"}.get(verdict, "?")
    print(f"\n  {icon}  VERDICT: {verdict}")

    if result.get("reasoning"):
        print(f"     Reasoning: {result['reasoning']}")

    if result.get("supported_claims"):
        print(f"\n  Supported claims ({len(result['supported_claims'])}):")
        for c in result["supported_claims"]:
            print(f"    ✔  {c}")

    if result.get("unsupported_claims"):
        print(f"\n  Unsupported / hallucinated claims ({len(result['unsupported_claims'])}):")
        for c in result["unsupported_claims"]:
            print(f"    ⚠  {c}")

    if result.get("raw"):
        print(f"\n  Raw output: {result['raw']}")


# ─────────────────────────────────────────────────────────────────────────────
# CASE 1 — Grounded answer  → judge should return PASS
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("CASE 1 — Grounded answer  (expect: judge returns PASS)")
print(SEP)

question = "How does zero-copy cloning work in Snowflake and what are its limitations?"

print(f"\n  Question: {question}")
print("\n  Step 1 — Generating grounded answer ...")
answer_grounded = generate_answer(question, CONTEXT, grounded=True)
print(f"\n  Answer:\n  {answer_grounded.replace(chr(10), chr(10) + '  ')}")

print("\n  Step 2 — Running LLM-as-judge ...")
result1 = llm_as_judge(CONTEXT, answer_grounded)
print_verdict(result1)

# ─────────────────────────────────────────────────────────────────────────────
# CASE 2 — Hallucinated answer  → judge should return FAIL
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("CASE 2 — Hallucinated answer  (expect: judge returns FAIL)")
print(SEP)

print(f"\n  Question: {question}")
print("\n  Step 1 — Generating ungrounded (hallucinated) answer ...")
answer_hallucinated = generate_answer(question, CONTEXT, grounded=False)
print(f"\n  Answer:\n  {answer_hallucinated.replace(chr(10), chr(10) + '  ')}")

print("\n  Step 2 — Running LLM-as-judge ...")
result2 = llm_as_judge(CONTEXT, answer_hallucinated)
print_verdict(result2)

# ─────────────────────────────────────────────────────────────────────────────
# CASE 3 — Manually crafted partial answer  → judge should return PARTIAL
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("CASE 3 — Partial answer  (expect: judge returns PARTIAL)")
print(SEP)

# This answer mixes a correct claim (shared micro-partitions) with
# an invented claim (available only on Enterprise edition).
answer_partial = (
    "Zero-copy cloning in Snowflake creates an instant snapshot without "
    "duplicating storage — clones share micro-partitions until data diverges. "
    "This feature is available only on Enterprise edition and above. "
    "Cloning does not copy data streams or tasks."
)

print(f"\n  Question: {question}")
print(f"\n  Answer (manually crafted to mix correct and invented claims):")
print(f"  {answer_partial.replace(chr(10), chr(10) + '  ')}")

print("\n  Running LLM-as-judge ...")
result3 = llm_as_judge(CONTEXT, answer_partial)
print_verdict(result3)

# ─────────────────────────────────────────────────────────────────────────────
# Summary: cost model and usage guidance
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("COST MODEL & USAGE GUIDANCE")
print(SEP)
print("""
  LLM-as-judge makes two API calls per user query:
    Call 1 — Generate the answer        (~200 output tokens)
    Call 2 — Judge the answer           (~400 output tokens)

  This roughly doubles API cost per query.

  Recommended usage pattern:
    Development / evaluation  : judge every query — catch issues early
    CI test suite             : judge a labelled golden set on every deploy
    Production (high-stakes)  : judge every query if budget allows
    Production (standard)     : rely on score thresholds (Level 2) as primary
                                defence; sample 5-10% of traffic for judging

  The judge prompt can be tuned to return:
    - A binary PASS/FAIL         (cheapest, fastest)
    - A confidence score 0-1     (more nuanced, easier to threshold)
    - Specific unsupported claims (most actionable, costs more tokens)

  All three cases above use the most actionable format: specific claims.
""")

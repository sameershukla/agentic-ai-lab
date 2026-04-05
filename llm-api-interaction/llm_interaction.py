import anthropic

# ── Client setup ──────────────────────────────────────────────
# Reads ANTHROPIC_API_KEY from environment automatically
client = anthropic.Anthropic()

# ── System prompt ──────────────────────────────────────────────
# Written once. Sets persona, rules, output format.
# Never changes between requests.
system = """
You are a precise medical clinic assistant.

Answer using ONLY the information provided in the user message.
After every factual claim, cite the source like [Doc 1].

If the answer is not in the provided information, say:
"This is not covered in our guidelines. Please speak with a doctor."

Never estimate or use outside medical knowledge.
Keep answers concise — 2 to 4 sentences maximum.
"""

# ── Retrieved context (from RAG pipeline) ──────────────────────
# This would come from retriever.retrieve(question) in production.
# Shown here as a static string for clarity.
retrieved_context = """
[Doc 1] Paracetamol Dosage Policy:
Adults may take 500mg to 1000mg every 4 to 6 hours.
Maximum daily dose: 4000mg.
Liver disease patients: 2000mg max.
"""

# ── User message ────────────────────────────────────────────────
# Combines retrieved context + the actual question.
# Changes on every request.
question = "How much paracetamol can I take today?"

user_message = f"""
=== CLINIC GUIDELINES ===
{retrieved_context}

=== PATIENT QUESTION ===
{question}
"""

# ── API call with streaming ─────────────────────────────────────
print(f"Q: {question}\nA: ", end="")

with client.messages.stream(
    model="claude-sonnet-4-20250514",  # pinned model
    max_tokens=200,                   # short answer ceiling
    temperature=0.0,                 # deterministic — facts only
    system=system,                   # persona + rules
    messages=[                       # single-turn call
        {
            "role": "user",
            "content": user_message
        }
    ]
) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)

print()  # final newline

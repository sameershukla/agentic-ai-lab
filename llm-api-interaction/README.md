# Grounded Answer Generation with Anthropic

This example shows how to generate a grounded answer using Anthropic with a simple RAG-style prompt.

The script demonstrates three important ideas:

- a **system prompt** that sets rules and behavior
- **retrieved context** inserted into the user message
- **streaming output** so the answer appears token by token

The model is instructed to answer **only from the provided context**, cite every factual claim, and fall back safely if the answer is not covered.

---

## What this example is teaching

A real RAG application usually works like this:

1. User asks a question
2. Retriever finds relevant documents
3. Retrieved context is added to the prompt
4. LLM generates an answer grounded in that context

This example focuses only on the **generation step**.

Instead of using a live retriever, it uses a static `retrieved_context` string so the prompt structure is easy to understand.

---

## Files

```text
.
├── requirements.txt
├── README.md
└── your_script.py

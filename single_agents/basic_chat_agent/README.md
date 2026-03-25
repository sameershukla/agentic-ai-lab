# 🧠 01 - Basic Chat Agent (LangGraph)

This is the **first and simplest agent** in the repository.

It demonstrates how to build a **basic conversational AI agent using LangGraph**, focusing on:
- state management
- graph execution
- message flow
- LLM interaction

---

## 🎯 What This Agent Does

- Takes user input from the terminal
- Maintains conversation history
- Sends messages to the LLM
- Returns AI responses
- Continues the conversation

👉 This is a **stateful chat agent**, not just a one-shot prompt.

---

## 🧩 Architecture Overview

```text
User Input → State (messages) → LangGraph → Chatbot Node → LLM → Response → State Updated
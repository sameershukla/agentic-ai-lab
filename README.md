# Agentic AI Lab

Companion code repository for the book  
**"From Tokens to Agents: Understanding, Building, and Scaling Modern AI Systems"**  
by Sameer Shukla

Every folder maps directly to a part of the book. Clone once, follow chapter by chapter.

---

## Who This Is For

- Developers moving from basic LLM API calls toward production-grade agentic systems
- Data engineers and architects entering the AI space
- Anyone reading the book who wants hands-on code alongside every concept

---

## Repository Structure

```
agentic-ai-lab/
│
├── llm-api-interaction/     # Part 1 — Talking to LLMs
├── llm_tools/               # Part 1 — Function calling & tool use
├── memory/                  # Part 1 — Conversation & agent memory
├── prompt_template/         # Part 1 — Prompt engineering patterns
│
├── rag/                     # Part 2 — RAG fundamentals
├── advanced-rag/            # Part 3 — Hybrid search, reranking, evaluation
│
├── agentic_ai/              # Part 4 & 5 — Agents, ReAct, multi-agent
│
└── end-to-end/              # Part 6 — Production-grade full system
```

---

## Learning Path

Start here and follow in order — each module builds directly on the previous one.

```
llm-api-interaction   →   Understand how to talk to LLMs cleanly
        ↓
llm_tools             →   Give LLMs the ability to call functions and use tools
        ↓
memory                →   Make LLMs remember across turns and sessions
        ↓
prompt_template       →   Engineer prompts that produce reliable outputs
        ↓
rag                   →   Ground LLMs in your own private, current data
        ↓
advanced-rag          →   Make retrieval smarter with hybrid search and reranking
        ↓
agentic_ai            →   Build agents that reason, plan, and act autonomously
        ↓
end-to-end            →   Combine everything into a production-ready system
```

---

## Getting Started

```bash
# 1. Clone the repo
git clone https://github.com/sameershukla/agentic-ai-lab.git
cd agentic-ai-lab

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables
cp .env.example .env
# Add your API keys to .env
```

---

## Environment Variables

```bash
# .env
ANTHROPIC_API_KEY=sk-ant-your-key-here
OPENAI_API_KEY=sk-your-key-here
COHERE_API_KEY=your-cohere-key     # for reranking examples
```

Your API key is never written into any source file. It is always read from the environment.

---

## Module Overview

### `llm-api-interaction`
Direct LLM API calls — completions, streaming, structured outputs, and multi-turn conversations.  
*Book: Part 1*

### `llm_tools`
Function calling, tool definitions, and how agents use external tools to take action.  
*Book: Part 1*

### `memory`
Buffer memory, summary memory, and vector memory — how agents remember across turns.  
*Book: Part 1*

### `prompt_template`
Prompt templates, few-shot examples, chain-of-thought, and output parsers.  
*Book: Part 1*

### `rag`
Naive RAG from scratch — chunking strategies, FAISS indexing, retrieval, and prompt augmentation.  
*Book: Part 2*

### `advanced-rag`
Hybrid search (BM25 + dense), cross-encoder reranking, query transformation, hallucination detection, and RAG evaluation with RAGAS.  
*Book: Part 3*

### `agentic_ai`
ReAct agents, tool-use agents, NL-to-SQL agent with self-correction, multi-agent collaboration, and agentic RAG.  
*Book: Part 4 & 5*

### `end-to-end`
A production-grade intelligent assistant combining LLM API, RAG, memory, tools, and an agent orchestration layer into one deployable system.  
*Book: Part 6*

---

## Tech Stack

| Component | Tool |
|---|---|
| LLM Clients | Anthropic Claude, OpenAI |
| Agent Framework | LangGraph, LangChain |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) |
| Vector Store | FAISS (local) |
| Hybrid Search | BM25 + dense retrieval |
| Reranking | Cohere Rerank |
| RAG Evaluation | RAGAS |
| Language | Python 3.10+ |

---

## About the Author

**Sameer Shukla** — Director of Data & AI Architecture  

[LinkedIn]([https://linkedin.com/in/sameershukla](https://www.linkedin.com/in/sameershukla30/)) · [GitHub](https://github.com/sameershukla) · [Book]([https://bpbonline.com](https://www.freecodecamp.org/news/how-to-optimize-pyspark-jobs-handbook/))

---

## License

MIT — free to use, modify, and distribute with attribution.

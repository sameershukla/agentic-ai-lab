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
├── llm-api-interaction/          # Part 4 — Talking to LLMs
├── llm_tools/                    # Part 4 — Function calling & tool use
├── memory/                       # Part 4 — Conversation & agent memory
├── prompt_template/              # Part 4 — Prompt engineering patterns
│
├── rag/                          # Part 2 — RAG fundamentals
│
├── advance-rag/                  # Part 3 — Advanced retrieval patterns
│   ├── agentic_rag/              #   Agentic RAG with retrieval agents
│   ├── hybrid_search/            #   BM25 + dense vector hybrid retrieval
│   ├── query_transformation/     #   Query rewriting and expansion
│   ├── rag_evaluation/           #   RAGAS-based evaluation pipelines
│   └── reranking/                #   Cross-encoder reranking strategies
│
├── agentic_ai/                   # Part 5 — Agents & multi-agent systems
│   ├── 01_minimal_langgraph_chatbot.py
│   ├── 02_two_node_workflow.py
│   ├── 03_conditional_routing.py
│   ├── 04_tool_calling_react_agent.py
│   ├── 05_memory_with_message_history.py
│   ├── 06_human_approval_agent.py
│   ├── 07_supervisor_multi_agent.py
│   ├── 08_multi_tool_agent.py
│   └── 09_reflection_agent.py
│
└── end-to-end/                   # Part 6 — Production-grade full system
```

---

## Learning Path

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
advance-rag           →   Make retrieval smarter with hybrid search and reranking
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
*Book: Part 4*

### `llm_tools`
Function calling, tool definitions, and how agents use external tools to take action.  
*Book: Part 4*

### `memory`
Buffer memory, summary memory, and vector memory — how agents remember across turns.  
*Book: Part 4*

### `prompt_template`
Prompt templates, few-shot examples, chain-of-thought, and output parsers.  
*Book: Part 4*

### `rag`
Naive RAG from scratch — chunking strategies, FAISS indexing, retrieval, and prompt augmentation.  
*Book: Part 2*

### `advance-rag`
Advanced retrieval engineering across five focused sub-modules:

| Sub-module | What it covers |
|---|---|
| `agentic_rag` | Retrieval agents that decide when and how to retrieve |
| `hybrid_search` | BM25 + dense vector retrieval combined for higher recall |
| `query_transformation` | Query rewriting, decomposition, and HyDE |
| `rag_evaluation` | End-to-end RAG evaluation with RAGAS (faithfulness, relevance, recall) |
| `reranking` | Cross-encoder reranking to push the most relevant chunks to the top |

*Book: Part 3*

### `agentic_ai`
A progressive, numbered sequence from a minimal LangGraph chatbot to a full reflection agent. Each file is self-contained and runnable independently.

| File | What it teaches |
|---|---|
| `01_minimal_langgraph_chatbot.py` | Minimal LangGraph state machine with a single LLM node |
| `02_two_node_workflow.py` | Two-node graph with explicit state transitions |
| `03_conditional_routing.py` | Conditional edges — routing decisions based on agent output |
| `04_tool_calling_react_agent.py` | ReAct agent with tool use and observe/act cycles |
| `05_memory_with_message_history.py` | Persistent memory across turns using message history |
| `06_human_approval_agent.py` | Human-in-the-loop: interrupt, review, and resume workflows |
| `07_supervisor_multi_agent.py` | Supervisor pattern — orchestrating multiple specialized subagents |
| `08_multi_tool_agent.py` | Agent with multiple tools and dynamic tool selection |
| `09_reflection_agent.py` | Self-critique and reflection loops for improved output quality |

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

[LinkedIn](https://www.linkedin.com/in/sameershukla30/) · [GitHub](https://github.com/sameershukla) · [Book](https://www.freecodecamp.org/news/how-to-optimize-pyspark-jobs-handbook/)

---

## License

MIT — free to use, modify, and distribute with attribution.

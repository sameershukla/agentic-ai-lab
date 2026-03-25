# 01 · Basic ReAct Agent

The simplest possible LangGraph agent. Nothing hidden, nothing abstracted away.
Start here before looking at any other agent in this repo.

---

## What It Does

Takes a plain English question, decides which tools to use (if any), calls them,
and returns a final answer. It loops until the LLM is satisfied.

**Available tools:**
| Tool | What it does |
|------|-------------|
| `add(a, b)` | Adds two numbers |
| `multiply(a, b)` | Multiplies two numbers |
| `get_current_datetime()` | Returns current date and time |

---

## The Graph

```
START → llm_node → should_continue? ──── "tools" ──→ tool_node ──┐
                          │                                        │
                        "end"                                      │
                          │                                        │
                         END           ◄──────────────────────────┘
                                         (loops back to llm_node)
```

This pattern is called **ReAct** (Reason + Act). The LLM reasons about what to do,
acts by calling a tool, observes the result, then reasons again.

---

## File Structure

```
01_basic_react_agent/
├── state.py        ← AgentState: the shared memory flowing through all nodes
├── tools.py        ← Tool definitions (@tool decorated functions)
├── agent.py        ← Graph construction: nodes, edges, conditional routing
├── run.py          ← Entry point with step-by-step verbose output
└── requirements.txt
```

---

## Key Concepts Demonstrated

| Concept | Where |
|---------|-------|
| `TypedDict` state schema | `state.py` |
| `add_messages` reducer | `state.py` |
| `@tool` decorator | `tools.py` |
| `bind_tools` | `agent.py` |
| `ToolNode` (prebuilt) | `agent.py` |
| Conditional edge routing | `agent.py → should_continue()` |
| `stream_mode="values"` | `run.py` |

---

## Setup & Run

```bash
# Install dependencies
pip install -r requirements.txt

# Set your API key
export ANTHROPIC_API_KEY=your_key_here

# Run the agent
python run.py
```

---

## Sample Output

```
────────────────────────────────────────────────
  USER: What is 47 multiplied by 13?
────────────────────────────────────────────────

[Step 1] HumanMessage
  What is 47 multiplied by 13?

[Step 2] AIMessage
  (tool call in progress...)
  → calling tool: multiply({'a': 47.0, 'b': 13.0})

[Step 3] ToolMessage
  611.0

[Step 4] AIMessage
  47 multiplied by 13 is **611**.
```

---

## What to Study Next

Once this is clear, move to:
- `02_nl_to_sql_agent` — adds **reflection/retry loops** and richer state fields
- `03_incident_triage_agent` — adds **parallel tools** and **human-in-the-loop**

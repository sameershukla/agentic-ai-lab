"""
agent.py — Builds and compiles the LangGraph agent graph.

Graph structure (ReAct pattern):

    START
      │
      ▼
  ┌──────────┐
  │  llm_node │  ◄─────────────────────┐
  └──────────┘                         │
       │                               │
       ▼                               │
  should_continue?                     │
  ├── "tools"  → ┌──────────────┐      │
  │              │  tool_node   │──────┘
  │              └──────────────┘
  └── "end"    → END

The loop runs until the LLM produces a response with NO tool calls,
at which point `should_continue` routes to END.
"""

from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

from state import AgentState
from tools import tools

# 1. LLM Setup
llm = ChatAnthropic(model="claude-sonnet-4-20250514")
llm_with_tools = llm.bind_tools(tools)


# 2. Node definitions
def llm_node(state: AgentState) -> dict:
    """
        The 'brain' node. Sends all messages to the LLM and gets back either:
          - A plain text response  (loop ends)
          - A tool_call response   (loop continues via tool_node)
    """
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


# ToolNode is a prebuilt LangGraph node that:
#   1. Reads the tool_calls from the last AIMessage
#   2. Executes each tool
#   3. Returns ToolMessages with the results
tool_node = ToolNode(tools)


# 3. Routing Logic
def should_continue(state: AgentState) -> str:
    """
    Conditional edge function — decides what happens after llm_node.

    Returns:
        "tools"  → if the LLM wants to call a tool  (keep looping)
        "end"    → if the LLM gave a final answer    (stop)
    """
    last_message = state["messages"][-1]

    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"

    return "end"


# 4. Build the graph
def build_graph():
    graph = StateGraph(AgentState)

    # Register nodes
    graph.add_node("llm_node", llm_node)
    graph.add_node("tool_node", tool_node)

    # Entry point, always start with llm node
    graph.add_edge(START, "llm_node")

    # After llm_node: route conditionally
    graph.add_conditional_edges(
        "llm_node",
        should_continue,
        {
            "tools": "tool_node",  # tool call detected → run tools
            "end": END  # final answer → stop
        }
    )

    # After tool_node: always go back to llm_node
    graph.add_edge("tool_node", "llm_node")

    return graph.compile()


# Compile once and export
agent = build_graph()

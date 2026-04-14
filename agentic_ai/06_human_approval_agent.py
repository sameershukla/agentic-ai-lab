"""
Human-in-the-Loop Agent — LangGraph
=====================================
Book companion: "From Tokens to Agents" — Part 4 (Agentic AI)

Pattern: The agent plans a tool call (e.g. book a flight), then PAUSES
and asks the human to approve or reject it before executing.

Key LangGraph concepts used:
  - StateGraph + TypedDict state
  - MemorySaver  → persists state across interrupt/resume
  - interrupt()  → suspends the graph mid-node for human input
  - graph.invoke(None, config) → resumes from the saved checkpoint

Flow:
    User Query
        ↓
    [agent] — LLM decides what tool to call
        ↓
    [human_review] — graph PAUSES here; human sees proposed action
        ↓  (approved)
    [run_tool] — tool executes
        ↓
    [agent] — LLM composes final answer
        ↓
    END
"""

from typing import Annotated, TypedDict, Literal
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt


# ── 1. State ─────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]   # full conversation history
    pending_tool_call: dict | None            # tool call awaiting approval


# ── 2. Tools ─────────────────────────────────────────────────────────────────

@tool
def book_flight(origin: str, destination: str, date: str) -> str:
    """Book a flight ticket between two cities on a given date."""
    # In production this calls an airline API.
    return f" Flight booked: {origin} → {destination} on {date}. Ref: FL-{hash(destination) % 9999:04d}"


@tool
def check_weather(city: str) -> str:
    """Check current weather for a city."""
    # Stub — replace with a real weather API call.
    return f"🌤  {city}: 22°C, partly cloudy"


tools = [book_flight, check_weather]
tools_by_name = {t.name: t for t in tools}

# ── 3. LLM ───────────────────────────────────────────────────────────────────

llm = ChatAnthropic(model="claude-sonnet-4-20250514").bind_tools(tools)


# ── 4. Nodes ─────────────────────────────────────────────────────────────────

def agent_node(state: AgentState) -> dict:
    """Call the LLM and let it decide the next action (tool or final answer)."""
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


def human_review_node(state: AgentState) -> dict:
    """
    Pause here and surface the proposed tool call to the human.
    `interrupt()` serialises state into the checkpointer and raises an
    exception that LangGraph catches — the graph stays suspended until
    the caller resumes it by passing a Command(resume=...).
    """
    last_message: AIMessage = state["messages"][-1]
    tool_call = last_message.tool_calls[0]            # first proposed tool call

    # interrupt() pauses execution and returns whatever value the
    # human sends back when they resume (True = approved, False = rejected).
    approved = interrupt({
        "action":  tool_call["name"],
        "args":    tool_call["args"],
        "message": f"Agent wants to call `{tool_call['name']}` with {tool_call['args']}. Approve? (y/n)"
    })

    if approved:
        return {"pending_tool_call": tool_call}       # approved → store for execution
    else:
        # Inject a rejection notice into the message stream so the LLM
        # can explain to the user that the action was cancelled.
        rejection_msg = ToolMessage(
            content="Action rejected by user.",
            tool_call_id=tool_call["id"]
        )
        return {"messages": [rejection_msg], "pending_tool_call": None}


def run_tool_node(state: AgentState) -> dict:
    """Execute the approved tool call and return the result."""
    tool_call = state["pending_tool_call"]
    tool_fn   = tools_by_name[tool_call["name"]]
    result    = tool_fn.invoke(tool_call["args"])

    tool_msg = ToolMessage(
        content=str(result),
        tool_call_id=tool_call["id"]
    )
    return {"messages": [tool_msg], "pending_tool_call": None}


# ── 5. Routing ────────────────────────────────────────────────────────────────

def route_after_agent(state: AgentState) -> Literal["human_review", END]:
    """If the LLM produced a tool call → pause for review; otherwise finish."""
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "human_review"
    return END


def route_after_review(state: AgentState) -> Literal["run_tool", "agent"]:
    """After human review: run the tool if approved, re-prompt agent if rejected."""
    if state.get("pending_tool_call"):
        return "run_tool"
    return "agent"          # LLM will now compose an apology / alternative


# ── 6. Graph ──────────────────────────────────────────────────────────────────

checkpointer = MemorySaver()  # in-memory; swap for SqliteSaver in production

builder = StateGraph(AgentState)

builder.add_node("agent",        agent_node)
builder.add_node("human_review", human_review_node)
builder.add_node("run_tool",     run_tool_node)

builder.add_edge(START,        "agent")
builder.add_conditional_edges("agent",        route_after_agent)
builder.add_conditional_edges("human_review", route_after_review)
builder.add_edge("run_tool",   "agent")   # after tool runs, LLM composes final reply

graph = builder.compile(checkpointer=checkpointer)


# ── 7. Runner ─────────────────────────────────────────────────────────────────

def run(user_input: str, thread_id: str = "thread-1") -> None:
    """
    Drive the graph through a full interrupt → human approval → resume cycle.
    In a real app, the interrupt pause would be a UI prompt, Slack message,
    email approval, etc.  Here we simulate with a console input().
    """
    config = {"configurable": {"thread_id": thread_id}}

    print(f"\n{'─'*55}")
    print(f"User: {user_input}")
    print(f"{'─'*55}")

    # ── First run: graph runs until it hits interrupt() ────────────────────
    events = graph.stream(
        {"messages": [HumanMessage(content=user_input)], "pending_tool_call": None},
        config,
        stream_mode="values"
    )

    interrupt_payload = None
    for event in events:
        last = event["messages"][-1]
        # Print intermediate AI thoughts (text before tool call)
        if isinstance(last, AIMessage) and last.content:
            print(f"\nAgent (thinking): {last.content}")

    # After streaming, check graph state — did it pause at an interrupt?
    state = graph.get_state(config)

    if state.next:  # graph is suspended — there's a pending interrupt
        # Extract the interrupt payload surfaced by human_review_node
        for task in state.tasks:
            if task.interrupts:
                interrupt_payload = task.interrupts[0].value
                break

    if interrupt_payload:
        # ── Human decision ─────────────────────────────────────────────────
        print(f"\n⚠  Approval required")
        print(f"   Action : {interrupt_payload['action']}")
        print(f"   Args   : {interrupt_payload['args']}")
        decision = input("   Approve? (y/n): ").strip().lower()
        approved = decision == "y"

        # ── Resume the graph with the human's decision ─────────────────────
        from langgraph.types import Command
        resume_events = graph.stream(
            Command(resume=approved),
            config,
            stream_mode="values"
        )
        for event in resume_events:
            pass   # consume stream to completion

    # ── Final answer ───────────────────────────────────────────────────────
    final_state = graph.get_state(config)
    final_msg   = final_state.values["messages"][-1]
    if isinstance(final_msg, AIMessage) and final_msg.content:
        print(f"\nAgent (final): {final_msg.content}")


# ── 8. Demo ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Scenario 1 — user approves the booking
    run("Book me a flight from New York to London on 2025-06-15", thread_id="demo-1")

    # Scenario 2 — user checks weather (no approval needed for read-only tool)
    run("What's the weather like in Tokyo right now?", thread_id="demo-2")

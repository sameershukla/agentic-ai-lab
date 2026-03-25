"""
state.py — Defines the agent's shared memory (state) that flows through the graph.

In LangGraph, every node reads from and writes to this state object.
Think of it as the "whiteboard" all nodes share.
It's just a TypedDict
"""
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """
      The ONLY state this basic agent needs.
      `messages` holds the full conversation history.
      `add_messages` is a reducer — it APPENDS new messages
       instead of overwriting, so history is never lost.
    """
    messages: Annotated[list, add_messages]
from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """
    State shared across the graph.

    messages:
        Stores the conversation history.
        `add_messages` tells LangGraph to append new messages
        instead of replacing the whole list.
    """
    messages: Annotated[list, add_messages]

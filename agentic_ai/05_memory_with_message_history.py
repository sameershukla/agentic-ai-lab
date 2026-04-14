# 05_memory_with_message_history.py

import os
from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage, HumanMessage
from langchain_anthropic import ChatAnthropic
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# 1. Define state
class ChatState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

# 2. LLM Setup
llm = ChatAnthropic(model="claude-sonnet-4-20250514")

# 3. ChatNode
def chat_node(state: ChatState) -> ChatState:
    """
        Send the full message history to the model.
        Because the graph uses a checkpointer, messages from prior turns
        in the same thread will be available automatically.
        """
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

# 4. Build graph
def build_agent():
    graph = StateGraph(ChatState)

    # Node
    graph.add_node("chatbot", chat_node)

    # Edge
    graph.add_edge(START, "chatbot")
    graph.add_edge("chatbot", END)

    # MemorySaver stores conversation state in memory
    memory = MemorySaver()

    return graph.compile(checkpointer=memory)

if __name__ == "__main__":
    graph = build_agent()
    config = {
        "configurable": {
            "thread_id": "demo-thread-1"
        }
    }

    # Turn 1
    print("TURN 1")
    result_1 = graph.invoke(
        {
            "messages": [
                HumanMessage(content="My name is Sameer.")
            ]
        },
        config=config,
    )

    for msg in result_1["messages"]:
        print(f"\n[{msg.__class__.__name__}]")
        print(msg.content)

    print("\n" + "=" * 80)

    # Turn 2
    print("\nTURN 2")
    result_2 = graph.invoke(
        {
            "messages": [
                HumanMessage(content="What is my name?")
            ]
        },
        config=config,
    )

    for msg in result_2["messages"]:
        print(f"\n[{msg.__class__.__name__}]")
        print(msg.content)

    print("\n" + "=" * 80)

    # Turn 3
    print("\nTURN 3")
    result_3 = graph.invoke(
        {
            "messages": [
                HumanMessage(content="Please greet me personally.")
            ]
        },
        config=config,
    )

    for msg in result_3["messages"]:
        print(f"\n[{msg.__class__.__name__}]")
        print(msg.content)

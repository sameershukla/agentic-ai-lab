from typing import Annotated

from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

@tool
def add(a: float, b: float) -> float:
    "Invoke this function when + or string 'add' or 'sum' is encountered "
    return a + b

@tool
def multiply(a: float, b: float) -> float:
    "Multiply two numbers together "
    return a * b

@tool
def get_current_year() -> int:
    """Return the current year."""
    return 2026

TOOLS = [add, multiply, get_current_year]
llm = ChatAnthropic(model="claude-sonnet-4-20250514")
llm_with_tools = llm.bind_tools(TOOLS)

def llm_node(state: AgentState):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

tool_node = ToolNode(TOOLS)


def should_continue(state: AgentState):
    last_message = state["messages"][-1]

    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tool_node"

    return END


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("llm_node", llm_node)
    graph.add_node("tool_node", tool_node)

    graph.add_edge(START, "llm_node")
    graph.add_conditional_edges("llm_node", should_continue)
    graph.add_edge("tool_node", "llm_node")

    return graph.compile(checkpointer=memory)

if __name__ == "__main__":
    graph = build_graph()
    config = {
        "configurable": {
            "thread_id": "multi-tool-demo"
        }
    }

    queries = [
        "What is 25 plus 17?",
        "What is 8 multiplied by 12?",
        "What is the current year?",
        "If the current year is fetched, multiply it by 2."
    ]
    for query in queries:
        print("=" * 70)
        print("USER:", query)
        result = graph.invoke({
            "messages": [HumanMessage(query)]
        },
        config)
        print("AGENT:", result["messages"][-1].content)

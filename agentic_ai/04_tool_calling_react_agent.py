from typing import Annotated
from typing_extensions import TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

# State: Memory of the Agent
class AgentState(TypedDict):
    """
    Stores the full conversation history
    """
    messages: Annotated[list, add_messages]

# Tools: Actions the agent can take
@tool
def add(a: float, b: float) -> float:
    "Add two numbers together"
    return a + b

@tool
def multiply(a: float, b: float) -> float:
    "Multiply two numbers together"
    return a * b

TOOLS = [add, multiply]

# LLM Setup and bind tools with LLM
llm = ChatAnthropic(model="claude-sonnet-4-20250514")
llm_with_tools = llm.bind_tools(TOOLS)

# Nodes
def llm_node(state: AgentState):
    """
    The reasoning step
    The LLM:
       - reads conversation
       - decides to answer or call a tool
    """
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

# ToolNode(TOOLS) is required to actually execute the tool calls requested by the LLM and return the results back into the agent.
tool_node = ToolNode(TOOLS)


# Routing Logic
def should_continue(state: AgentState) -> str:
    """
        Decide whether to:
        - continue to tools
        - or finish execution
        """
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "end"

def build_agent():
    graph = StateGraph(AgentState)

    #Node
    graph.add_node("llm", llm_node)
    graph.add_node("tools", tool_node)

    # Flow
    graph.add_edge(START, "llm")
    graph.add_conditional_edges(
        "llm",
        should_continue, {
            "tools": "tools",
            "end": END,
       },
    )

    # After a tool runs, go back to the LLM again
    graph.add_edge("tools", "llm")
    return graph.compile()

agent = build_agent()

# Run Agent
def run_agent(query: str):
    print(f"\n{'=' * 50}")
    print(f"USER: {query}")
    print(f"{'=' * 50}")
    state = {"messages": [HumanMessage(content = query)]}

    # We are streaming here, instead of running the agent once (agent.invoke)
    for step, s in enumerate(agent.stream(state, stream_mode="values"), start=1):
        msg = s["messages"][-1]

        print(f"\n[Step {step}] {type(msg).__name__}")
        print(msg.content or "(tool call...)")

        """
        This block of code checks whether the latest message returned by the LLM contains a request to use any tools and, 
        if so, prints out the details of those tool calls. Since not every message has a tool_calls attribute, 
        hasattr ensures the code only accesses it when present, and msg.tool_calls confirms that the LLM actually requested one 
        or more tools. If both conditions are true, the loop iterates through each requested tool call and prints its name 
        along with the arguments the LLM wants to pass. In short, this block is used to detect and display 
        when the LLM decides to take an action (call a tool) as part of the ReAct reasoning process."""
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                print(f"→ Tool Call: {tc['name']}({tc['args']})")
    print(f"\n{'=' * 50}\n")

if __name__ == "__main__":
    run_agent("What is 12 multiplied by 7?")

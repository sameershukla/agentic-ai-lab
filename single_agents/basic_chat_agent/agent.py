from langchain_anthropic import ChatAnthropic
from langgraph.graph import START, StateGraph
from langchain_core.messages import AIMessage
from state import AgentState


llm = ChatAnthropic(model="claude-sonnet-4-20250514")


def chatbot_node(state: AgentState) -> AgentState:
    """
    Single node in the graph.

    It takes the full conversation history from state["messages"],
    sends it to the LLM, and appends the model's response back
    into the state.
    """
    response = llm.invoke(state["messages"])

    return {
        "messages": [AIMessage(content=response.content)]
    }


def build_graph():
    """
    Builds and compiles the LangGraph workflow.
    """
    graph_builder = StateGraph(AgentState)

    graph_builder.add_node("chatbot", chatbot_node)
    graph_builder.add_edge(START, "chatbot")

    return graph_builder.compile()
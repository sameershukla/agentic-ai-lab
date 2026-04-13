from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END

# 1. Define the State
class MultiNodeChatState(TypedDict):
    user_input: str
    message_type: Literal["greeting", "question"]
    response: str

# 2. Node 1: Classify the Input
def classifier_node(state: MultiNodeChatState) -> MultiNodeChatState:
    text = state['user_input'].lower().strip()
    greeting_words = ["hi", "hello", "hey"]
    if any(word in text for word in greeting_words):
        message_type = "greeting"
    else:
        message_type = "question"
    return {
        "user_input": state["user_input"],
        "message_type": message_type,
        "response": state["response"],  # still empty at this point
    }

# 3. Node 2: Generate the response
def response_node(state: MultiNodeChatState) -> MultiNodeChatState:
    user_text = state["user_input"]
    message_type = state["message_type"]

    if message_type == "greeting":
        response = "Hello! How can I assist you today?"
    else:
        response = f"That sounds like a question about: {user_text}"

    return {
        "user_input": user_text,
        "message_type": message_type,
        "response": response,
    }

# 4. Build the graph
graph_builder = StateGraph(MultiNodeChatState)

graph_builder.add_node("classifier", classifier_node)
graph_builder.add_node("response", response_node)

graph_builder.add_edge(START, "classifier")
graph_builder.add_edge("classifier", "response")
graph_builder.add_edge("response", END)

graph = graph_builder.compile()

# --------------------------------------------------
# 5. Run examples
# --------------------------------------------------
if __name__ == "__main__":
    print("Example 1: Greeting")
    result_1 = graph.invoke(
        {
            "user_input": "Hi there!",
            "message_type": "question",  # placeholder value
            "response": "",
        }
    )
    print(result_1)

    print("\nExample 2: Question")
    result_2 = graph.invoke(
        {
            "user_input": "What is LangGraph?",
            "message_type": "question",  # placeholder value
            "response": "",
        }
    )
    print(result_2)

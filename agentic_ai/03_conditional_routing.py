from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END


# 1. Define the shared state
class ConditionalChatState(TypedDict):
    user_input: str
    message_type: Literal["greeting", "question"]
    response: str


# 2. Node 1: Classifier]
def classifier_node(state: ConditionalChatState) -> ConditionalChatState:
    """
    Classify the user input as either a greeting or a question.
    """
    text = state["user_input"].lower().strip()
    greeting_words = ["hi", "hello", "hey"]

    if any(word in text for word in greeting_words):
        message_type = "greeting"
    else:
        message_type = "question"

    return {
        "user_input": state["user_input"],
        "message_type": message_type,
        "response": state["response"],
    }

# 3. Node 2A: Greeting response
def greeting_node(state: ConditionalChatState) -> ConditionalChatState:
    """
    Handle greeting messages.
    """
    return {
        "user_input": state["user_input"],
        "message_type": state["message_type"],
        "response": "Hello! Nice to meet you. How can I help you today?",
    }

# 4. Node 2B: Question response
def question_node(state: ConditionalChatState) -> ConditionalChatState:
    """
    Handle question-style messages.
    """
    user_text = state["user_input"]

    return {
        "user_input": user_text,
        "message_type": state["message_type"],
        "response": f"That looks like a question. Let me think about: {user_text}",
    }

# 5. Router function
def route_by_message_type(state: ConditionalChatState) -> str:
    """
    Decide which node to run next based on message_type.
    """
    if state["message_type"] == "greeting":
        return "greeting"
    return "question"

# 6. Build the graph
graph_builder = StateGraph(ConditionalChatState)

graph_builder.add_node("classifier", classifier_node)
graph_builder.add_node("greeting", greeting_node)
graph_builder.add_node("question", question_node)

graph_builder.add_edge(START, "classifier")
# Conditional routing happens here
graph_builder.add_conditional_edges(
    "classifier",
    route_by_message_type,
    {
        "greeting": "greeting",
        "question": "question",
    },
)

graph_builder.add_edge("greeting", END)
graph_builder.add_edge("question", END)

graph = graph_builder.compile()

# 7. Run examples
if __name__ == "__main__":
    print("Example 1: Greeting input")
    result_1 = graph.invoke(
        {
            "user_input": "Hello LangGraph!",
            "message_type": "question",  # placeholder
            "response": "",
        }
    )
    print(result_1)

    print("\nExample 2: Question input")
    result_2 = graph.invoke(
        {
            "user_input": "What is LangGraph?",
            "message_type": "question",  # placeholder
            "response": "",
        }
    )
    print(result_2)

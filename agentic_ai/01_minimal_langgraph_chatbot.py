from typing import TypedDict
from langgraph.graph import StateGraph, START, END

class ChatState(TypedDict):
    user_input: str
    response: str

def chatbot_node(state: ChatState) -> ChatState:
    user_text = state['user_input']
    return {
        'user_input': user_text,
        'response': f"You said {user_text}"
    }

#Build graph
graph_builder = StateGraph(ChatState)

# Add one node
graph_builder.add_node("chatbot", chatbot_node)

# Define flow
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

#Compile Graph
graph = graph_builder.compile()


#Run graph
result = graph.invoke({"user_input": "Hello LangGraph!", "response": ""})
print(result)

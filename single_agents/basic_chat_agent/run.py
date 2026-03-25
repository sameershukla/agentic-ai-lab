from langchain_core.messages import HumanMessage

from agent import build_graph


def main():
    graph = build_graph()

    print("\nBasic LangGraph Chat Agent")
    print("Type 'exit' to quit.\n")

    conversation = []

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        conversation.append(HumanMessage(content=user_input))

        result = graph.invoke({"messages": conversation})

        assistant_message = result["messages"][-1]
        print(f"AI: {assistant_message.content}\n")

        conversation = result["messages"]


if __name__ == "__main__":
    main()
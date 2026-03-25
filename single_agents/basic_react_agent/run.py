"""
run.py — Entry point. Run this to talk to the agent.

Usage:
    python run.py

Set your API key before running:
    export ANTHROPIC_API_KEY=your_key_here
"""

from langchain_core.messages import HumanMessage
from agent import agent


def run_agent(user_input: str, verbose: bool = True):
    """
    Send a message to the agent and print the result.
    Optionally prints every step in the graph for learning purposes.
    """
    initial_state = {"messages": [HumanMessage(content=user_input)]}

    if verbose:
        # stream_mode="values" emits the FULL state after every node runs.
        # Great for learning — you can see exactly what each node contributed.
        print(f"\n{'─'*60}")
        print(f"  USER: {user_input}")
        print(f"{'─'*60}")

        for step, state in enumerate(agent.stream(initial_state, stream_mode="values")):
            last_msg = state["messages"][-1]
            msg_type = type(last_msg).__name__

            print(f"\n[Step {step + 1}] {msg_type}")
            print(f"  {last_msg.content or '(tool call in progress...)'}")

            # Show tool call details if present
            if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                for tc in last_msg.tool_calls:
                    print(f"  → calling tool: {tc['name']}({tc['args']})")

        print(f"\n{'─'*60}\n")

    else:
        # Non-verbose: just get the final answer
        result = agent.invoke(initial_state)
        print(result["messages"][-1].content)


if __name__ == "__main__":
    # Try a few example queries
    queries = [
        "What is 47 multiplied by 13?",
        "What day and time is it right now?",
        "If I have 120 apples and give away 35, then multiply the remainder by 4, how many do I have?",
    ]

    for query in queries:
        run_agent(query, verbose=True)
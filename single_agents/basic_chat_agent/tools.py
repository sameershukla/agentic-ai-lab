import os
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool


@tool
def get_llm():
    """
    Creates and returns the LLM instance.

    Make sure you have:
        export ANTHROPIC_API_KEY="your-key"
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError(
            "ANTHROPIC_API_KEY is not set. Please set it in your environment."
        )

    return ChatAnthropic(
        model="claude-sonnet-4-20250514",
        temperature=0,
        api_key=api_key,
    )

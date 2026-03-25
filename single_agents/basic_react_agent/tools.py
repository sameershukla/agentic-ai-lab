"""
In Agentic AI terms a tool is just a Python function decorated with @tool annotation.
Langgraph will make these available to LLM automatically as callable Actions during the ReAct loop.
Every tool should have a docstring embed into it.
"""
from datetime import datetime
from langchain_core.tools import tool


@tool
def add(a: float, b: float) -> float:
    """Add two numbers together. Use this for any addition"""
    return a + b


@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers together. Use this for any multiplication."""
    return a * b


@tool
def get_current_datetime() -> str:
    """Return the current date and time. Use this when the user asks about time or date."""
    return datetime.now().strftime("%A, %B %d, %Y — %I:%M %p")


# Collect all tools in one list — this is what we'll bind to the LLM
tools = [add, multiply, get_current_datetime]

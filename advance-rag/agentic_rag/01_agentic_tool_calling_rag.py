"""
The description inside the function is important, if the description says only 
'Search the internal knowledge base for answers to technical questions' 
Technical questions" includes Python. The agent has no reason not to search and call 
the knowledge-base as the description is too broad.
"""
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage

# Simulate a knowledge base
KB = {
    "glue oom": "AWS Glue OOM errors are caused by insufficient DPU. Fix: increase DPU from 4 to 8.",
    "glue timeout": "Glue job timeout occurs when job exceeds 48 hours. Fix: enable job bookmarks.",
    "snowflake connection": "Snowflake connection failures are usually caused by expired credentials.",
}

@tool
def search_knowledge_base(query: str) -> str:
    """Search the internal knowledge base for AWS Glue, Snowflake,
        Airflow, and internal pipeline issues ONLY.
        Do NOT use for general programming concepts, definitions,
        or anything answerable from general knowledge"""
    for key, value in KB.items():
        if key in query.lower():
            return value
    return "No relevant knowledge base found"

llm = ChatAnthropic(model="claude-sonnet-4-20250514")
llm_with_tools = llm.bind_tools([search_knowledge_base])

# Test it
queries = [
    "What is Python?",                          # should NOT retrieve
    "Why is my Glue job running out of memory?", # should retrieve
]

for query in queries:
    print(f"\nQ:{query}")
    response = llm_with_tools.invoke([HumanMessage(content=query)])
    if response.tool_calls:
        print(f"→ Agent decided to search: {response.tool_calls[0]['args']}")
    else:
        print(f"→ Agent answered directly: {response.content[:100]}")

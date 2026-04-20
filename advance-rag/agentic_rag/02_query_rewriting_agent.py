from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

llm = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0)

def rewrite_query(origian_question: str) -> str:
    """Rewrite a user question into the best possible search query."""
    message = SystemMessage(content=""""You are a search query specialist.
                Rewrite the user's question as a short, precise search query.
                Remove conversational filler. Resolve pronouns. Use technical terminology.
                Return ONLY the rewritten query, nothing else. Max 10 words.""")
    response = llm.invoke([message, HumanMessage(content=origian_question)])
    return response.content.strip()

vague_queries = [
    "it keeps crashing when processing big files",
    "how do I fix the connection thing",
    "the job runs forever and never finishes",
]

for q in vague_queries:
    rewritten = rewrite_query(q)
    print(f"\Original: {q}")
    print(f"\Rewritten: {rewritten}")

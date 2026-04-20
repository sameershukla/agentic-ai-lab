from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool
from pydantic import BaseModel
from typing import Literal

KB = {
    "glue oom outofmemory memory":
        "AWS Glue OOM (OutOfMemoryError) is caused by insufficient executor memory. "
        "Fix: (1) Increase DPU from 4 to 8 or higher. "
        "(2) Repartition your data to reduce partition size. "
        "(3) Avoid collect() on large DataFrames — use write() instead. "
        "(4) Enable Glue job bookmarks to process only new data.",

    "glue timeout":
        "Glue job timeout occurs when a job exceeds its configured timeout limit (default 48 hours). "
        "Fix: (1) Enable job bookmarks to process incrementally. "
        "(2) Increase the timeout limit in job configuration. "
        "(3) Break large jobs into smaller batches.",

    "snowflake connection credential":
        "Snowflake connection failures are typically caused by: "
        "(1) Expired credentials or rotated passwords. "
        "(2) Network/VPC configuration blocking outbound connections. "
        "(3) IP whitelist not including your execution environment. "
        "Fix: Rotate credentials, verify network config, and check Snowflake audit logs.",

    "glue job bookmark":
        "AWS Glue job bookmarks track which data has already been processed. "
        "Enable bookmarks to avoid reprocessing the same S3 files on each run. "
        "Set Job Bookmark to 'Enable' in the job parameters. "
        "Bookmarks work with S3, JDBC, and DynamoDB sources.",

    "airflow dag failure":
        "Airflow DAG failures are commonly caused by: "
        "(1) Upstream task failure propagating downstream. "
        "(2) Worker resource exhaustion. "
        "(3) Python import errors in task functions. "
        "Fix: Check task logs, increase worker memory, and validate DAG syntax with 'airflow dags test'.",
}

def search_knowledge_base_impl(query: str) -> str:
    """Search the KB by keyword matching. In production replace with vector search."""
    query_lower = query.lower()
    for keywords, answer in KB.items():
        if any(kw in query_lower for kw in keywords.split()):
            return answer
    return "NO_RESULT"


@tool
def search_knowledge_base(query: str) -> str:
    """
    Search the internal knowledge base for AWS Glue, Snowflake,
    Airflow, and internal pipeline issues ONLY.

    Use for: Glue job failures, Snowflake connection errors,
             Airflow DAG issues, internal runbooks, pipeline errors.

    Do NOT use for: general programming concepts, public technology
                    definitions, or anything answerable from training knowledge.

    Examples: 'Glue OOM error fix', 'Snowflake credential failure',
              'Airflow DAG not triggering'
    """
    return search_knowledge_base_impl(query)

# LLM Instances
agent_llm = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0)

# Rewriter LLM — rewrites user query into optimal search query
rewriter_llm = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0)


# Grader LLM — structured output to grade relevance
class RelevanceGrade(BaseModel):
    grade: Literal["relevant", "not_relevant"]
    reasoning: str


grader_llm = ChatAnthropic(
    model="claude-sonnet-4-20250514",
    temperature=0
).with_structured_output(RelevanceGrade)

# Bind tool to agent LLM
agent_llm_with_tools = agent_llm.bind_tools([search_knowledge_base])

# System prompt — guides routing behavior
AGENT_SYSTEM = SystemMessage(content="""You are a data engineering assistant \
specialising in AWS Glue, Snowflake, and Apache Airflow.

You have access to an internal knowledge base tool.

SEARCH the knowledge base when the question is about:
- AWS Glue job failures, errors, or configuration
- Snowflake connection or query issues
- Airflow DAG failures or scheduling problems
- Internal pipeline runbooks or procedures

ANSWER DIRECTLY from your training knowledge when:
- The question is about general programming concepts (Python, SQL basics)
- The question is about publicly documented technology
- You are confident you already know the answer

When in doubt about an internal system — always search first.""")

# Route
def route(question: str) -> dict:
    """
    Ask the LLM whether this question needs retrieval.
    Returns {'needs_retrieval': bool, 'direct_answer': str | None}
    """
    response = agent_llm_with_tools.invoke([AGENT_SYSTEM, HumanMessage(content=question)])

    if response.tool_calls:
        return {"needs_retrieval": True, "direct_answer": None}
    else:
        return {"needs_retrieval": False, "direct_answer": response.content}

# Rewries
def rewrite_query(question: str, attempt: int = 0) -> str:
    """
    Rewrite the user's question into the best possible search query.
    On retry attempts, prompts for a different angle.
    """
    retry_note = ""
    if attempt > 0:
        retry_note = (
            "\nIMPORTANT: The previous search query returned no useful results. "
            "Try a completely different angle, different terminology, or broader terms."
        )

    response = rewriter_llm.invoke([
        SystemMessage(content=f"""You are a search query specialist for a \
data engineering knowledge base.

Rewrite the user's question as the best possible search query.
Rules:
- Remove conversational filler ('how do I', 'can you help me', 'I want to know')
- Resolve all pronouns to their actual referents
- Use technical terminology the knowledge base likely uses
- Keep it under 10 words
- Return ONLY the rewritten query, nothing else{retry_note}"""),
        HumanMessage(content=question)
    ])
    return response.content.strip()

#Retries
def retrieve(query: str) -> str:
    """Run the search against the knowledge base."""
    return search_knowledge_base_impl(query)

# Grade Relevance
def grade_relevance(question: str, document: str) -> RelevanceGrade:
    """
    Grade whether the retrieved document actually helps answer the question.
    Strict grading — partial relevance is not enough.
    """
    if document == "NO_RESULT":
        return RelevanceGrade(
            grade="not_relevant",
            reasoning="No document was found in the knowledge base."
        )

    return grader_llm.invoke([
        SystemMessage(content="""You are a strict relevance grader.
Grade whether the retrieved document directly helps answer the question.

relevant     = document contains information that directly addresses the question
not_relevant = document does not contain useful information for this question

Be strict. A tangentially related document is not_relevant."""),
        HumanMessage(content=f"Question: {question}\n\nDocument: {document}")
    ])

# final answer
def generate_answer(question: str, context: str) -> str:
    """Generate a grounded answer using only the retrieved context."""
    response = agent_llm.invoke([
        SystemMessage(content="""You are a data engineering assistant.
Answer using ONLY the provided context.
Be specific — include exact values (DPU counts, timeouts, commands) from the context.
If the answer is not in the context, say: 'I could not find that in our knowledge base.'
Do not use outside knowledge."""),
        HumanMessage(content=f"Context:\n{context}\n\nQuestion: {question}")
    ])
    return response.content

# Agentic RAG loop
MAX_RETRIES = 2


def agentic_rag(question: str) -> str:
    print(f"\n{'=' * 60}")
    print(f"QUESTION: {question}")
    print("-" * 60)

    # Decision 1: Route
    routing = route(question)

    if not routing["needs_retrieval"]:
        print("ROUTE   : answered directly (no retrieval needed)")
        print(f"ANSWER  : {routing['direct_answer']}")
        return routing["direct_answer"]

    print("ROUTE   : retrieval needed → searching knowledge base")

    # Decision 2 + 3 + 4: Rewrite → Retrieve → Grade loop ──
    context = None

    for attempt in range(MAX_RETRIES):
        # Rewrite
        search_query = rewrite_query(question, attempt=attempt)
        print(f"REWRITE : '{search_query}' (attempt {attempt + 1})")

        # Retrieve
        document = retrieve(search_query)
        preview = document[:80] + "..." if len(document) > 80 else document
        print(f"RETRIEVE: {preview}")

        # Grade
        grade = grade_relevance(question, document)
        print(f"GRADE   : {grade.grade} — {grade.reasoning}")

        if grade.grade == "relevant":
            context = document
            break

        if attempt < MAX_RETRIES - 1:
            print(f"         → retrying with different query...")

    #  Generate or admit failure 
    if context is None:
        answer = (
            "I searched the knowledge base but could not find relevant information. "
            "Please check with your team or escalate to the relevant channel."
        )
        print(f"ANSWER  : {answer}")
        return answer

    answer = generate_answer(question, context)
    print(f"ANSWER  : {answer}")
    return answer


if __name__ == "__main__":

    test_questions = [
        # Should answer directly — general knowledge
        "What is Python?",

        # Should retrieve and answer — Glue OOM
        "Why is my Glue job running out of memory?",

        # Should retrieve — vague phrasing, needs rewrite
        "The job keeps crashing when processing large files",

        # Should retrieve — Snowflake
        "Our Snowflake connection keeps failing in production",

        # Should retrieve — Airflow
        "My Airflow DAG failed this morning, how do I debug it?",

        # Should retrieve — Glue bookmarks
        "How do I stop Glue from reprocessing the same files every run?",

        # Should fail gracefully — nothing in KB
        "Why is my Kafka consumer lagging?",
    ]

    for question in test_questions:
        agentic_rag(question)

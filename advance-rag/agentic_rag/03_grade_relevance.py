from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel
from typing import Literal

class RelevanceGrade(BaseModel):
    grade: Literal["relevant", "not_relevant"]
    reasoning: str

grader_llm = ChatAnthropic(model="claude-sonnet-4-20250514",
                           temperature=0).with_structured_output(RelevanceGrade)

def grade_relevance(question: str, retrieved_doc: str) -> RelevanceGrade:
    return grader_llm.invoke([
        SystemMessage(content="""Grade whether the retrieved document
    helps answer the question. Be strict.
    relevant = document directly addresses the question.
    not_relevant = document does not help."""),
        HumanMessage(content=f"Question: {question}\n\nDocument: {retrieved_doc}")
    ])

#Test it
question = "How do I fix a Glue OOM error?"
docs = [
    "AWS Glue OOM errors are caused by insufficient DPU. Fix: increase DPU from 4 to 8.",  # relevant
    "Snowflake connection failures are usually caused by expired credentials.",               # not relevant
]

for doc in docs:
    grade = grade_relevance(question, doc)
    print(f"Grade:{grade.grade}")
    print(f"Reason:{grade.reasoning}")

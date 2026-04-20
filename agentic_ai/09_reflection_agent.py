from langchain_anthropic import ChatAnthropic

llm = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0)

def generate_draft(question: str) -> str:
    prompt = f"""
     You are a helpful technical assistant.
     Write a short answer to this question:
     Question: {question} """.strip()
    response = llm.invoke(prompt)
    return response.content.strip()

def critique_draft(question: str, draft: str) -> str:
    prompt = f"""
    You are a strict reviewer.
    Review the draft answer below.
    Question: {question}
    Draft answer: {draft}

    Your job:
    - identify what is weak, vague, or missing
    - suggest how to improve it
    - be concise
    """.strip()
    response = llm.invoke(prompt)
    return response.content.strip()

def revise_draft(question: str, draft: str, critique: str) -> str:
    prompt = f"""
    You are improving an answer using reviewer feedback.

    Question:
    {question}
    
    Original draft:
    {draft}
    
    Reviewer feedback:
    {critique}
    
    Write a better final answer.
    Make it:
    - clearer
    - more specific
    - more useful
    """.strip()
    response = llm.invoke(prompt)
    return response.content.strip()


if __name__ == "__main__":
    question = "Why did my AWS Glue job fail?"

    draft = generate_draft(question)
    critique = critique_draft(question, draft)
    final_answer = revise_draft(question, draft, critique)

    print("=" * 60)
    print("USER QUESTION:")
    print(question)

    print("\nDRAFT ANSWER:")
    print(draft)

    print("\nCRITIQUE:")
    print(critique)

    print("\nFINAL IMPROVED ANSWER:")
    print(final_answer)

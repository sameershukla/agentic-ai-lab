from datasets import Dataset
from langchain_anthropic import ChatAnthropic
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

# Step1: Create KnowledgeBase
KnowledgeBase = [
    Document(
        page_content=(
            "RAG stands for Retrieval-Augmented Generation. "
            "It retrieves relevant documents and passes them to an LLM to generate grounded answers."
        )
    ),
    Document(
            page_content=(
                "Chunking splits large documents into smaller pieces so embeddings and retrieval work better."
            )
        ),
        Document(
            page_content=(
                "Embeddings convert text into vectors. Similar meanings usually have nearby vectors."
            )
        ),
        Document(
            page_content=(
                "RAGAS evaluates RAG pipelines using metrics such as faithfulness, "
                "answer relevancy, context precision, and context recall."
            )
        ),
]

# Step2: Create embeddings and vector store
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = FAISS.from_documents(KnowledgeBase, embedding_model)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

#Step3: LLM
llm = ChatAnthropic(model="claude-sonnet-4-20250514")

def rag_answer(question: str):
    retrieved_docs = retriever.invoke(question)
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    prompt = f"""
    You are a helpful RAG assistant.
    Answer the question using only the context below.
    If the answer is not present in the context, say "I don't know."
    Context: {context}
    Question: {question}
    """
    response = llm.invoke(prompt)
    return {
        "question": question,
        "answer": response.content,
        "contexts": [doc.page_content for doc in retrieved_docs],
    }

#Run RAG Pipeline
questions = [
    "What is RAG?",
    "Why is chunking important?",
    "What does RAGAS evaluate?",
]
rag_outputs = []

for question in questions:
    output = rag_answer(question)
    rag_outputs.append(output)
    print("\nQuestion:", output["question"])
    print("Answer:", output["answer"])
    print("Contexts:", output["contexts"])

# Add ground truths
ground_truths = [
    "RAG retrieves relevant documents and gives them to an LLM so it can generate grounded answers.",
    "Chunking breaks large documents into smaller pieces, improving embeddings and retrieval.",
    "RAGAS evaluates RAG pipelines using faithfulness, answer relevancy, context precision, and context recall.",
]

eval_data = {
    "question": [item["question"] for item in rag_outputs],
    "answer": [item["answer"] for item in rag_outputs],
    "contexts": [item["contexts"] for item in rag_outputs],
    "ground_truth": ground_truths,
}

dataset = Dataset.from_dict(eval_data)

# Step 6: Evaluate using RAGAS
# LangchainLLMWrapper and LangchainEmbeddingsWrapper are required —
# without them RAGAS ignores your LLM and silently falls back to OpenAI
judge_llm = LangchainLLMWrapper(ChatAnthropic(model="claude-sonnet-4-6", temperature=0))
judge_embeddings = LangchainEmbeddingsWrapper(embedding_model)

result = evaluate(
    dataset=dataset,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    llm=judge_llm,
    embeddings=judge_embeddings,
)

print("\nRAGAS Evaluation Result:")
print(result)

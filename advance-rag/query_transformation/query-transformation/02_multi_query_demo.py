from langchain_anthropic import ChatAnthropic
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


def load_documents():
    return [
        Document(
            page_content="""
Glue Job Troubleshooting Guide

If a Glue job fails with AmazonS3Exception Access Denied, check the following:
1. The IAM role has s3:GetObject permission
2. The IAM role has kms:Decrypt permission if the files are encrypted
3. The S3 bucket policy allows the Glue execution role
4. The target S3 path is correct
""".strip(),
            metadata={"source": "runbook_glue_access_denied"},
        ),
        Document(
            page_content="""
Glue Job Memory Troubleshooting

If a Glue job fails with OutOfMemoryError, common causes include:
1. Input partitions are too large
2. Worker type is too small
3. Skewed joins create large executor pressure
4. Caching large DataFrames increases memory usage
""".strip(),
            metadata={"source": "runbook_glue_memory"},
        ),
        Document(
            page_content="""
IAM Policy Guidance

Approved Glue execution roles must include:
- s3:GetObject
- s3:ListBucket
- logs:CreateLogGroup
- logs:CreateLogStream
- logs:PutLogEvents
- kms:Decrypt
""".strip(),
            metadata={"source": "iam_policy_glue"},
        ),
        Document(
            page_content="""
Large Operations Guide

For Glue failures involving S3 access issues, investigate IAM permissions,
KMS decrypt access, S3 bucket policy alignment, recent deployment changes,
and incorrect bucket paths.
""".strip(),
            metadata={"source": "operations_guide"},
        ),
        Document(
            page_content="""
CloudWatch Error Notes

A recent Glue job failed because the execution role could not access
an encrypted object in S3. The job role was missing kms:Decrypt permission.
""".strip(),
            metadata={"source": "cloudwatch_notes"},
        ),
    ]


def build_vector_store(documents):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return FAISS.from_documents(documents, embeddings)


def generate_query_variants(llm, question: str) -> list[str]:
    prompt = f"""
You are helping a retrieval system.

Generate 3 alternative search queries for the question below.
Keep the meaning the same.
Make each variation short and useful for document retrieval.

Return only the queries, one per line, with no numbering.

Question:
{question}
""".strip()

    response = llm.invoke(prompt)
    lines = [line.strip() for line in response.content.splitlines() if line.strip()]

    # Keep original question also
    queries = [question] + lines

    # Deduplicate while preserving order
    unique_queries = []
    seen = set()
    for q in queries:
        q_norm = q.lower().strip()
        if q_norm not in seen:
            seen.add(q_norm)
            unique_queries.append(q)

    return unique_queries


def retrieve_for_each_query(vector_store, queries: list[str], k: int = 2):
    all_results = []

    for query in queries:
        docs = vector_store.similarity_search(query, k=k)
        for doc in docs:
            all_results.append((query, doc))

    return all_results


def deduplicate_documents(query_doc_pairs):
    unique_docs = []
    seen = set()

    for query, doc in query_doc_pairs:
        doc_id = (
            doc.metadata.get("source", "") + "||" + doc.page_content.strip()
        )
        if doc_id not in seen:
            seen.add(doc_id)
            unique_docs.append((query, doc))

    return unique_docs


def print_query_variants(queries):
    print("\n" + "=" * 100)
    print("QUERY VARIANTS")
    print("=" * 100)
    for i, q in enumerate(queries, start=1):
        print(f"{i}. {q}")


def print_raw_results(results):
    print("\n" + "=" * 100)
    print("RAW RETRIEVAL RESULTS")
    print("=" * 100)
    for i, (query, doc) in enumerate(results, start=1):
        print(f"\nResult {i}")
        print(f"Query Used: {query}")
        print(f"Source: {doc.metadata.get('source', 'unknown')}")
        print(doc.page_content)


def print_final_results(results):
    print("\n" + "=" * 100)
    print("FINAL DEDUPLICATED RESULTS")
    print("=" * 100)
    for i, (query, doc) in enumerate(results, start=1):
        print(f"\nDocument {i}")
        print(f"Matched From Query: {query}")
        print(f"Source: {doc.metadata.get('source', 'unknown')}")
        print(doc.page_content)


if __name__ == "__main__":
    documents = load_documents()
    vector_store = build_vector_store(documents)

    llm = ChatAnthropic(
        model="claude-sonnet-4-6",
        temperature=0
    )

    question = "Why did my Glue job fail with an access issue this morning?"

    queries = generate_query_variants(llm, question)
    print_query_variants(queries)

    raw_results = retrieve_for_each_query(vector_store, queries, k=2)
    print_raw_results(raw_results)

    final_results = deduplicate_documents(raw_results)
    print_final_results(final_results)

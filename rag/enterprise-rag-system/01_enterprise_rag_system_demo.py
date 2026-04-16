from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_anthropic import ChatAnthropic

"""
These are private and recent documents that the model was never trained on.
In a real system, these would come from CloudWatch, Snowflake, Confluence,
internal runbooks, policy repositories, or a document database.
"""
def load_private_documents():
    docs = [
        Document(
            page_content="""
                Glue Job Failure Log
                Timestamp: 2026-04-16 02:00:12 UTC
                Job Name: customer_daily_etl
                Error: AmazonS3Exception Access Denied
                Root cause: The IAM role used by the Glue job lost permission to read
                from s3://company-raw-zone/customer/.
                Suggested action: Verify s3:GetObject permission and KMS decrypt access.
                """,
            metadata={"source": "cloudwatch_log", "topic": "glue_failure"}
        ),
        Document(
            page_content="""
                Team Runbook for Glue Failures

                If a Glue job fails with AmazonS3Exception Access Denied, first verify:
                1. The IAM role has s3:GetObject on the target bucket
                2. The IAM role has kms:Decrypt if the files are encrypted
                3. The bucket policy allows the Glue execution role
                4. The data path is correct and has not changed

                Common symptom:
                AccessDenied errors after deployment usually indicate either a missing
                IAM permission or a changed KMS key policy.
                """,
            metadata={"source": "runbook", "topic": "glue_failure"}
        ),
        Document(
            page_content="""
                Snowflake Data Catalog
                Table Name: customer_master

                Columns:
                customer_id
                first_name
                last_name
                date_of_birth
                email
                phone_number
                customer_status
                created_at
                updated_at
                """,
            metadata={"source": "snowflake_catalog", "topic": "schema"}
        ),
        Document(
            page_content="""
                Approved IAM Policy for Glue Service Role

                Allowed actions:
                s3:GetObject
                s3:ListBucket
                logs:CreateLogGroup
                logs:CreateLogStream
                logs:PutLogEvents
                kms:Decrypt

                Restricted actions:
                s3:DeleteObject
                iam:PassRole except approved execution roles

                This policy is approved for the Glue service role used in ETL jobs.
                """,
            metadata={"source": "iam_policy", "topic": "policy"}
        ),
        Document(
            page_content="""
                Large Operations Runbook

                Section A: Network troubleshooting steps...
                Section B: Spark executor memory tuning...
                Section C: IAM and S3 access troubleshooting...
                Section D: Glue bookmark issues...
                Section E: Snowflake load troubleshooting...
                Section F: Partition repair process...
                Section G: KMS key validation process...

                For Access Denied issues in Glue, focus on IAM permissions, KMS decrypt
                access, bucket policy validation, and bucket path correctness.
                """,
            metadata={"source": "large_runbook", "topic": "operations"}
        ),
    ]
    return docs


"""
RecursiveCharacterTextSplitter helps preserve meaning while chunking large text.
This solves the context window problem by sending only relevant chunks later.
"""
def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50
    )
    return splitter.split_documents(documents)


"""
Searchable Vector index over private documents
"""
def build_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store

"""
    Retrieve only the top relevant chunks.
    This is the key RAG step that fixes context window issues.
    """
def retrieve_context(vector_store, question, k=3):
    results = vector_store.similarity_search(question, k=k)
    return results


def format_context(retrieved_docs):
    blocks = []
    for i, doc in enumerate(retrieved_docs, start=1):
        block = (
            f"[Document {i}]\n"
            f"Source: {doc.metadata.get('source', 'unknown')}\n"
            f"Topic: {doc.metadata.get('topic', 'unknown')}\n"
            f"Content:\n{doc.page_content}"
        )
        blocks.append(block)
    return "\n\n".join(blocks)

def build_prompt(question, context):
    return f"""
You are an enterprise assistant helping with internal engineering questions.

Rules:
1. Answer only from the provided context.
2. Do not invent facts.
3. If the answer is not present in the context, say:
   "I do not know based on the provided documents."
4. When possible, mention the source names used.

Context:
{context}

Question:
{question}
""".strip()


def ask_question(vector_store, llm, question):
    retrieved_docs = retrieve_context(vector_store, question, k=3)
    context = format_context(retrieved_docs)
    
    prompt = build_prompt(question, context)

    response = llm.invoke(prompt)

    print("\n" + "=" * 100)
    print(f"QUESTION: {question}")
    print("=" * 100)

    print("\nRETRIEVED DOCUMENTS:\n")
    for i, doc in enumerate(retrieved_docs, start=1):
        print(f"[{i}] source={doc.metadata.get('source')} topic={doc.metadata.get('topic')}")
        print(doc.page_content)
        print("-" * 100)

    print("\nFINAL ANSWER:\n")
    print(response.content)
    print("=" * 100)

if __name__ == "__main__":
    #Step 1: Load documents
    documents = load_private_documents()

    #Step 2: Create meaningful chunks
    chunks = split_documents(documents)

    #Step 3: Store chunks in vector store
    vector_store = build_vector_store(chunks)

    #Step 4: Create LLM
    llm = ChatAnthropic(model="claude-sonnet-4-20250514")

    questions = [
        "Why did the Glue job fail at 2 AM today?",
        "What columns are in our customer_master table in Snowflake?",
        "What is our approved IAM policy for our Glue service role?",
        "What should I check for an Access Denied error in Glue?",
        "What is the retention policy for our production Kafka cluster?"
    ]

    for question in questions:
        ask_question(vector_store, llm, question)

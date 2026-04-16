from langchain_text_splitters import RecursiveCharacterTextSplitter

runbook = """
Executor OOM: Exit Status 137
When an executor runs out of heap memory, AWS Glue will terminate the task and 
log exit status 137. This is not a Glue bug — it means the data volume exceeded 
the memory allocated to that executor.

To prevent this, increase the executor-memory parameter to 8G in the job 
configuration. The default of 4G is sufficient for small datasets but will 
fail under heavy joins or wide aggregations.

Executors running large joins may also benefit from repartitioning the dataset 
before the shuffle step. Use df.repartition(200) before any groupBy or join 
on a large table.

If the job continues to fail after increasing memory, enable autoscaling DPUs 
so Glue can acquire additional workers dynamically rather than failing outright.
"""

splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,       # maximum characters per chunk
    chunk_overlap=60,     # overlap between consecutive chunks to avoid boundary gaps
    separators=["\n\n", "\n", ". ", " ", ""]   # hierarchy: paragraph → line → sentence → word → character
)

chunks = splitter.split_text(runbook)

for i, chunk in enumerate(chunks, start=1):
    print(f"------------Chunk {i} ----------------")
    print(chunk.strip())
    print()

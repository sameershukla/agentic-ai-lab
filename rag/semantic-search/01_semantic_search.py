from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Step 1: A small knowledge base — each string is one "document".
documents = [
    "Transformers use self-attention to model token relationships.",
    "FAISS performs fast nearest-neighbour search over dense vectors.",
    "Retrieval-Augmented Generation combines a retriever with an LLM.",
    "Embeddings map text into a continuous high-dimensional space.",
    "ChromaDB is an open-source vector database with metadata filtering.",
    "Cosine similarity measures the angle between two embedding vectors.",
    "LangChain provides abstractions for building LLM-powered pipelines.",
    "Fine-tuning adapts a pre-trained model to a downstream task.",
]

# Step 2: Embedding
model = SentenceTransformer("all-MiniLM-L6-v2")
doc_embeddings = model.encode(documents, convert_to_numpy=True)   # shape: (8, 384)
doc_embeddings = doc_embeddings.astype("float32")                  # FAISS requires float32

# Step 3: Build a FAISS Index
dimension = doc_embeddings.shape[1]  # 384
index = faiss.IndexFlatL2(dimension)

# FAISS keeps only vectors; we maintain a parallel list for the actual text.
index.add(doc_embeddings)
print(f"Index built — {index.ntotal} vectors stored.\n")

# Step 4: Query
query = "How does attention work in neural networks?"
query_embedding = model.encode([query], convert_to_numpy=True).astype("float32")

# Search: k = number of nearest neighbours to retrieve.
k = 3
distances, indices = index.search(query_embedding, k)

# Step 5: Return Results
print(f"Query : {query}\n")
print("Top results (lower L2 distance = more similar):")
print("-" * 60)
for rank, (idx, dist) in enumerate(zip(indices[0], distances[0]), start=1):
    print(f"  {rank}. [{dist:.4f}]  {documents[idx]}")

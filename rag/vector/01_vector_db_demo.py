import chromadb
from sentence_transformers import SentenceTransformer

products = [
    {"id": "p1", "text": "Budget wireless mouse for daily office work", "price": 18, "rating": 4.1},
    {"id": "p2", "text": "Premium gaming mouse with customizable DPI", "price": 75, "rating": 4.7},
    {"id": "p3", "text": "Mechanical keyboard with RGB lighting", "price": 55, "rating": 4.6},
    {"id": "p4", "text": "USB office keyboard for basic typing", "price": 44, "rating": 4.0},
    {"id": "p5", "text": "Noise cancelling wireless headphones", "price": 120, "rating": 4.8},
]

model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.Client()
collection = client.create_collection(name="products_demo")

collection.add(
    ids=[p["id"] for p in products],
    documents=[p["text"] for p in products],
    metadatas=[{"price": p["price"], "rating": p["rating"]} for p in products],
    embeddings=model.encode([p["text"] for p in products]).tolist()
)

query = "wireless mouse under 30 dollars"
query_embedding = model.encode([query]).tolist()

results = collection.query(
    query_embeddings=query_embedding,
    n_results=5,
    where={"price": {"$lt": 30}}
)

print("----- Results -----")
print(results["documents"][0])

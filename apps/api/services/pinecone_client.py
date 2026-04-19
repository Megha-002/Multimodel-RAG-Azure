from pinecone import Pinecone
from config import PINECONE_API_KEY, PINECONE_INDEX, SIMILARITY_THRESHOLD, TOP_K

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)

def query_pinecone(vector: list, top_k: int = TOP_K) -> list:

    results = index.query(
        vector=vector,
        top_k=top_k,
        include_metadata=True
    )

    matches = [
        {
            "text": m["metadata"].get("text", ""),
            "source": m["metadata"].get("source", ""),
            "score": m["score"]
        }
        for m in results["matches"]
        if m["score"] >= SIMILARITY_THRESHOLD
    ]

    return matches
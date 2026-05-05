import os
import time
from dotenv import load_dotenv, find_dotenv
from pinecone import Pinecone
from loaders import load_all_documents
from chunking import chunk_documents
from embed_client import get_embedding

load_dotenv(find_dotenv())

# ── Pinecone Setup ─────────────────────────────────────────
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX", "rag-index"))


def ingest(data_folder: str):
    """
    Full ingestion pipeline:
    1. Load documents from folder
    2. Chunk them into smaller pieces
    3. Embed each chunk via HF API
    4. Upsert vectors into Pinecone
    """
    print("\n🚀 Starting ingestion...\n")

    # Step 1 — Load documents
    print("📂 Loading documents...")
    documents = load_all_documents(data_folder)
    print(f"   Found {len(documents)} documents.\n")

    if not documents:
        print("❌ No documents found. Check your data folder.")
        return

    # Step 2 — Chunk documents
    print("✂️  Chunking documents...")
    chunk_size = int(os.getenv("CHUNK_SIZE", 500))
    chunk_overlap = int(os.getenv("CHUNK_OVERLAP", 50))
    chunks = chunk_documents(documents, chunk_size, chunk_overlap)
    print()

    # Step 3 — Embed and upsert
    print("🔄 Embedding and uploading to Pinecone...")
    vectors = []
    total = len(chunks)

    for i, chunk in enumerate(chunks):
        # Get embedding from HF API
        embedding = get_embedding(chunk["text"])

        vectors.append({
            "id": f"chunk-{i}",
            "values": embedding,
            "metadata": {
                "text": chunk["text"],
                "source": chunk["source"]
            }
        })

        # Batch upsert every 50 vectors
        if len(vectors) >= 50:
            index.upsert(vectors=vectors)
            vectors = []
            print(f"   ✅ Uploaded {i + 1}/{total} chunks...")

        # Small delay to avoid HF rate limits
        time.sleep(0.1)

    # Upsert remaining vectors
    if vectors:
        index.upsert(vectors=vectors)

    print(f"\n🎉 Ingestion complete! {total} chunks uploaded to Pinecone.\n")


if __name__ == "__main__":
    # Default: look for data folder in project root
    data_path = os.path.join(os.path.dirname(__file__), "..", "data")
    ingest(data_path)
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
import ollama
import time
import warnings
warnings.filterwarnings("ignore")

app = FastAPI(title="Enterprise RAG Chatbot API")

# Load ChromaDB once when app starts
print("Loading ChromaDB...")
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings
)
print("✅ ChromaDB loaded!")

# ===============================
# HEALTH CHECK
# ===============================

@app.get("/health")
def health():
    return {"status": "running"}

# ===============================
# TEXT QUERY ENDPOINT
# ===============================

class QueryRequest(BaseModel):
    question: str

@app.post("/query")
def query(request: QueryRequest):
    start_time = time.time()

    # Retrieve relevant chunks from ChromaDB
    retriever = vector_store.as_retriever()
    relevant_docs = retriever.invoke(request.question)
    context = "\n\n".join([doc.page_content for doc in relevant_docs])

    # Build prompt
    prompt = f"""
You are an enterprise employee assistant. Answer using ONLY the context from company documents.

Context:
{context}

Question:
{request.question}

If the answer is not in the context, say "I don't have that information in the company documents."
"""

    # Get LLaMA 3 response
    response = ollama.chat(
        model="llama3",
        messages=[
            {"role": "system", "content": "You are a helpful enterprise employee assistant."},
            {"role": "user", "content": prompt}
        ]
    )

    latency = round(time.time() - start_time, 2)

    return {
        "question": request.question,
        "answer": response['message']['content'],
        "latency_seconds": latency,
        "sources": [doc.metadata.get("source", "unknown") for doc in relevant_docs]
    }
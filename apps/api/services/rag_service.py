from services.embed_client import get_embedding
from services.pinecone_client import query_pinecone
from services.groq_client import generate_answer
from config import TOP_K


def run_rag(question: str) -> dict:
    # Step 1 — Convert question to vector
    query_vector = get_embedding(question)

    # Step 2 — Find relevant chunks from Pinecone
    chunks = query_pinecone(query_vector, top_k=TOP_K)

    # Step 3 — If nothing relevant found
    if not chunks:
        return {
            "answer": "I could not find any relevant information to answer your question.",
            "sources": []
        }

    # Step 4 — Build context from retrieved chunks
    context = "\n\n".join([c["text"] for c in chunks])

    # Step 5 — Generate answer using Groq LLM
    answer = generate_answer(context, question)

    # Step 6 — Return answer and sources
    return {
        "answer": answer,
        "sources": list(set([c["source"] for c in chunks]))
    }
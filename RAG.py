from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
import ollama

# Load existing ChromaDB
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

vector_store = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings
)

print("✅ ChromaDB loaded successfully!")

# Query
query = input("\nEnter a test question: ")

# Retrieve relevant chunks
retriever = vector_store.as_retriever()
relevant_docs = retriever.invoke(query)

# Build context
context = "\n\n".join([doc.page_content for doc in relevant_docs])

# Build prompt
prompt = f"""
You are an enterprise employee assistant. Answer the question using ONLY the provided context from company documents.

Context:
{context}

Question:
{query}

If the answer is not in the context, say "I don't have that information in the company documents."
"""

# Get response from LLaMA 3
response = ollama.chat(
    model="llama3",
    messages=[
        {"role": "system", "content": "You are a helpful enterprise employee assistant."},
        {"role": "user", "content": prompt}
    ]
)

print("\n============================")
print("ANSWER")
print("============================\n")
print(response['message']['content'])
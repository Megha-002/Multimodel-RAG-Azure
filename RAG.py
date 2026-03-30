import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
import ollama

# Load embedding model
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# Load PDF
pdf_path = "data/mock_data.pdf"
loader = PyPDFLoader(pdf_path)
documents = loader.load()

print(f"\nTotal pages loaded: {len(documents)}")

# Split into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)
chunks = text_splitter.split_documents(documents)

print(f"Total chunks created: {len(chunks)}\n")

# Create ChromaDB vector store
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="chroma_db"
)

print("ChromaDB vector store created successfully!")

# Query
query = input("\nEnter a test question: ")

# Retrieve relevant chunks
retriever = vector_store.as_retriever()
relevant_docs = retriever.invoke(query)

# Build context
context = "\n\n".join([doc.page_content for doc in relevant_docs])

# Build prompt
prompt = f"""
You are an AI assistant. Answer the question using ONLY the provided context.

Context:
{context}

Question:
{query}

If the answer is not present in the context, say you don't know.
"""

# Get response from Ollama LLaMA 3
response = ollama.chat(
    model="llama3",
    messages=[
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": prompt}
    ]
)

print("\n============================")
print("FINAL RAG ANSWER")
print("============================\n")

print(response['message']['content'])

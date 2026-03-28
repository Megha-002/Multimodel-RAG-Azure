
import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

load_dotenv()

endpoint = os.getenv("AZURE_OPENAI_EMBD_ENDPOINT")
api_key = os.getenv("AZURE_OPENAI_EMBD_API_KEY")
api_version = os.getenv("AZURE_OPENAI_EMBD_API_VERSION")
embedding_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")

embeddings = AzureOpenAIEmbeddings(
    azure_endpoint=endpoint,
    api_key=api_key,
    api_version=api_version,
    deployment=embedding_deployment
)

OPENAI_KEY = os.getenv("AZURE_OPENAI_API_KEY")
OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
OPENAI_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")

llm = AzureChatOpenAI(
    azure_endpoint= OPENAI_ENDPOINT,
    api_key= OPENAI_KEY,
    api_version= OPENAI_VERSION,
    deployment_name= OPENAI_DEPLOYMENT
)

pdf_path = "data/mock_data.pdf"

loader = PyPDFLoader(pdf_path)
documents = loader.load()

print(f"\nTotal pages loaded: {len(documents)}")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = text_splitter.split_documents(documents)

print(f"Total chunks created: {len(chunks)}\n")

chunk_texts = [chunk.page_content for chunk in chunks]

vectors = embeddings.embed_documents(chunk_texts)

print(f"Total embeddings created: {len(vectors)}")
print(f"Vector dimension of first embedding: {len(vectors[0])}")

vector_store = FAISS.from_documents(chunks, embeddings)

print("\nFAISS vector store created successfully!")

query = input("\nEnter a test question: ")

retriever = vector_store.as_retriever()

relevant_docs = retriever.invoke(query)

# Combine retrieved chunks
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

# Get response from GPT
response = llm.invoke(prompt)

print("\n============================")
print("FINAL RAG ANSWER")
print("============================\n")

print(response.content)
# meghna
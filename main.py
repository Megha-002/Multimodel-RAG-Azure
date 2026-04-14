import os
import time
import warnings
import shutil
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import FakeEmbeddings

from groq import Groq
import mlflow
from prometheus_fastapi_instrumentator import Instrumentator

warnings.filterwarnings("ignore")
load_dotenv()

# ===============================
# MLflow Setup
# ===============================
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("enterprise-rag-chatbot")

# ===============================
# FastAPI App
# ===============================
app = FastAPI(title="Enterprise RAG Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Instrumentator().instrument(app).expose(app)

# ===============================
# LIGHTWEIGHT EMBEDDINGS
# ===============================
embeddings = FakeEmbeddings(size=384)

# ===============================
# LAZY LOAD VECTOR STORE
# ===============================
vector_store = None

def get_vector_store():
    global vector_store
    if vector_store is None:
        print("🔄 Loading ChromaDB...")
        vector_store = Chroma(
            persist_directory="chroma_db",
            embedding_function=embeddings
        )
        print("✅ ChromaDB loaded!")
    return vector_store

# ===============================
# GROQ CLIENT
# ===============================
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ===============================
# LLM HELPER
# ===============================
def get_llm_response(prompt):
    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "You are a helpful enterprise employee assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

# ===============================
# HEALTH CHECK
# ===============================
@app.get("/health")
def health():
    return {"status": "running"}

# ===============================
# QUERY ENDPOINT
# ===============================
class QueryRequest(BaseModel):
    question: str

@app.post("/query")
def query(request: QueryRequest):
    start_time = time.time()

    with mlflow.start_run():
        mlflow.log_param("model", "llama-3.1-8b-instant-groq")
        mlflow.log_param("embedding_model", "fake-embeddings")
        mlflow.log_param("vector_db", "chromadb")
        mlflow.log_param("question", request.question)

        retriever = get_vector_store().as_retriever()
        relevant_docs = retriever.invoke(request.question)

        context = "\n\n".join([doc.page_content for doc in relevant_docs])

        prompt = f"""
You are an enterprise assistant. Answer ONLY from context.

Context:
{context}

Question:
{request.question}

If not found, say:
"I don't have that information in the company documents."
"""

        answer = get_llm_response(prompt)

        latency = round(time.time() - start_time, 2)

        mlflow.log_metric("latency_seconds", latency)
        mlflow.log_metric("chunks_retrieved", len(relevant_docs))

    return {
        "question": request.question,
        "answer": answer,
        "latency_seconds": latency,
        "sources": [doc.metadata.get("source", "unknown") for doc in relevant_docs]
    }

# ===============================
# UPLOAD ENDPOINT
# ===============================
@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    save_path = f"data/{file.filename}"

    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    os.system("python Ingest.py")

    return {"message": f"{file.filename} uploaded and ingested successfully"}
import os
import time
import warnings
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from groq import Groq
import mlflow
from prometheus_fastapi_instrumentator import Instrumentator
from fastapi.middleware.cors import CORSMiddleware

warnings.filterwarnings("ignore")
load_dotenv()

print("🚀 Starting FastAPI app...")

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("enterprise-rag-chatbot")

app = FastAPI(title="Enterprise RAG Chatbot API")

# ===============================
# CORS
# ===============================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Instrumentator().instrument(app).expose(app)

# ===============================
# LOAD CHROMADB SAFELY
# ===============================
print("Loading ChromaDB...")

# Force absolute paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")
DB_FILE = os.path.join(CHROMA_PATH, "chroma.sqlite3")

# Stick to local SentenceTransformers
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# Check for the actual file, not the folder Docker created
if os.path.exists(DB_FILE):
    vector_store = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )
    print(f"✅ ChromaDB safely loaded from: {CHROMA_PATH}")
else:
    print(f"⚠️ DANGER: SQLite file not found at {DB_FILE}. Creating a blank one...")
    vector_store = Chroma(
        embedding_function=embeddings,
        persist_directory=CHROMA_PATH
    )

# ===============================
# GROQ CLIENT
# ===============================
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ===============================
# LLM FUNCTION
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
# HEALTH
# ===============================
@app.get("/health")
def health():
    return {"status": "running"}

# ===============================
# QUERY
# ===============================
class QueryRequest(BaseModel):
    question: str

@app.post("/query")
def query(request: QueryRequest):
    start_time = time.time()

    with mlflow.start_run():
        retriever = vector_store.as_retriever()
        relevant_docs = retriever.invoke(request.question)
        context = "\n\n".join([doc.page_content for doc in relevant_docs])

        prompt = f"""
You are an enterprise assistant. Answer ONLY from context.

Context:
{context}

Question:
{request.question}
"""

        answer = get_llm_response(prompt)
        latency = round(time.time() - start_time, 2)

    return {
        "question": request.question,
        "answer": answer,
        "latency_seconds": latency,
        "sources": [doc.metadata.get("source", "unknown") for doc in relevant_docs]
    }
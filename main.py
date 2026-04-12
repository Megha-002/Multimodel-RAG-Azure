import os
import time
import warnings
import shutil
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
import ollama
import mlflow
from prometheus_fastapi_instrumentator import Instrumentator

warnings.filterwarnings("ignore")
load_dotenv()

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("enterprise-rag-chatbot")

app = FastAPI(title="Enterprise RAG Chatbot API")
Instrumentator().instrument(app).expose(app)

# ===============================
# LOAD CHROMADB ONCE AT STARTUP
# ===============================

print("Loading ChromaDB...")
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings
)
print("✅ ChromaDB loaded!")

# ===============================
# LLM HELPER FUNCTION
# ===============================

def get_llm_response(prompt):
    use_groq = os.getenv("USE_GROQ", "false").lower() == "true"

    if use_groq:
        from groq import Groq
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {"role": "system", "content": "You are a helpful enterprise employee assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content
    else:
        response = ollama.chat(
            model="llama3",
            messages=[
                {"role": "system", "content": "You are a helpful enterprise employee assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        return response['message']['content']

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

    with mlflow.start_run():
        mlflow.log_param("model", "llama3")
        mlflow.log_param("embedding_model", "all-MiniLM-L6-v2")
        mlflow.log_param("vector_db", "chromadb")
        mlflow.log_param("question", request.question)

        retriever = vector_store.as_retriever()
        relevant_docs = retriever.invoke(request.question)
        context = "\n\n".join([doc.page_content for doc in relevant_docs])

        prompt = f"""
You are an enterprise employee assistant. Answer using ONLY the context from company documents.

Context:
{context}

Question:
{request.question}

If the answer is not in the context, say "I don't have that information in the company documents."
"""

        answer = get_llm_response(prompt)
        latency = round(time.time() - start_time, 2)
        chunks_retrieved = len(relevant_docs)

        mlflow.log_metric("latency_seconds", latency)
        mlflow.log_metric("chunks_retrieved", chunks_retrieved)

    return {
        "question": request.question,
        "answer": answer,
        "latency_seconds": latency,
        "sources": [doc.metadata.get("source", "unknown") for doc in relevant_docs]
    }

# ===============================
# SPEECH QUERY ENDPOINT
# ===============================

@app.post("/speech")
async def speech_query(file: UploadFile = File(...)):
    import whisper

    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    whisper_model = whisper.load_model("base")
    result = whisper_model.transcribe(temp_path)
    transcribed_text = result["text"]
    os.remove(temp_path)

    request = QueryRequest(question=transcribed_text)
    rag_result = query(request)

    return {**rag_result, "transcribed_text": transcribed_text}

# ===============================
# UPLOAD ENDPOINT
# ===============================

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    save_path = f"data/{file.filename}"
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    os.system("python Ingest.py")
    return {"message": f"{file.filename} uploaded and ingested successfully in the database."}
import time
import os
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

from schemas import QueryRequest, QueryResponse, STTResponse
from services.rag_service import run_rag
from services.groq_client import transcribe_audio
from config import MLFLOW_TRACKING_URI, DAGSHUB_USERNAME, DAGSHUB_TOKEN

import mlflow

# ── App Setup ──────────────────────────────────────────────
app = FastAPI(
    title="Multimodal RAG API",
    description="RAG pipeline using Groq, Hugging Face and Pinecone",
    version="1.0.0"
)

# ── CORS ───────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)


os.environ["MLFLOW_TRACKING_USERNAME"] = DAGSHUB_USERNAME
os.environ["MLFLOW_TRACKING_PASSWORD"] = DAGSHUB_TOKEN

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("multimodal-rag")



# ── Routes ─────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    start = time.time()
    result = run_rag(req.question)
    latency = round(time.time() - start, 3)

    with mlflow.start_run():
        mlflow.log_param("question", req.question[:100])
        mlflow.log_param("model", "llama-3.1-8b-instant")
        mlflow.log_metric("latency_seconds", latency)
        mlflow.log_metric("num_sources", len(result["sources"]))

    return result


@app.post("/stt", response_model=STTResponse)
async def speech_to_text(file: UploadFile = File(...)):
    audio_bytes = await file.read()
    transcript = transcribe_audio(audio_bytes, file.filename)
    return {"transcript": transcript}
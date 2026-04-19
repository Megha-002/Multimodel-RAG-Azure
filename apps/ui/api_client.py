import requests
import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")


def query_rag(question: str) -> dict:
    """Sends question to FastAPI /query endpoint and returns answer + sources."""
    try:
        r = requests.post(
            f"{API_BASE}/query",
            json={"question": question},
            timeout=30
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"answer": f"Error contacting API: {str(e)}", "sources": []}


def transcribe_voice(audio_bytes: bytes, filename: str) -> str:
    """Sends audio file to FastAPI /stt endpoint and returns transcript."""
    try:
        r = requests.post(
            f"{API_BASE}/stt",
            files={"file": (filename, audio_bytes, "audio/wav")},
            timeout=60
        )
        r.raise_for_status()
        return r.json()["transcript"]
    except Exception as e:
        return f"Error: {str(e)}"
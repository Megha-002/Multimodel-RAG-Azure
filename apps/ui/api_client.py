import requests
import os
import logging
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

logger = logging.getLogger(__name__)

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")


def query_rag(question: str) -> dict:
    """Sends question to FastAPI /query endpoint and returns answer + sources."""
    try:
        r = requests.post(
            f"{API_BASE}/query",
            json={"question": question},
            timeout=30,
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.error("query_rag failed: %s", e)
        return {"answer": f"Error contacting API: {str(e)}", "sources": []}


def transcribe_voice(audio_bytes: bytes, filename: str) -> str | None:
    """
    Sends audio bytes to FastAPI /stt endpoint and returns the transcript,
    or None if anything goes wrong (caller should show a warning).
    """
    try:
        r = requests.post(
            f"{API_BASE}/stt",
            files={"file": (filename, audio_bytes, "audio/wav")},
            timeout=60,
        )
        r.raise_for_status()
        data = r.json()
        transcript = data.get("transcript", "").strip()
        return transcript if transcript else None
    except requests.HTTPError as e:
        logger.error("STT HTTP error %s: %s", e.response.status_code, e.response.text)
        return None
    except Exception as e:
        logger.error("transcribe_voice failed: %s", e)
        return None
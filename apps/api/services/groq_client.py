import requests
from config import (
    GROQ_API_KEY,
    GROQ_LLM_MODEL,
    GROQ_WHISPER_MODEL,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS
)

GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_STT_URL = "https://api.groq.com/openai/v1/audio/transcriptions"


def generate_answer(context: str, question: str) -> str:
    """Sends retrieved context + user question to Groq LLM and returns the answer."""
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    prompt = f"""You are a helpful assistant. 
Answer the question based ONLY on the context below.
If the answer is not in the context, say 'I don't have enough information to answer this.'

Context:
{context}

Question: {question}

Answer:"""

    payload = {
        "model": GROQ_LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": LLM_TEMPERATURE,
        "max_tokens": LLM_MAX_TOKENS
    }

    r = requests.post(GROQ_CHAT_URL, headers=headers, json=payload, timeout=30)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def transcribe_audio(audio_bytes: bytes, filename: str = "audio.wav") -> str:
    """Sends audio bytes to Groq Whisper API and returns the transcript text."""
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
    files = {"file": (filename, audio_bytes, "audio/wav")}
    data = {"model": GROQ_WHISPER_MODEL, "response_format": "text"}

    r = requests.post(GROQ_STT_URL, headers=headers, files=files, data=data, timeout=60)
    r.raise_for_status()
    return r.text
import streamlit as st
import requests
import sounddevice as sd
import scipy.io.wavfile as wav
import tempfile
import os

# ===============================
# PAGE CONFIG
# ===============================

st.set_page_config(
    page_title="Enterprise RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

# ===============================
# CUSTOM CSS — GRADIENT THEME
# ===============================

st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        color: white;
    }
    .user-message {
        background: linear-gradient(135deg, #667eea, #764ba2);
        padding: 12px 18px;
        border-radius: 18px 18px 4px 18px;
        margin: 8px 0;
        max-width: 70%;
        margin-left: auto;
        color: white;
        font-size: 15px;
    }
    .bot-message {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border: 1px solid #667eea;
        padding: 12px 18px;
        border-radius: 18px 18px 18px 4px;
        margin: 8px 0;
        max-width: 70%;
        color: white;
        font-size: 15px;
    }
    .stTextInput > div > div > input {
        background: rgba(255,255,255,0.1);
        color: white;
        border: 1px solid #667eea;
        border-radius: 25px;
        padding: 10px 20px;
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 8px 24px;
        font-weight: bold;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #764ba2, #667eea);
        transform: scale(1.02);
    }
    .latency-badge {
        background: rgba(102,126,234,0.2);
        border: 1px solid #667eea;
        border-radius: 20px;
        padding: 4px 12px;
        font-size: 11px;
        color: #667eea;
        display: inline-block;
        margin-top: 6px;
    }
    .source-badge {
        background: rgba(56,239,125,0.1);
        border: 1px solid #38ef7d;
        border-radius: 20px;
        padding: 4px 12px;
        font-size: 11px;
        color: #38ef7d;
        display: inline-block;
        margin: 2px;
    }
</style>
""", unsafe_allow_html=True)

# ===============================
# SESSION STATE
# ===============================

if "messages" not in st.session_state:
    st.session_state.messages = []

# ===============================
# QUERY FASTAPI FUNCTION
# ===============================

def query_rag(question):
    try:
        response = requests.post(
            "http://localhost:8000/query",
            json={"question": question},
            timeout=120
        )
        return response.json()
    except Exception as e:
        return {"error": str(e)}

# ===============================
# RECORD VOICE FUNCTION
# ===============================

def record_voice():
    DURATION = 5
    SAMPLE_RATE = 16000
    st.info("🎤 Recording for 5 seconds... Speak now!")
    import sounddevice as sd
    import numpy as np
    audio = sd.rec(
        int(DURATION * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype='int16'
    )
    sd.wait()
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    wav.write(temp_file.name, SAMPLE_RATE, audio)
    return temp_file.name

# ===============================
# MAIN UI LAYOUT
# ===============================

st.markdown("""
<div style='text-align: center; padding: 20px 0'>
    <h1 style='background: linear-gradient(135deg, #667eea, #764ba2, #38ef7d);
               -webkit-background-clip: text;
               -webkit-text-fill-color: transparent;
               font-size: 2.5em;
               font-weight: 800;'>
        🤖 Enterprise RAG Chatbot
    </h1>
    <p style='color: rgba(255,255,255,0.5); font-size: 14px;'>
        Powered by LLaMA 3 · ChromaDB · MiniLM
    </p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ System Info")
    st.markdown("**LLM:** LLaMA 3 (Ollama)")
    st.markdown("**Embeddings:** MiniLM-L6-v2")
    st.markdown("**Vector DB:** ChromaDB")
    st.markdown("**Speech:** Whisper Base")
    st.markdown("---")
    st.markdown("### 📁 Knowledge Base")
    st.markdown("- HR Policy PDFs")
    st.markdown("- Finance Documents")
    st.markdown("- Scanned Invoices")
    st.markdown("- Process Diagrams")
    st.markdown("---")
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

st.markdown("---")

# Chat history display
chat_container = st.container()
with chat_container:
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(
                f'<div class="user-message">👤 {msg["content"]}</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="bot-message">🤖 {msg["content"]}</div>',
                unsafe_allow_html=True
            )
            if "latency" in msg:
                st.markdown(
                    f'<div class="latency-badge">⏱️ {msg["latency"]}s</div>',
                    unsafe_allow_html=True
                )
            if "sources" in msg:
                for source in set(msg["sources"]):
                    st.markdown(
                        f'<div class="source-badge">📎 {os.path.basename(source)}</div>',
                        unsafe_allow_html=True
                    )

st.markdown("---")

# Input area
col1, col2, col3 = st.columns([6, 1, 1])

with col1:
    user_input = st.text_input(
        "Ask anything...",
        placeholder="e.g. What is the leave policy?",
        label_visibility="collapsed"
    )

with col2:
    send_clicked = st.button("Send 🚀")

with col3:
    voice_clicked = st.button("🎤 Voice")

# ===============================
# HANDLE TEXT QUERY
# ===============================

if send_clicked and user_input:
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    with st.spinner("Thinking..."):
        result = query_rag(user_input)

    if "error" in result:
        st.error(f"Error: {result['error']}")
    else:
        st.session_state.messages.append({
            "role": "assistant",
            "content": result.get("answer", "No answer returned"),
            "latency": result.get("latency_seconds"),
            "sources": result.get("sources", [])
        })

    st.rerun()

# ===============================
# HANDLE VOICE QUERY
# ===============================

if voice_clicked:
    audio_path = record_voice()

    with open(audio_path, "rb") as f:
        response = requests.post(
            "http://localhost:8000/speech",
            files={"file": ("audio.wav", f, "audio/wav")}
        )

    os.remove(audio_path)

    if response.status_code == 200:
        data = response.json()
        transcribed = data.get("transcribed_text", "")

        st.session_state.messages.append({
            "role": "user",
            "content": f"🎤 {transcribed}"
        })

        st.session_state.messages.append({
            "role": "assistant",
            "content": data.get("answer", "No answer returned"),
            "latency": data.get("latency_seconds"),
            "sources": data.get("sources", [])
        })

        st.rerun()
    else:
        st.error("Voice query failed. Make sure FastAPI is running.")